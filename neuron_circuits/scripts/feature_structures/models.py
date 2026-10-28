"""Shared support, implication gates, additive refinements, and computation DAGs.

Hard forward gates have exact set semantics. Straight-through gradients are a
surrogate, not derivatives of discrete selection. Fixed budgets and threshold
selection are distinct modes; expose actual hard costs in both modes.
"""
import torch
from torch import nn
from torch.nn import functional as F


def binary_gate(logits, budget=None, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    soft = torch.sigmoid(logits / temperature)
    if budget is None:
        hard = (logits > 0).to(logits.dtype)
    else:
        if not 0 <= budget <= logits.shape[-1]:
            raise ValueError('invalid gate budget')
        hard = torch.zeros_like(logits)
        if budget:
            hard.scatter_(-1, logits.topk(budget, dim=-1).indices, 1)
    return hard + (soft - soft.detach())


def validate_parents(parents):
    for child, ps in enumerate(parents):
        if len(set(ps)) != len(ps) or any(p < 0 or p >= child for p in ps):
            raise ValueError('parents must be unique earlier nodes; order ensures acyclicity')


def implication_closure(seeds, parents):
    """child => all parents, including multiple parents; never adds decoder vectors."""
    validate_parents(parents)
    if seeds.shape[-1] != len(parents):
        raise ValueError('gate/graph size mismatch')
    gates = list(seeds.unbind(-1))
    for child in range(len(parents) - 1, -1, -1):
        for parent in parents[child]:
            gates[parent] = 1 - (1 - gates[parent]) * (1 - gates[child])
    return torch.stack(gates, -1)


def union_support(gates, membership):
    """OR over selected group memberships: a shared atom is decoded exactly once."""
    if gates.shape[-1] != membership.shape[0]:
        raise ValueError('group dimension mismatch')
    return 1 - (1 - gates.unsqueeze(-1) * membership).prod(dim=-2)


class SharedGroups(nn.Module):
    def __init__(self, dimension, atoms, groups, group_budget, membership_budget,
                 parents=None, initial_membership=None):
        super().__init__()
        self.dimension, self.atoms, self.groups = dimension, atoms, groups
        self.group_budget, self.membership_budget = group_budget, membership_budget
        self.parents = tuple(tuple(p) for p in parents) if parents is not None else None
        if self.parents is not None:
            validate_parents(self.parents)
            if len(self.parents) != groups:
                raise ValueError('group/graph size mismatch')
        self.dictionary = nn.Parameter(torch.randn(dimension, atoms))
        self.coordinate_encoder = nn.Linear(dimension, atoms)
        self.group_encoder = nn.Linear(dimension, groups)
        self.membership_logits = nn.Parameter(torch.randn(groups, atoms) * .1)
        self.bias = nn.Parameter(torch.zeros(dimension))
        if initial_membership is not None:
            if initial_membership.shape != (groups, atoms):
                raise ValueError('membership shape mismatch')
            with torch.no_grad():
                self.membership_logits.copy_(initial_membership * 4 - 2)

    def dictionary_matrix(self):
        return F.normalize(self.dictionary, dim=0)

    def forward(self, x, gates=None, coordinates=None, membership=None, remove_atoms=()):
        seed = binary_gate(self.group_encoder(x), self.group_budget) if gates is None else gates
        s = implication_closure(seed, self.parents) if self.parents is not None else seed
        M = binary_gate(self.membership_logits, self.membership_budget) if membership is None else membership
        q = union_support(s, M)
        if remove_atoms:
            keep = torch.ones(self.atoms, device=x.device, dtype=x.dtype)
            keep[list(remove_atoms)] = 0
            q = q * keep
        a = self.coordinate_encoder(x) if coordinates is None else coordinates
        code = a * q
        reconstruction = code @ self.dictionary_matrix().T + self.bias
        # Exact nonzero forward count; surrogate gradient only through selected
        # support, not a claimed derivative of discrete L0 at zero coordinates.
        coordinate_count = (code.detach() != 0).to(code.dtype).sum(-1).mean()
        coordinate_count = coordinate_count + (q - q.detach()).sum(-1).mean()
        costs = dict(groups=s.sum(-1).mean(), coordinates=coordinate_count,
                     memberships=M.sum())
        return dict(reconstruction=reconstruction, code=code, support=q,
                    seed_gates=seed, gates=s, membership=M, costs=costs)

    def objective(self, x, result, group_cost=0., coordinate_cost=0., membership_cost=0.):
        c = result['costs']
        return F.mse_loss(result['reconstruction'], x) + group_cost*c['groups'] + coordinate_cost*c['coordinates'] + membership_cost*c['memberships']


class AdditiveRefinements(nn.Module):
    """Ancestors contribute base terms; children contribute separate refinements.

This is deliberately not shared-atom union decoding and does not enforce nested
subspaces or guarantee learned residual semantics merely through architecture.
"""
    def __init__(self, dimension, rank, parents):
        super().__init__(); validate_parents(parents)
        self.parents = tuple(tuple(p) for p in parents)
        self.dictionary = nn.Parameter(torch.randn(len(parents), dimension, rank))

    def forward(self, coordinates, seed_gates):
        gates = implication_closure(seed_gates, self.parents)
        decoded = torch.einsum('...gr,gdr->...gd', coordinates, F.normalize(self.dictionary, dim=1))
        return (decoded * gates.unsqueeze(-1)).sum(-2), gates


class ComputationDAG(nn.Module):
    """Ordered scalar nodes, linear and explicit parent-coordinate products.

Each node is evaluated once, then its value is reused by consumers. Parent-edge
removal affects every downstream monomial using that edge. Node removal affects
all consumers. Graph topology is explicit; node and edge gates may be learned.
Purely linear variants are not assumed identifiable from reconstruction.
"""
    def __init__(self, input_dimension, output_dimension, parents, products=None):
        super().__init__(); validate_parents(parents)
        self.parents = tuple(tuple(p) for p in parents)
        self.products = tuple(tuple(ps) for ps in (products or [() for _ in parents]))
        if len(self.products) != len(parents): raise ValueError('product/graph size mismatch')
        self.functions = nn.ModuleList()
        self.edge_logits = nn.ParameterList()
        for node, ps in enumerate(parents):
            if any(a not in ps or b not in ps for a,b in self.products[node]):
                raise ValueError('product operands must be parents')
            self.functions.append(nn.Linear(input_dimension + len(ps) + len(self.products[node]), 1))
            self.edge_logits.append(nn.Parameter(torch.ones(len(ps))*2))
        self.node_encoder = nn.Linear(input_dimension, len(parents))
        self.readout = nn.Linear(len(parents), output_dimension, bias=False)

    def forward(self, x, gates=None, edge_gates=None, remove_nodes=(), remove_edges=()):
        gates = binary_gate(self.node_encoder(x)) if gates is None else gates
        hs=[]
        for node, ps in enumerate(self.parents):
            edges = binary_gate(self.edge_logits[node]) if edge_gates is None else edge_gates[node]
            parent_values={}
            for k,parent in enumerate(ps):
                edge = edges[k] * (0. if (parent,node) in remove_edges else 1.)
                parent_values[parent] = hs[parent] * edge
            values=[x] + [parent_values[p] for p in ps]
            values += [parent_values[a]*parent_values[b] for a,b in self.products[node]]
            h=self.functions[node](torch.cat(values,-1))*gates[...,node,None]
            if node in remove_nodes: h=h*0
            hs.append(h)
        hidden=torch.cat(hs,-1)
        return dict(reconstruction=self.readout(hidden),hidden=hidden,gates=gates)
