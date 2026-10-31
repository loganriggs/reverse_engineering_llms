"""Learn a shared dictionary with sample-dependent overlapping group gates.

Only primitive and AND query activations fit the featurizer. XOR remains a
whole-family holdout. The decoder counts a shared atom once when multiple
active groups contain it. Flat top-four sparse autoencoders are controls.
"""

import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contextual_composition_reduction import FIT_QUERIES, exact_head, head_table
from dual_joint_function_canonical import HERE
from train_contextual_composition_attention import (
    ContextualCompositionAttention, FAMILIES, examples)


def hard_topk(values, k):
    mask = torch.zeros_like(values)
    return mask.scatter_(1, values.topk(k, dim=1).indices, 1.)


class FlatSparse(nn.Module):
    def __init__(self, atoms):
        super().__init__()
        self.encoder = nn.Parameter(torch.randn(atoms, 8)/8**.5)
        self.decoder = nn.Parameter(torch.randn(atoms, 8)/8**.5)

    def forward(self, h):
        a = h@self.encoder.T
        gate = hard_topk(a.abs(), 4)
        return (a*gate)@F.normalize(self.decoder, dim=1), dict(active=gate.sum(1),
                                                                membership=None)


class SharedGroupSparse(nn.Module):
    def __init__(self, atoms=12, groups=8):
        super().__init__()
        self.encoder = nn.Parameter(torch.randn(atoms, 8)/8**.5)
        self.decoder = nn.Parameter(torch.randn(atoms, 8)/8**.5)
        self.group_encoder = nn.Parameter(torch.randn(groups, 8)/8**.5)
        self.membership_logits = nn.Parameter(torch.randn(groups, atoms)*.8)

    def forward(self, h):
        a = h@self.encoder.T
        scaled = h/(h.square().mean(1, keepdim=True).sqrt()+1e-6)
        group_scores = scaled@self.group_encoder.T
        hard_groups = hard_topk(group_scores, 2)
        soft_groups = torch.sigmoid(group_scores)
        p = torch.sigmoid(self.membership_logits)
        hard_membership = (p>.5).float()
        hard_union = 1.-torch.prod(1.-hard_groups[:, :, None]*
                                    hard_membership[None], dim=1)
        soft_union = 1.-torch.prod(1.-soft_groups[:, :, None]*p[None], dim=1)
        union = hard_union+soft_union-soft_union.detach()
        # Top-four within the union gives exactly the same coordinate budget
        # as the flat baseline; a group cannot silently activate all atoms.
        selected = hard_topk(a.abs()*union.detach(), 4)
        gate = selected*union
        reconstruction = (a*gate)@F.normalize(self.decoder, dim=1)
        membership_count = hard_membership.sum()+p.sum()-p.sum().detach()
        return reconstruction, dict(active=gate.detach().sum(1),
                                    membership=membership_count,
                                    groups=hard_groups.detach())


def train_featurizer(model, train_h, val_h, steps, seed, membership_cost=0.):
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    rng = np.random.default_rng(seed+1)
    scale = train_h.square().mean().item()
    best = (float('inf'), None, None)
    for step in range(1, steps+1):
        idx = torch.from_numpy(rng.integers(0, len(train_h), 512))
        h = train_h[idx]
        recon, detail = model(h)
        error = (recon-h).square().mean()/scale
        penalty = (membership_cost*detail['membership']/model.membership_logits.numel()
                   if detail['membership'] is not None else 0.)
        loss = error+penalty
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        if step%100==0 or step==steps:
            model.eval()
            with torch.no_grad():
                pred, d = model(val_h)
                val_error = float((pred-val_h).square().mean()/scale)
                val_penalty = (membership_cost*float(d['membership'])/
                               model.membership_logits.numel()
                               if d['membership'] is not None else 0.)
                score = val_error+val_penalty
                if score<best[0]:
                    best = (score, {k:v.detach().clone()
                                    for k,v in model.state_dict().items()}, step)
            model.train()
    model.load_state_dict(best[1]);model.eval()
    return dict(selected_step=best[2], validation_objective=best[0])


def evaluate(featurizer, native, table, x, y, query):
    h = torch.from_numpy(exact_head(table, x)).float()
    with torch.no_grad():
        reconstructed, detail = featurizer(h)
        q = native.embed(x[:, -1])
        original = native.read(torch.cat((h, q), dim=1)).squeeze(-1)
        predicted = native.read(torch.cat((reconstructed, q), dim=1)).squeeze(-1)
    families = {}
    for family, ids in FAMILIES.items():
        mask = torch.from_numpy(np.isin(query, ids))
        families[family] = dict(n=int(mask.sum()),
            accuracy=float(((predicted[mask]>0)==y[mask].bool()).float().mean()),
            native_accuracy=float(((original[mask]>0)==y[mask].bool()).float().mean()),
            head_relative_mse=float((reconstructed[mask]-h[mask]).square().sum()/
                                    h[mask].square().sum()),
            relative_logit_mse=float((predicted[mask]-original[mask]).square().sum()/
                                     original[mask].square().sum()),
            mean_active_atoms=float(detail['active'][mask].mean()))
    return dict(families=families,
                membership_links=(float(detail['membership'])
                                  if detail['membership'] is not None else None))


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(5):
        native = ContextualCompositionAttention()
        native.load_state_dict(torch.load(HERE/f'contextual_composition_attention_seed{seed}.pt',
                                         map_location='cpu', weights_only=True))
        native.eval()
        table = head_table(native)
        train_x, _, train_q = examples(40000, 20262400+seed)
        val_x, _, val_q = examples(10000, 20262500+seed)
        train_h = torch.from_numpy(exact_head(table, train_x[
            np.isin(train_q, FIT_QUERIES)])).float()
        val_h = torch.from_numpy(exact_head(table, val_x[
            np.isin(val_q, FIT_QUERIES)])).float()
        test_x, test_y, test_q = examples(20000, 20262600+seed)
        p=np.arange(1,17,dtype=float);p/=p.sum()
        shifted_x, shifted_y, shifted_q=examples(20000,20262700+seed,
                                                  token_prob=p)
        torch.manual_seed(20262800+seed)
        candidates={
            'flat_12': (FlatSparse(12),0.),
            'flat_22': (FlatSparse(22),0.),
            'shared_groups_cost0.02': (SharedGroupSparse(),.02),
            'shared_groups_cost0.10': (SharedGroupSparse(),.10),
        }
        methods={}
        flat_reference_state=None
        for index,(name,(featurizer,cost)) in enumerate(candidates.items()):
            if name.startswith('shared'):
                assert flat_reference_state is not None
                featurizer.encoder.data.copy_(flat_reference_state['encoder'])
                featurizer.decoder.data.copy_(flat_reference_state['decoder'])
                # Start with genuine overlapping memberships instead of an
                # all-to-all graph, which is a degenerate flat autoencoder.
                generator=torch.Generator().manual_seed(20262900+seed*10+index)
                membership=torch.rand(featurizer.membership_logits.shape,
                                      generator=generator) < .65
                for atom in range(membership.shape[1]):
                    if membership[:, atom].sum()<3:
                        membership[:3, atom]=True
                featurizer.membership_logits.data.copy_(
                    torch.where(membership,torch.ones_like(featurizer.membership_logits),
                                -torch.ones_like(featurizer.membership_logits)))
            selected=train_featurizer(featurizer,train_h,val_h,1200,
                                      20262800+seed*100+index,cost)
            if name=='flat_12':
                flat_reference_state={k:v.detach().clone()
                                      for k,v in featurizer.state_dict().items()}
            methods[name]=dict(parameters=sum(p.numel() for p in featurizer.parameters()),
                               fit=selected,
                               test=evaluate(featurizer,native,table,test_x,test_y,test_q),
                               shifted=evaluate(featurizer,native,table,
                                                shifted_x,shifted_y,shifted_q))
            torch.save(featurizer.state_dict(),
                       HERE/f'contextual_composition_groups_seed{seed}_{name}.pt')
            print(seed,name,'xor',methods[name]['test']['families']['xor']['accuracy'],
                  'head',methods[name]['test']['families']['xor']['head_relative_mse'],
                  'links',methods[name]['test']['membership_links'],flush=True)
        rows.append(dict(seed=seed,methods=methods))
    summary={}
    for name in rows[0]['methods']:
        summary[name]={}
        for split in ('test','shifted'):
            summary[name][split]={}
            for family in FAMILIES:
                summary[name][split][family]={key:float(np.mean([
                    row['methods'][name][split]['families'][family][key]
                    for row in rows])) for key in ('accuracy','head_relative_mse',
                                                   'relative_logit_mse','mean_active_atoms')}
        summary[name]['mean_membership_links']=float(np.mean([
            row['methods'][name]['test']['membership_links'] for row in rows]))\
            if name.startswith('shared') else None
    record=dict(description='Learned overcomplete shared dictionary and sample-dependent overlapping group gates. Two active groups, union counts shared atoms once, top-four coordinate cap. Flat top-four sparse autoencoders with 12 or 22 atoms are controls. Fit and select on primitive+AND head activations only; six XOR queries completely withheld from featurizer fit and validation. Native model trained on all tasks. Five model seeds, independent test and token-frequency shift.',
                rows=rows,summary=summary)
    (HERE/'contextual_composition_shared_groups.json').write_text(
        json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':run()
