"""Compare local attention-output bond bases on the original mixed-rule DGP.

Fit bases on one set of generated sequences and evaluate truncated bonds on
another. A full-width orthogonal basis must leave logits unchanged. This is
local gauge/compression evidence, not full-function canonicalization.
"""

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_existing_mixed_oracle import candidate_info
from mixed_rule_oracle import observable_distributions_for_sequence


SOURCE = '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits'


def make_data(gen, n, length, device):
    ids, _ = gen.sample_batch(n, length)
    x = torch.tensor([row[:-1] for row in ids], dtype=torch.long, device=device)
    return ids, x


def oracle_and_groups(gen, ids):
    n, length = len(ids), len(ids[0])
    q = np.zeros((n, length-1, len(gen.vocab)), dtype=np.float32)
    conflict = np.zeros((n, length-1), dtype=bool)
    for b, row in enumerate(ids):
        tokens = [gen.id2token[i] for i in row]
        distributions = observable_distributions_for_sequence(gen, tokens)
        for t in range(1, length):
            for token, mass in distributions[t].items():
                q[b, t-1, gen.token2id[token]] = mass
            conflict[b, t-1] = candidate_info(gen, tokens[:t])[1]
    return q, conflict


def bases_for(activations, weight, seed):
    d = activations.shape[-1]
    a = activations.reshape(-1, d).double()
    gram = a.T @ a / len(a)
    _, u = torch.linalg.eigh(gram)
    activation_pca = torch.flip(u, dims=(1,)).float()
    _, _, vh = torch.linalg.svd(weight.double(), full_matrices=True)
    weight_svd = vh.T.float()
    rng = np.random.default_rng(seed)
    random, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return {'physical': torch.eye(d), 'random': torch.from_numpy(random).float(),
            'activation_pca': activation_pca, 'weight_svd': weight_svd}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/models/bilinear_gpt_all_2L_4H_d64_10000steps.pt')
    p.add_argument('--config', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/configs/bilinear_gpt_all_2L_4H_d64_10000steps.json')
    p.add_argument('--source', default=SOURCE)
    p.add_argument('--nfit', type=int, default=128)
    p.add_argument('--ntest', type=int, default=128)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--fit-contexts', choices=['all', 'no_conflict'], default='all')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from generator import LanguageGenerator, ALL_RULES, DEFAULT_MIXING
    from model import BilinearGPT, Config

    config_json = json.loads(Path(a.config).read_text())
    config_keys = {f.name for f in fields(Config)}
    config = Config(**{k: v for k, v in config_json.items() if k in config_keys})
    assert a.length <= config.seq_len and 0 <= a.layer < config.n_layer
    torch.set_num_threads(4)
    device = torch.device(a.device)
    model = BilinearGPT(config).to(device)
    model.load_state_dict(torch.load(a.checkpoint, weights_only=True, map_location='cpu'))
    model.eval()
    bond = model.blocks[a.layer].attn.out
    fit_gen = LanguageGenerator(ALL_RULES, mixing_weights=DEFAULT_MIXING,
                                mode='mixed', seed=4701)
    test_gen = LanguageGenerator(ALL_RULES, mixing_weights=DEFAULT_MIXING,
                                 mode='mixed', seed=4702)
    fit_ids, xfit = make_data(fit_gen, a.nfit, a.length, device)
    test_ids, xtest = make_data(test_gen, a.ntest, a.length, device)
    saved = []

    def collect(_, inputs):
        saved.append(inputs[0].detach().cpu())

    handle = bond.register_forward_pre_hook(collect)
    with torch.no_grad():
        for batch in xfit.split(32):
            model(batch)
    handle.remove()
    fit_activation = torch.cat(saved)
    if a.fit_contexts == 'no_conflict':
        fit_conflicts = np.zeros((a.nfit, a.length-1), dtype=bool)
        for b, ids in enumerate(fit_ids):
            tokens = [fit_gen.id2token[i] for i in ids]
            for t in range(1, a.length):
                fit_conflicts[b, t-1] = candidate_info(fit_gen, tokens[:t])[1]
        fit_activation = fit_activation[torch.from_numpy(~fit_conflicts)]
    bases = bases_for(fit_activation, bond.weight.detach().cpu(), 4703)
    q_np, conflicts_np = oracle_and_groups(test_gen, test_ids)
    q = torch.from_numpy(q_np).to(device)
    entropy = -(q * q.clamp_min(1e-30).log()).sum(-1)
    conflicts = torch.from_numpy(conflicts_np).to(device)
    with torch.no_grad():
        baseline_logits = torch.cat([model(batch)[0] for batch in xtest.split(32)])
    baseline_logp = F.log_softmax(baseline_logits.float(), dim=-1)
    original_bond = []
    handle = bond.register_forward_pre_hook(collect)
    saved = original_bond
    with torch.no_grad():
        for batch in xtest.split(32):
            model(batch)
    handle.remove()
    original_bond = torch.cat(original_bond).to(device)
    rows = []
    widths = sorted(set([4, 8, 16, 32, 48, config.n_embd]))
    for name, basis_cpu in bases.items():
        u = basis_cpu.to(device)
        ortho_error = torch.linalg.matrix_norm(u.T @ u - torch.eye(u.shape[1], device=device)).item()
        assert ortho_error < 1e-4, (name, ortho_error)
        for k in widths:
            projector = u[:, :k]
            def truncate(_, inputs):
                x = inputs[0]
                return ((x @ projector) @ projector.T,)
            handle = bond.register_forward_pre_hook(truncate)
            with torch.no_grad():
                logits = torch.cat([model(batch)[0] for batch in xtest.split(32)])
            handle.remove()
            if k == config.n_embd:
                assert torch.max(torch.abs(logits-baseline_logits)).item() < 3e-4
            logp = F.log_softmax(logits.float(), dim=-1)
            kl = -(q*logp).sum(-1)-entropy
            delta_p = (baseline_logp.exp() * (baseline_logp-logp)).sum(-1)
            residual = original_bond - ((original_bond @ projector) @ projector.T)
            for group_name, mask in [('all', torch.ones_like(conflicts)),
                                     ('conflict', conflicts),
                                     ('no_conflict', ~conflicts)]:
                rows.append(dict(basis=name, width=k, group=group_name,
                                 n=int(mask.sum().item()),
                                 oracle_kl=float(kl[mask].mean()),
                                 baseline_kl=float((-(q*baseline_logp).sum(-1)-entropy)[mask].mean()),
                                 baseline_to_truncated_kl=float(delta_p[mask].mean()),
                                 bond_mse=float(residual[mask].square().mean())))
    output = dict(config=vars(a), widths=widths, rows=rows,
                  caveat='Local attention-output bond only. Full width is an exact gauge; '
                         'truncation is a causal intervention. No claim of full-function '
                         'canonicalization or rule-specific mechanism recovery.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    for row in rows:
        if row['group'] == 'conflict' and row['width'] in (16, 32):
            print(row)


if __name__ == '__main__':
    main()
