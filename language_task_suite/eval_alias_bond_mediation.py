"""Test whether a complete attention-output bond mediates alias edits.

Patch the bond input from an original prefix into its edited counterpart. This
is a location gate for later low-rank or sparse basis experiments, not itself
a decomposition.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_surface_alias_edits import build_pairs
from surface_alias_mixture import AliasMixtureLanguage


KINDS = ('different_referent', 'definition_referent',
         'irrelevant_definition', 'definition_order', 'alias_label_swap')


@torch.no_grad()
def run_batch(model, bond, original, edited, scope, device):
    x0 = torch.tensor(original, dtype=torch.long, device=device)
    x1 = torch.tensor(edited, dtype=torch.long, device=device)
    captured = []
    def capture(_, inputs):
        captured.append(inputs[0].detach().clone())
    hook = bond.register_forward_pre_hook(capture)
    p0 = F.softmax(model(x0)[0][:, -1].float(), -1)
    hook.remove()
    p1 = F.softmax(model(x1)[0][:, -1].float(), -1)
    source = captured[0]
    def patch(_, inputs):
        x = inputs[0]
        if scope == 'all':
            return (source,)
        out = x.clone()
        out[:, -1] = source[:, -1]
        return (out,)
    hook = bond.register_forward_pre_hook(patch)
    pp = F.softmax(model(x1)[0][:, -1].float(), -1)
    hook.remove()
    return p0.cpu().numpy(), p1.cpu().numpy(), pp.cpu().numpy()


def select(pairs, per_kind, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for kind in KINDS:
        available = [i for i, p in enumerate(pairs)
                     if p['kind'] == kind and p['first_post_definition_use']]
        if not available:
            continue
        indices.extend(rng.choice(available, min(per_kind, len(available)),
                                  replace=False).tolist())
    return [pairs[i] for i in indices]


def score(pairs, p0, p1, pp, gen):
    out = {}
    for kind in KINDS:
        ix = [i for i, pair in enumerate(pairs) if pair['kind'] == kind]
        if not ix:
            out[kind] = dict(n=0)
            continue
        oracle = np.zeros_like(p0[ix])
        for j, i in enumerate(ix):
            for token, prob in pairs[i]['oracle_after'].items():
                oracle[j, gen.token2id[token]] += prob
            for token, prob in pairs[i]['oracle_before'].items():
                oracle[j, gen.token2id[token]] -= prob
        effect = p1[ix]-p0[ix]
        patched = pp[ix]-p0[ix]
        effect_l1 = np.abs(effect).sum(-1)
        patched_l1 = np.abs(patched).sum(-1)
        out[kind] = dict(n=len(ix),
                         pair_holdout_n=sum(pairs[i]['current_pair_holdout']
                                            for i in ix),
                         oracle_tv=float(np.abs(oracle).sum(-1).mean()/2),
                         model_tv=float(effect_l1.mean()/2),
                         patched_tv_from_original=float(patched_l1.mean()/2),
                         patched_tv_from_edited=float(np.abs(pp[ix]-p1[ix]).sum(-1).mean()/2),
                         aggregate_mediation=float(1-patched_l1.mean()/max(effect_l1.mean(), 1e-8)),
                         model_effect_l1_error=float(np.abs(effect-oracle).sum(-1).mean()),
                         patched_effect_l1_error=float(np.abs(patched-oracle).sum(-1).mean()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--layer', type=int, default=1)
    p.add_argument('--scope', choices=('last', 'all'), default='last')
    p.add_argument('--nstreams', type=int, default=1024)
    p.add_argument('--length', type=int, default=63)
    p.add_argument('--per-kind', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261108)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config

    torch.set_num_threads(4)
    saved = json.loads((a.model_dir/'results.json').read_text())
    record = next(row for row in saved['records'] if row['depth'] == 2)
    device = torch.device(a.device)
    model = BilinearGPT(Config(**record['model_config'])).to(device)
    model.load_state_dict(torch.load(a.model_dir/'model_depth2.pt',
                                     map_location=device, weights_only=True))
    model.eval()
    gen = AliasMixtureLanguage(seed=a.seed, initial_alias_definitions=2,
                               parenthetical_rate=0, definition_start_rate=0,
                               alias_use=.9, alias_reuse_rate=.8)
    pairs = select(build_pairs(gen, a.nstreams, a.length, 'balanced'),
                   a.per_kind, a.seed+1)
    bond = model.blocks[a.layer].attn.out
    groups = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[len(pair['original'])].append(i)
    p0 = np.zeros((len(pairs), len(gen.vocab)), dtype=np.float32)
    p1, pp = p0.copy(), p0.copy()
    for _, indices in groups.items():
        for start in range(0, len(indices), 64):
            ix = indices[start:start+64]
            q0, q1, qp = run_batch(model, bond,
                                   [pairs[i]['original'] for i in ix],
                                   [pairs[i]['edited'] for i in ix],
                                   a.scope, device)
            p0[ix], p1[ix], pp[ix] = q0, q1, qp
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                          'output': str(a.output)},
                  scores=score(pairs, p0, p1, pp, gen),
                  caveat='Full-bond activation patch from original into edited '
                         'prefix. Mediation is computed on the model effect, '
                         'not proof of oracle fidelity or a unique circuit. '
                         'Only first post-definition uses are sampled.')
    holdout_ix = [i for i, pair in enumerate(pairs)
                  if pair['current_pair_holdout']]
    result['pair_holdout_scores'] = score(
        [pairs[i] for i in holdout_ix], p0[holdout_ix], p1[holdout_ix],
        pp[holdout_ix], gen)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print({kind: (row['n'], round(row.get('model_tv', 0), 3),
                        round(row.get('aggregate_mediation', 0), 3))
           for kind, row in result['scores'].items()})


if __name__ == '__main__':
    main()
