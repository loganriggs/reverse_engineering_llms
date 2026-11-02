"""Shared layer-0 token-score bases on the joint verb/object/bracket task.

Compare data-free uniform weighting with task-blind unlabeled token marginals
and a no-bracket marginal control. Replay the native downstream layers and
score both exact-oracle predictions and grammar-valid causal effects.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval_multi_genre_token_tensor import forward_from_table, score_tables
from eval_nested_shared_token_basis import (
    project_tables, projection_check, shared_bases, token_marginal)
from eval_surface_mixture_edits import score
from eval_surface_mixture_v2_edits import build_pairs as verb_pairs
from eval_surface_mixture_v2_nested_edits import (
    build_pairs as bracket_pairs, probabilities, summarize)
from eval_surface_mixture_v3_edits import build_object_pairs
from surface_mixture_v3 import SurfaceMixtureV3
from surface_mixture_v3_nested_varlen import SurfaceMixtureV3NestedVarlen
from train_surface_mixture import make_eval


@torch.no_grad()
def table_probabilities(model, prefixes, tables, downstream, vocab_size,
                        device):
    result = np.empty((len(prefixes), vocab_size), dtype=np.float32)
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    for indices in groups.values():
        for start in range(0, len(indices), 32):
            selected = indices[start:start+32]
            x = torch.tensor([prefixes[i] for i in selected],
                             dtype=torch.long, device=device)
            logits = forward_from_table(model, x, tables, downstream)[:, -1]
            result[selected] = F.softmax(logits.float(), -1).cpu().numpy()
    return result


@torch.no_grad()
def prediction_scores(model, data, tables=None, downstream=None,
                      native_logp=None):
    if tables is None:
        logits = torch.cat([model(batch)[0].float()
                            for batch in data['x'].split(32)])
    else:
        logits = torch.cat([forward_from_table(model, batch, tables,
                                               downstream).float()
                            for batch in data['x'].split(32)])
    logp = F.log_softmax(logits, -1)
    q = data['q']
    q_kl = (q*(q.clamp_min(1e-30).log()-logp)).sum(-1)
    native_kl = None
    if native_logp is not None:
        native_p = native_logp.exp()
        native_kl = (native_p*(native_logp-logp)).sum(-1)
    masks = dict(all=torch.ones_like(data['verb']), verb=data['verb'],
                 object=data['obj'], inner=data['nested_inner'],
                 outer=data['nested_outer'])
    scores = {name: dict(n=int(mask.sum()), oracle_kl=float(q_kl[mask].mean()),
                         teacher_kl=float(native_kl[mask].mean())
                         if native_kl is not None else 0.)
              for name, mask in masks.items() if mask.any()}
    return scores, logp


@torch.no_grad()
def edit_scores(model, gen, pairs, tables=None, downstream=None,
                device='cuda'):
    if tables is None:
        before = probabilities(model, [p['original'] for p in pairs], device)
        after = probabilities(model, [p['edited'] for p in pairs], device)
    else:
        before = table_probabilities(model, [p['original'] for p in pairs],
                                     tables, downstream, len(gen.vocab), device)
        after = table_probabilities(model, [p['edited'] for p in pairs],
                                    tables, downstream, len(gen.vocab), device)
    return summarize(pairs, before, after) if 'answer_before' in pairs[0] else (
        score(pairs, before, after, gen))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--nfit', type=int, default=2048)
    parser.add_argument('--ntest', type=int, default=128)
    parser.add_argument('--nedit-streams', type=int, default=64)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--ranks', default='8,16')
    parser.add_argument('--smoothing', type=float, default=16.)
    parser.add_argument('--fit-seed', type=int, default=20261110)
    parser.add_argument('--test-seed', type=int, default=20261111)
    parser.add_argument('--edit-seed', type=int, default=20261112)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    metadata = json.loads((args.model_dir/'results.json').read_text())
    assert metadata['config']['generator'] == 'v3_nested_varlen'
    kwargs = metadata['generator_kwargs']
    sys.path.insert(0, args.source)
    from model import BilinearGPT, Config
    record = next(row for row in metadata['records'] if row['depth'] == args.depth)
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(args.model_dir/f'model_depth{args.depth}.pt',
                                     map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    # Rotary.inv_freq is an ordinary tensor, not a registered model buffer.
    model.blocks[0].attn.rotary.inv_freq = (
        model.blocks[0].attn.rotary.inv_freq.to(device))
    assert args.length <= model.config.seq_len
    check = projection_check(device)
    joint_fit = SurfaceMixtureV3NestedVarlen(seed=args.fit_seed, **kwargs)
    no_bracket_fit = SurfaceMixtureV3(seed=args.fit_seed,
        object_vocab_size=kwargs['object_vocab_size'])
    vocab = joint_fit.vocab
    assert len(vocab) == model.config.vocab_size
    joint_weight, joint_stats = token_marginal(
        joint_fit, args.nfit, args.length, vocab, args.smoothing, device)
    no_bracket_weight, no_bracket_stats = token_marginal(
        no_bracket_fit, args.nfit, args.length, vocab, args.smoothing, device)
    weights = dict(uniform=torch.full((len(vocab),), 1/len(vocab), device=device),
                   joint_unlabeled=joint_weight,
                   no_bracket_unlabeled=no_bracket_weight)
    test = make_eval(SurfaceMixtureV3NestedVarlen(seed=args.test_seed, **kwargs),
                     args.ntest, args.length, device)
    pair_sets = {}
    for name, builder, offset, length in (
            ('verb', verb_pairs, 0, args.length-1),
            ('object', build_object_pairs, 1, args.length-1),
            ('bracket', bracket_pairs, 2, args.length)):
        gen = SurfaceMixtureV3NestedVarlen(seed=args.edit_seed+offset, **kwargs)
        pair_sets[name] = (gen, builder(gen, args.nedit_streams, length))
        assert pair_sets[name][1]
    tables, downstream = score_tables(model, args.length)
    native_scores, native_logp = prediction_scores(model, test)
    native_edits = {name: edit_scores(model, gen, pairs, device=device)
                    for name, (gen, pairs) in pair_sets.items()}
    full_scores, _ = prediction_scores(model, test, tables, downstream,
                                       native_logp)
    full_edits = {name: edit_scores(model, gen, pairs, tables, downstream,
                                   device=device)
                  for name, (gen, pairs) in pair_sets.items()}
    assert full_scores['all']['teacher_kl'] < 1e-4
    result = dict(config={**vars(args), 'model_dir':str(args.model_dir),
                          'output':str(args.output)},
                  projection_check=check,
                  fit=dict(joint=joint_stats,no_bracket=no_bracket_stats,
                           marginal='smoothed task-blind token frequency'),
                  pair_counts={name:dict(Counter(p['kind'] for p in pairs))
                               for name,(_,pairs) in pair_sets.items()},
                  native=dict(prediction=native_scores,edits=native_edits),
                  full_table=dict(prediction=full_scores,edits=full_edits),
                  rows=[],
                  caveat='Only the first-layer token-score table is '
                         'projected; all downstream model weights remain '
                         'native. Marginal fitting uses separate unlabeled '
                         'streams; evaluation uses independent streams and '
                         'valid edits. The model itself was trained on all '
                         'three tasks, so no new-task discovery is claimed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    save()
    for method, weight in weights.items():
        bases = shared_bases(tables, weight)
        for rank in (int(value) for value in args.ranks.split(',')):
            compressed = project_tables(tables, weight, bases, rank)
            prediction, _ = prediction_scores(model, test, compressed,
                                              downstream, native_logp)
            edits = {name:edit_scores(model,gen,pairs,compressed,downstream,
                                     device=device)
                     for name,(gen,pairs) in pair_sets.items()}
            result['rows'].append(dict(method=method,rank=rank,
                                       prediction=prediction,edits=edits))
            save()
            print(method, rank,
                  'teacher KL', round(prediction['all']['teacher_kl'], 5),
                  'oracle KL verb/object',
                  round(prediction['verb']['oracle_kl'], 4),
                  round(prediction['object']['oracle_kl'], 4),
                  'bracket outer L1',
                  round(edits['bracket']['outer_relevant']['effect_l1'], 4),
                  flush=True)


if __name__ == '__main__':
    main()
