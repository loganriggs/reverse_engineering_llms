"""Causal opener and nuisance edits at nested-bracket closing sites."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from surface_mixture_v2 import EVENTS, PLACES, _frame, _indices, _split
from surface_mixture_v2_nested import (CLOSERS, OPENERS,
                                       SurfaceMixtureV2Nested)
from surface_mixture_v2_nested_varlen import SurfaceMixtureV2NestedVarlen


def open_stack(suffix):
    stack = []
    for i, token in enumerate(suffix):
        if token in OPENERS:
            stack.append((i, token))
        elif token in CLOSERS.values():
            assert stack and CLOSERS[stack[-1][1]] == token
            stack.pop()
    return stack


def build_pairs(gen, nstreams, length):
    pairs = []
    for _ in range(nstreams):
        ids, labels, _ = gen.sample_sequence(length)
        tokens = [gen.id2token[i] for i in ids]
        for t, label in enumerate(labels):
            if label not in ('nested_close_inner', 'nested_close_outer'):
                continue
            prefix = tokens[:t]
            _, clause = _split(prefix)
            suffix = clause[_indices(_frame(clause))[3]:]
            start = len(prefix)-len(suffix)
            assert suffix[0] in OPENERS
            stack = open_stack(suffix)
            if label == 'nested_close_inner':
                assert len(stack) == 2
                relevant = prefix.copy()
                inner_pos = stack[-1][0]
                relevant[start+inner_pos] = next(x for x in OPENERS
                                                  if x != suffix[inner_pos])
                control = prefix.copy()
                control[start] = next(x for x in OPENERS if x != suffix[0])
                lag = len(suffix)-inner_pos
                edits = [('inner_relevant', relevant, lag),
                         ('inner_outer_opener_control', control, lag)]
            else:
                assert len(stack) == 1
                relevant = prefix.copy()
                relevant[start] = next(x for x in OPENERS if x != suffix[0])
                control = prefix.copy()
                fillers = [(i+2, PLACES if suffix[i] == 'near' else EVENTS)
                           for i in range(1, len(suffix)-2)
                           if suffix[i] in ('near', 'during') and
                           suffix[i+1] == 'the']
                assert fillers
                filler_pos, filler = fillers[-1]
                control[start+filler_pos] = next(x for x in filler
                                                 if x != suffix[filler_pos])
                lag = len(suffix)
                edits = [('outer_relevant', relevant, lag),
                         ('outer_inner_filler_control', control, lag)]
            q0 = gen.next_distribution(prefix)
            assert len(q0) == 1
            for kind, changed, distance in edits:
                q1 = gen.next_distribution(changed)
                assert len(q1) == 1
                assert (q0 != q1) == ('relevant' in kind)
                pairs.append(dict(kind=kind, original=[gen.token2id[x]
                    for x in prefix], edited=[gen.token2id[x]
                    for x in changed], answer_before=gen.token2id[next(iter(q0))],
                    answer_after=gen.token2id[next(iter(q1))],
                    opener_distance=distance))
    return pairs


@torch.no_grad()
def probabilities(model, prefixes, device):
    output = np.zeros((len(prefixes), model.config.vocab_size), np.float32)
    groups = defaultdict(list)
    for i, prefix in enumerate(prefixes):
        groups[len(prefix)].append(i)
    for indices in groups.values():
        for start in range(0, len(indices), 64):
            ix = indices[start:start+64]
            x = torch.tensor([prefixes[i] for i in ix],
                             dtype=torch.long, device=device)
            output[ix] = F.softmax(model(x)[0][:, -1].float(), -1).cpu().numpy()
    return output


def summarize(pairs, before, after):
    by = defaultdict(list)
    for i, pair in enumerate(pairs):
        q0, q1 = pair['answer_before'], pair['answer_after']
        b, a = before[i], after[i]
        oracle = np.zeros_like(b)
        oracle[q1] += 1
        oracle[q0] -= 1
        row = dict(
            before_ce=-np.log(max(float(b[q0]), 1e-30)),
            after_ce=-np.log(max(float(a[q1]), 1e-30)),
            before_correct=int(b.argmax() == q0),
            after_correct=int(a.argmax() == q1),
            oracle_tv=.5*float(np.abs(oracle).sum()),
            model_tv=.5*float(np.abs(a-b).sum()),
            effect_l1=float(np.abs((a-b)-oracle).sum()),
            target_mass_shift=float(a[q1]-b[q1]) if q0 != q1 else 0.)
        by[pair['kind']].append(row)
        by[f"{pair['kind']}@lag{pair['opener_distance']}"].append(row)
    return {kind: dict(n=len(rows), **{key: float(np.mean([
        row[key] for row in rows])) for key in rows[0]})
        for kind, rows in by.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--depths', default='1,2,3,4')
    p.add_argument('--nstreams', type=int, default=512)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261201)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    torch.set_num_threads(4)
    meta = json.loads((a.model_dir/'results.json').read_text())
    generator_name = meta['config']['generator']
    generator_class = {'v2_nested': SurfaceMixtureV2Nested,
                       'v2_nested_varlen': SurfaceMixtureV2NestedVarlen}[
                           generator_name]
    gen = generator_class(seed=a.seed, **meta['generator_kwargs'])
    pairs = build_pairs(gen, a.nstreams, a.length)
    assert pairs
    result = dict(config={**vars(a), 'model_dir': str(a.model_dir),
                           'output': str(a.output)},
                  pair_counts=dict(Counter(pair['kind'] for pair in pairs)),
                  baseline='Each correct closer is balanced 1/2 across '
                           'independent opener types; a blind baseline has '
                           'CE ln2 and key-edit TV zero.',
                  models={})
    for depth in [int(x) for x in a.depths.split(',')]:
        record = next(row for row in meta['records'] if row['depth']==depth)
        model = BilinearGPT(Config(**record['model_config']))
        model.load_state_dict(torch.load(a.model_dir/f'model_depth{depth}.pt',
                                         map_location='cpu', weights_only=True))
        model = model.to(a.device).eval()
        before = probabilities(model,[p['original'] for p in pairs],a.device)
        after = probabilities(model,[p['edited'] for p in pairs],a.device)
        result['models'][str(depth)] = summarize(pairs, before, after)
        print('depth',depth,{kind:round(row['effect_l1'],3) for kind,row in
              result['models'][str(depth)].items()},flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
