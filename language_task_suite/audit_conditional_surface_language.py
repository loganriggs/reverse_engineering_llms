"""Audit the readable conditional rule mixture and fixed-prefix effects."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from compare_realism import summary
from conditional_surface_language import (ID2TOKEN, MODES,
                                          ConditionalSurfaceLanguage, render)


def tv(q0, q1):
    return .5*sum(abs(q0.get(token, 0)-q1.get(token, 0))
                   for token in q0.keys() | q1.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--regime', choices=('full', 'pair'), default='full')
    p.add_argument('--output', default='language_task_suite/conditional_surface_audit.json')
    a = p.parse_args()
    gen = ConditionalSurfaceLanguage(seed=6101, regime=a.regime)
    ids, labels, qs = gen.sample_batch(a.n, a.length)
    mode_counts, label_counts = Counter(), Counter()
    conflict_counts = Counter()
    edit_tvs = defaultdict(list)
    texts = []
    for row, labs, distributions in zip(ids, labels, qs):
        tokens = [ID2TOKEN[i] for i in row]
        mode = tokens[0]
        assert mode in MODES and tokens[1] == ':'
        mode_counts[mode] += 1
        label_counts.update(labs)
        texts.append(render(row))
        for t, (token, q, label) in enumerate(zip(tokens, distributions, labs)):
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            replay, replay_label = gen.next_distribution(row[:t])
            assert q == replay and label == replay_label
            if t < 2:
                continue
            body = tokens[2:t]
            if label == 'mixed_verb':
                candidates = gen.candidates(tokens[:t])
                # Independent direct weighted-mixture calculation.
                weights = gen.mode_weights[mode]
                z = sum(weights[name] for name in candidates)
                expected = defaultdict(float)
                for name, dist in candidates.items():
                    for outcome, mass in dist.items():
                        expected[outcome] += weights[name]*mass/z
                assert all(abs(q.get(k, 0)-v) < 1e-12
                           for k, v in expected.items())
                modes = {max(dist, key=dist.get) for dist in candidates.values()}
                if len(modes) > 1:
                    conflict_counts[mode] += 1
            for alternate in MODES:
                changed = [alternate, ':'] + body
                q1, _ = gen.next_distribution(changed)
                assert q1.get(token, 0) > 0
                distance = tv(q, q1)
                if label == 'frame':
                    assert distance < 1e-12
                elif label == 'mixed_verb' and alternate != mode:
                    edit_tvs[f'{mode}->{alternate}'].append(distance)
    panel = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    matched_n = min(a.n, len(panel))
    natural = [tokenizer.decode(r[:a.length].tolist(),
                                skip_special_tokens=False)
               for r in panel[:matched_n]]
    result = dict(config=vars(a), n_vocab=len(gen.vocab),
                  mode_counts=dict(mode_counts), label_counts=dict(label_counts),
                  modal_conflicts=dict(conflict_counts),
                  fixed_prefix_edit_tv={name: dict(n=len(v), mean=sum(v)/len(v),
                                                   nonzero_fraction=sum(x > 1e-9 for x in v)/len(v))
                                        for name, v in edit_tvs.items()},
                  matched_surface=dict(n=matched_n, tokens_per_window=a.length,
                                       synthetic=summary(texts[:matched_n]),
                                       fineweb=summary(natural)),
                  examples=texts[:8],
                  caveat='Mode-token interventions are valid because all body '
                         'rules have positive weights. Synthetic and FineWeb '
                         'windows use different tokenizers. This remains a '
                         'rigid toy grammar, not a natural-corpus match.')
    Path(a.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in
                      ('mode_counts', 'label_counts', 'modal_conflicts',
                       'fixed_prefix_edit_tv', 'matched_surface')}, indent=2))


if __name__ == '__main__':
    main()
