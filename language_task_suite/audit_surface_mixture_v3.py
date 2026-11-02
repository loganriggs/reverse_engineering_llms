"""Audit both exact rule mixtures and sentence grammar in the v3 stream."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from audit_surface_mixture_v2 import check_clause
from compare_realism import summary
from surface_mixture_v2 import SurfaceMixtureV2, _frame, _indices, _split, render
from surface_mixture_v3 import OBJECT_INDEX, SurfaceMixtureV3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--object-vocab-size', type=int, default=None)
    p.add_argument('--output', default='language_task_suite/surface_mixture_v3_audit.json')
    a = p.parse_args()
    gen = SurfaceMixtureV3(seed=12001,
                           object_vocab_size=a.object_vocab_size)
    v2 = SurfaceMixtureV2()
    ids, labels, qs = gen.sample_batch(a.n, a.length)
    frames, slots, active, conflicts = Counter(), Counter(), Counter(), Counter()
    examples = []
    for row, labs, distributions in zip(ids, labels, qs):
        tokens = [gen.id2token[i] for i in row]
        examples.append(render(row))
        history, _ = _split(tokens)
        for clause in history:
            frame, _ = check_clause(clause)
            frames[frame] += 1
        for t, (token, q, label) in enumerate(zip(tokens, distributions, labs)):
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            assert q == gen.next_distribution(tokens[:t])
            _, clause = _split(tokens[:t])
            expected = 'frame'
            if clause:
                frame = _frame(clause)
                if len(clause) == _indices(frame)[1]:
                    expected = 'mixed_verb'
                elif len(clause) == OBJECT_INDEX[frame]:
                    expected = 'mixed_object'
            assert label == expected
            slots[label] += 1
            if label not in ('mixed_verb', 'mixed_object'):
                continue
            candidates = (gen.candidates(tokens[:t]) if label == 'mixed_verb'
                          else gen.object_candidates(tokens[:t]))
            if label == 'mixed_verb':
                assert q == v2.next_distribution(tokens[:t])
            weights = gen.weights if label == 'mixed_verb' else gen.object_weights
            active[(label, len(candidates))] += 1
            conflicts[label] += (len({max(dist, key=dist.get)
                                      for dist in candidates.values()}) > 1)
            z = sum(weights[name] for name in candidates)
            independent = defaultdict(float)
            for name, dist in candidates.items():
                for outcome, mass in dist.items():
                    independent[outcome] += weights[name]*mass/z
            assert set(q) == set(independent)
            assert all(abs(q[outcome]-mass) < 1e-12
                       for outcome, mass in independent.items())
    panel = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    matched_n = min(a.n, len(panel))
    natural = [tokenizer.decode(row[:a.length].tolist(),
                                skip_special_tokens=False)
               for row in panel[:matched_n]]
    output = dict(config=vars(a), n_vocab=len(gen.vocab),
                  complete_sentence_frames=dict(frames),
                  slots=dict(slots),
                  active_rule_counts={f'{label}_{n}': count
                                      for (label, n), count in active.items()},
                  modal_conflicts=dict(conflicts),
                  matched_surface=dict(n=matched_n, tokens_per_window=a.length,
                                       synthetic=summary(examples[:matched_n]),
                                       fineweb=summary(natural)),
                  examples=examples[:8],
                  caveat='Every generated prefix was replayed and both mixtures '
                         'were independently recomputed. Frame grammar was '
                         'checked on complete sentences. The frozen FineWeb '
                         'panel uses a different tokenizer and does not establish '
                         'natural dependency rates or semantic plausibility.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({key: output[key] for key in
                      ('n_vocab', 'slots', 'active_rule_counts',
                       'modal_conflicts', 'matched_surface')}, indent=2))


if __name__ == '__main__':
    main()
