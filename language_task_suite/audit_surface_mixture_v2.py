"""Audit the multi-frame joint-rule generator and its natural-text gap."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from compare_realism import summary
from surface_mixture_v2 import (ADJECTIVES, EVENTS, OBJECTS, PEOPLE, PLACES,
                                VERBS,
                                SurfaceMixtureV2, _frame, _indices, _metadata,
                                _split, render)


def check_clause(clause):
    frame = _frame(clause)
    subject_i, verb_i, place_i, base_end = _indices(frame)
    assert clause[-1] == '.' and len(clause) > base_end
    assert clause[subject_i] in PEOPLE and clause[verb_i] in VERBS
    assert clause[place_i] in PLACES
    if frame == 'lead':
        assert clause[1] == ',' and clause[4] == 'the' and clause[5] in ADJECTIVES
        assert clause[6] in OBJECTS and clause[8] == 'the' and clause[11] == 'the'
        assert clause[12] in EVENTS
    elif frame == 'subject':
        assert clause[1] == ',' and clause[2] in ('after', 'before')
        assert clause[3] == 'the' and clause[4] in EVENTS and clause[5] == ','
        assert clause[7] == 'the' and clause[8] in ADJECTIVES
        assert clause[9] in OBJECTS and clause[11] == 'the'
    elif frame == 'at':
        assert clause[0:2] == ['At', 'the'] and clause[3] == ','
        assert clause[6] == 'the' and clause[7] in ADJECTIVES
        assert clause[8] in OBJECTS and clause[9:11] == ['before', 'the']
        assert clause[11] in EVENTS
    elif frame == 'when':
        assert clause[0:2] == ['When', 'the'] and clause[2] in EVENTS
        assert clause[3:5] == ['ended', ','] and clause[7] == 'the'
        assert clause[8] in ADJECTIVES and clause[9] in OBJECTS
        assert clause[10:12] == ['near', 'the']
    elif frame == 'in_place':
        assert clause[0:2] == ['In', 'the'] and clause[5] == 'the'
        assert clause[6] in ADJECTIVES and clause[7] in OBJECTS
        assert clause[8:10] == ['after', 'the'] and clause[10] in EVENTS
    else:
        assert frame == 'during' and clause[0:2] == ['During', 'the']
        assert clause[2] in EVENTS and clause[5] == 'the'
        assert clause[6] in ADJECTIVES and clause[7] in OBJECTS
        assert clause[8:10] == ['near', 'the']
    return frame, clause[base_end]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--output', default='language_task_suite/surface_mixture_v2_audit.json')
    a = p.parse_args()
    gen = SurfaceMixtureV2(seed=11001)
    ids, labels, qs = gen.sample_batch(a.n, a.length)
    frames, tails, labs = Counter(), Counter(), Counter()
    conflicts = 0
    verb_candidates = Counter()
    texts = []
    for row, row_labs, distributions in zip(ids, labels, qs):
        tokens = [gen.id2token[i] for i in row]
        texts.append(render(row))
        history, _ = _split(tokens)
        for clause in history:
            frame, tail = check_clause(clause)
            frames[frame] += 1
            tails[tail] += 1
            _metadata(clause)
        for t, (token, q, label) in enumerate(zip(tokens, distributions, row_labs)):
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            assert q == gen.next_distribution(tokens[:t])
            _, clause = _split(tokens[:t])
            expected = ('mixed_verb' if clause and
                        len(clause) == _indices(_frame(clause))[1]
                        else 'frame')
            assert label == expected
            labs[label] += 1
            if label == 'mixed_verb':
                candidates = gen.candidates(tokens[:t])
                verb_candidates[len(candidates)] += 1
                conflicts += len({max(dist, key=dist.get)
                                  for dist in candidates.values()}) > 1
                z = sum(gen.weights[name] for name in candidates)
                independent = defaultdict(float)
                for name, dist in candidates.items():
                    for outcome, mass in dist.items():
                        independent[outcome] += gen.weights[name]*mass/z
                assert all(abs(q.get(outcome, 0)-mass) < 1e-12
                           for outcome, mass in independent.items())
    panel = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    matched_n = min(a.n, len(panel))
    natural = [tokenizer.decode(r[:a.length].tolist(),
                                skip_special_tokens=False)
               for r in panel[:matched_n]]
    output = dict(config=vars(a), n_vocab=len(gen.vocab),
                  complete_sentence_frames=dict(frames),
                  complete_sentence_tail_starts=dict(tails),
                  labels=dict(labs), active_rule_counts=dict(verb_candidates),
                  modal_conflicts=conflicts,
                  matched_surface=dict(n=matched_n, tokens_per_window=a.length,
                                       synthetic=summary(texts[:matched_n]),
                                       fineweb=summary(natural)),
                  examples=texts[:8],
                  caveat='Exact visible-prefix replay and independent fixed '
                         'frame checks pass on generated streams. Different '
                         'tokenizers and fixed windows limit the FineWeb form '
                         'comparison; naturalness and dependency prevalence '
                         'are not established by these statistics.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({key: output[key] for key in
                      ('n_vocab', 'complete_sentence_frames',
                       'complete_sentence_tail_starts', 'labels',
                       'active_rule_counts', 'modal_conflicts',
                       'matched_surface')}, indent=2))


if __name__ == '__main__':
    main()
