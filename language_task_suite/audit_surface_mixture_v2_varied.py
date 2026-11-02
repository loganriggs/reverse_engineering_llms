"""Exact-oracle and grammar audit for the structurally varied v2 branch."""

import argparse
import json
from collections import Counter
from pathlib import Path

from audit_dependency_proxies import summaries
from surface_mixture_v2 import (ADJECTIVES, EVENTS, OBJECTS, PEOPLE, PLACES,
                                VERBS, _frame, _indices, _metadata, _split)
from surface_mixture_v2_varied import (DETERMINER_SLOTS,
                                       SurfaceMixtureV2Varied, render)


def check_complete(clause):
    frame = _frame(clause)
    subject_i, verb_i, place_i, base_end = _indices(frame)
    assert clause[-1] == '.' and len(clause) > base_end
    assert clause[subject_i] in PEOPLE and clause[verb_i] in VERBS
    assert clause[place_i] in PLACES
    for pos in DETERMINER_SLOTS[frame]:
        assert clause[pos] in ('the', 'this', 'that')
    suffix = clause[base_end:]
    if suffix[0] == '(':
        assert len(suffix) in (6, 10) and suffix[-2:] == [')', '.']
        starts = (1,) if len(suffix) == 6 else (1, 5)
        if len(suffix) == 10:
            assert suffix[4] == ','
        for start in starts:
            kind = suffix[start]
            if kind == 'near':
                assert suffix[start+1] == 'the' and suffix[start+2] in PLACES
            elif kind == 'during':
                assert suffix[start+1] == 'the' and suffix[start+2] in EVENTS
            else:
                assert kind == 'according' and suffix[start+1] == 'to'
                assert suffix[start+2] in PEOPLE
    else:
        assert suffix[0] in ('.', ',', 'because')
    _metadata(clause)
    return frame, suffix[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--length', type=int, default=160)
    p.add_argument('--seed', type=int, default=20261120)
    p.add_argument('--determiner-variation', type=float, default=.5)
    p.add_argument('--parenthetical-mass', type=float, default=.2)
    p.add_argument('--subject-repeat-mass', type=float, default=.4)
    p.add_argument('--subject-return-mass', type=float, default=0.)
    p.add_argument('--output', type=Path,
                   default=Path('language_task_suite/surface_mixture_v2_varied_audit.json'))
    a = p.parse_args()
    gen = SurfaceMixtureV2Varied(
        seed=a.seed, determiner_variation=a.determiner_variation,
        parenthetical_mass=a.parenthetical_mass,
        subject_repeat_mass=a.subject_repeat_mass,
        subject_return_mass=a.subject_return_mass)
    frames, tails, labels = Counter(), Counter(), Counter()
    n_verb = n_induction = modal_conflicts = 0
    texts = []
    for _ in range(a.n):
        ids, labs, qs = gen.sample_sequence(a.length)
        tokens = [gen.id2token[i] for i in ids]
        texts.append(render(ids))
        for t, (token, label, q) in enumerate(zip(tokens, labs, qs)):
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            assert q == gen.next_distribution(tokens[:t])
            _, clause = _split(tokens[:t])
            expected = 'frame'
            if clause:
                frame = _frame(clause)
                pos = len(clause)
                if pos == _indices(frame)[1]:
                    expected = 'mixed_verb'
                    candidates = gen.candidates(tokens[:t])
                    n_verb += 1
                    n_induction += 'induction' in candidates
                    modal_conflicts += len({max(dist, key=dist.get)
                                            for dist in candidates.values()}) > 1
                elif pos in DETERMINER_SLOTS[frame]:
                    expected = 'determiner'
                elif pos >= _indices(frame)[3]:
                    suffix = clause[_indices(frame)[3]:]
                    if suffix and suffix[0] == '(' and len(suffix) in (4, 8):
                        expected = 'parenthesis_close'
                        assert q == ({')': .6, ',': .4} if len(suffix) == 4
                                     else {')': 1.})
            assert label == expected
            labels[label] += 1
        history, _ = _split(tokens)
        for clause in history:
            frame, tail = check_complete(clause)
            frames[frame] += 1
            tails[tail] += 1
    output = dict(config={**vars(a), 'output': str(a.output)},
                  vocab_size=len(gen.vocab),
                  complete_frames=dict(frames), complete_tail_starts=dict(tails),
                  labels=dict(labels), verb_sites=n_verb,
                  induction_active=n_induction,
                  modal_conflicts=modal_conflicts,
                  surface=summaries(texts[:192]), examples=texts[:5],
                  caveat='All sampled prefixes replay the visible-prefix oracle; '
                         'completed clauses pass independent structural checks. '
                         'The surface summary uses a heuristic word filter, not '
                         'natural-language task labels or held-out FineWeb.')
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({key: output[key] for key in
                      ('vocab_size', 'complete_frames', 'complete_tail_starts',
                       'labels', 'verb_sites', 'induction_active',
                       'modal_conflicts')}, indent=2))


if __name__ == '__main__':
    main()
