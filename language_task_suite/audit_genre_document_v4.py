"""Check V4 paraphrase oracle, canonical semantic replay, and site coverage."""

import argparse
import collections
import json
from pathlib import Path

from genre_document_v3 import _base_prefix
from genre_document_v4 import (GENRES, PosteriorTrace, next_distribution,
                               sample_sequence, to_v3)
from joint_document_v1 import _split, visible_state


def audit(regime='task_rich', n=100, length=256, seed=59001):
    sites = collections.Counter()
    forms = collections.Counter()
    checks = collections.Counter()
    for row in range(n):
        words, labels, distributions, genre = sample_sequence(
            seed+row, length, regime)
        trace = PosteriorTrace(regime)
        for i, (word, label, recorded) in enumerate(zip(
                words, labels, distributions, strict=True)):
            q, qs = trace.predict()
            assert abs(sum(q.values())-1) < 1e-10
            assert q.get(word, 0.) > 0
            assert all(trace.posterior[g] > 0 for g in GENRES)
            assert all(abs(q.get(w, 0.)-p) < 1e-10
                       for w, p in recorded.items())
            sites[label] += 1
            if word in ('Filed', 'Which', 'In', 'Following', 'works'):
                forms[word] += 1
            if row < 20 and i in (0, 1, 2, 3, 4, 32, 64, 128, 255):
                direct = next_distribution(words[:i], regime)
                assert all(abs(direct.get(w, 0.)-p) < 1e-10
                           for w, p in q.items())
                checks['independent_prefix'] += 1
            if label in ('source_answer', 'arithmetic_answer',
                         'step_number', 'bracket_close', 'section_number'):
                v3_prefix = to_v3(words[:i])
                canonical = _base_prefix(v3_prefix)
                history, raw = _split(canonical)
                if label == 'source_answer':
                    claims = visible_state(history)['claims']
                    assert q == {claims[(raw[3], raw[5])]: 1.}
                elif label == 'arithmetic_answer':
                    _, problem_raw = _split(v3_prefix)
                    assert q == {str(int(problem_raw[2])+
                                     int(problem_raw[4])): 1.}
                elif label == 'step_number':
                    step_history, _ = _split(v3_prefix)
                    assert q == {str(1+sum(s[0] == 'Step'
                                          for s in step_history)): 1.}
                elif label == 'bracket_close':
                    assert q == {')' if '(' in raw else ']': 1.}
                elif label == 'section_number':
                    assert q == {str(visible_state(history)['section_number']+1): 1.}
                checks[label] += 1
            trace.observe(word, qs)
        # The entire generated document must reduce to a valid V3 history
        # followed by a valid partial sentence, with no unsupported token.
        canonical = _base_prefix(to_v3(words))
        history, _ = _split(canonical)
        visible_state(history)
        checks['document_replay'] += 1
    assert all(forms[word] for word in ('Filed', 'Which', 'In',
                                       'Following', 'works'))
    assert checks['source_answer'] and checks['bracket_close']
    return dict(version='genre_document_v4', regime=regime, n=n,
                length=length, seed=seed, sampled_token_support=n*length,
                sites=dict(sites), paraphrase_occurrences=dict(forms),
                checks=dict(checks),
                caveat='Exact-oracle and semantic-state audit of a surface '
                       'intervention; not a natural-text realism claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--regime', choices=('task_rich', 'reference'),
                   default='task_rich')
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--seed', type=int, default=59001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.regime, args.n, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({k: result[k] for k in ('paraphrase_occurrences', 'checks')})
