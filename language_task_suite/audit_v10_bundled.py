"""Audit exact-prefix semantics and event overlap in a bundled stream."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX
from surface_mixture_v4 import _content
from surface_mixture_v10_linked_events import SurfaceMixtureV10LinkedEvents
from surface_mixture_v10_bundled import SurfaceMixtureV10Bundled, bridge_state


def panel(cls, n, length, seed, kwargs):
    generator = cls(seed=seed, **kwargs)
    control = (SurfaceMixtureV10LinkedEvents(seed=seed, **kwargs)
               if cls is SurfaceMixtureV10Bundled else None)
    counts = Counter()
    for _ in range(n):
        ids, labels, distributions = generator.sample_sequence(length)
        words = [generator.id2token[token] for token in ids]
        assert len(words) == len(labels) == len(distributions) == length
        events = set()
        for t, (word, label, q) in enumerate(zip(words, labels,
                                                 distributions)):
            prefix = words[:t]
            assert abs(sum(q.values()) - 1.) < 1e-9
            assert q.get(word, 0.) > 0
            assert q == generator.next_distribution(prefix)
            counts[f'label_{label}'] += 1
            history, raw = _split(prefix)
            content = _content(raw)
            facts, _, ready = bridge_state(history)
            if label == 'source_answer':
                assert q == {word: 1.}
                events.add('source_answer')
                if content[3] in ready:
                    events.add('linked_query_answer')
            if content and content[0] == 'Who' and len(content) == 3:
                if word in ready:
                    events.add('linked_question')
                if control and ready:
                    baseline = control.next_distribution(prefix)
                    target = ready & set(baseline)
                    if target:
                        assert sum(q.get(z, 0.) for z in target) >= sum(
                            baseline.get(z, 0.) for z in target) - 1e-10
                        counts['query_object_boost_checks'] += 1
            if not content or content[0] in (
                    'As', 'Who', 'Where', 'How', 'Section', 'Detail',
                    'What', 'The'):
                continue
            frame = _frame(content)
            if len(content) == OBJECT_INDEX[frame]:
                source = generator.source_object_candidates(prefix)
                if source and word in source:
                    events.add('source_object_sample')
                    counts['source_object_samples'] += 1
            if len(content) == _indices(frame)[3]:
                obj = content[OBJECT_INDEX[frame]]
                speaker = content[_indices(frame)[0]]
                if facts.get(obj) == speaker and control:
                    baseline = control.next_distribution(prefix)
                    assert (q.get('(', 0.) + q.get('[', 0.) >=
                            baseline.get('(', 0.) + baseline.get('[', 0.) - 1e-10)
                    counts['linked_bracket_boost_checks'] += 1
            if label == 'bracket_close':
                speaker = content[_indices(frame)[0]]
                opener = '(' if ')' in q else '['
                closer = ')' if opener == '(' else ']'
                assert closer in q and q.get(')' if closer == ']' else ']', 0.) == 0
                k = max(k for k in range(t) if words[k] == opener and
                        '.' not in words[k:t])
                edited = prefix.copy()
                edited[k] = '[' if opener == '(' else '('
                swapped = generator.next_distribution(edited)
                other = ']' if closer == ')' else ')'
                assert abs(swapped[other] - q[closer]) < 1e-10
                assert {z: p for z, p in swapped.items() if z != other} == {
                    z: p for z, p in q.items() if z != closer}
                counts['bracket_swap_checks'] += 1
                if word == closer and facts.get(content[OBJECT_INDEX[frame]]) == speaker:
                    events.add('linked_bracket_completed')
        for event in events:
            counts[f'stream_{event}'] += 1
        counts['stream_source_answer_and_linked_bracket'] += (
            'source_answer' in events and 'linked_bracket_completed' in events)
        counts['stream_linked_bracket_and_linked_query_answer'] += (
            'linked_bracket_completed' in events and
            'linked_query_answer' in events)
    return dict(n=n, length=length, seed=seed, model=cls.__name__,
                counts=dict(counts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=98001)
    p.add_argument('--regime', choices=('diagnostic', 'reference_candidate'),
                   default='reference_candidate')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.regime == 'diagnostic':
        kwargs = dict(counterbalanced_queries=True, question_rate=.05,
                      count_rate=0., quote_rate=.30, source_rate=.30,
                      section_rate=.20, detail_rate=.8,
                      scope_question_rate=.25, source_object_mass=.25,
                      source_subject_mass=.25, bracket_rate=.12,
                      linked_subject_mass=.65, linked_object_mass=.65,
                      linked_question_mass=.30, linked_query_key_mass=.65)
    else:
        kwargs = dict(counterbalanced_queries=True, question_rate=.05,
                      count_rate=0., quote_rate=.08, source_rate=.15,
                      section_rate=.06, detail_rate=.8,
                      scope_question_rate=.08, source_object_mass=.15,
                      source_subject_mass=.15, bracket_rate=.06,
                      linked_subject_mass=.65, linked_object_mass=.65,
                      linked_question_mass=.30, linked_query_key_mass=.65)
    rows = [panel(cls, args.n, args.length, args.seed + index, kwargs)
            for index in (0, 1)
            for cls in (SurfaceMixtureV10LinkedEvents,
                        SurfaceMixtureV10Bundled)]
    report = {'config': {**vars(args), 'output': str(args.output),
                         'kwargs': kwargs}, 'panels': rows,
              'scope': 'Existing V10 and its bracket-mediated extension share '
                       'base and claim-link rates; the extension adds visible-'
                       'prefix conditional bracket/query boosts. Diagnostic settings are '
                       'task-rich, reference candidate settings are not '
                       'FineWeb-calibrated. This validates DGP mechanics, '
                       'not model competence or prose realism.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for row in rows:
        c = row['counts']
        print(row['model'], row['seed'],
              c.get('stream_source_object_sample', 0),
              c.get('stream_linked_bracket_completed', 0),
              c.get('stream_source_answer_and_linked_bracket', 0),
              c.get('stream_linked_bracket_and_linked_query_answer', 0),
              flush=True)


if __name__ == '__main__':
    main()
