"""Independent small-CFG enumeration and joint-document oracle checks."""

import argparse
import collections
import json
import math
from pathlib import Path

import genre_document_v7 as document
from genre_document_v3 import _base_prefix
from genre_document_v4 import to_v3
from joint_document_v1 import _split, visible_state
from small_genre_grammar_v7 import GRAMMARS
from weighted_cfg_oracle import EOS, WeightedCFG


def exhaustive(rules, sym):
    """Deliberately separate brute-force enumerator for the tiny fixture."""
    if sym not in rules:
        return {(sym,): 1.}
    totals = collections.defaultdict(float)
    for probability, rhs in rules[sym]:
        partial = {(): probability}
        for child in rhs:
            next_partial = collections.defaultdict(float)
            for prior, mass in partial.items():
                for suffix, child_mass in exhaustive(rules, child).items():
                    next_partial[prior+suffix] += mass*child_mass
            partial = next_partial
        for sentence, mass in partial.items():
            totals[sentence] += mass
    return dict(totals)


def check_exhaustive():
    rules = {
        'S': ((1., ('A', EOS)),),
        'A': ((.4, ('X', 'Y', '.')), (.6, ('X', 'Z', '.'))),
        'X': ((.7, ('a',)), (.3, ('b',))),
        'Y': ((.5, ('c',)), (.5, ('d',))),
        'Z': ((.25, ('c',)), (.75, ('e',))),
    }
    grammar = WeightedCFG(rules)
    strings = exhaustive(rules, 'S')
    assert abs(sum(strings.values())-1.) < 1e-12
    assert abs(strings[('a', 'c', '.', EOS)]-.245) < 1e-12
    prefixes = {(): 1.}
    for sentence, mass in strings.items():
        for length in range(1, len(sentence)+1):
            prefix = sentence[:length]
            prefixes[prefix] = prefixes.get(prefix, 0.)+mass
    for prefix, expected_mass in prefixes.items():
        got_mass = grammar.prefix_probability(prefix)
        assert abs(got_mass-expected_mass) < 1e-12, prefix
        q = grammar.next_distribution(prefix)
        if prefix and prefix[-1] == EOS:
            assert not q
            continue
        expected = collections.defaultdict(float)
        for sentence, mass in strings.items():
            if sentence[:len(prefix)] == prefix:
                expected[sentence[len(prefix)]] += mass/expected_mass
        assert set(q) == set(expected)
        assert all(abs(q[t]-expected[t]) < 1e-12 for t in q)
    assert grammar.prefix_probability(('a', 'z')) == 0.
    assert grammar.next_distribution(('a', 'z')) == {}
    return dict(ambiguous_surface_probability=strings[('a', 'c', '.', EOS)],
                checked_prefixes=len(prefixes))


def audit(regime='task_rich', n=30, length=256, seed=81001):
    labels = collections.Counter()
    checks = collections.Counter()
    bg_prefix = collections.defaultdict(lambda: collections.Counter())
    for row in range(n):
        words, sites, saved, genre = document.sample_sequence(
            seed+row, length, regime)
        trace = document.PosteriorTrace(regime)
        for i, (word, label, recorded) in enumerate(zip(
                words, sites, saved, strict=True)):
            q, qs = trace.predict()
            assert abs(sum(q.values())-1.) < 1e-10
            assert q.get(word, 0.) > 0
            assert set(recorded) == set(q)
            assert all(abs(recorded[t]-q[t]) < 1e-10 for t in q)
            labels[label] += 1
            if label == 'background_word':
                _, raw = _split(words[:i])
                bucket = ('1' if len(raw) == 1 else '2' if len(raw) == 2
                          else '3-4' if len(raw) <= 4 else '5-9'
                          if len(raw) <= 9 else '10+')
                bg_prefix[bucket]['n'] += 1
                bg_prefix[bucket]['deterministic'] += int(len(q) == 1)
                bg_prefix[bucket]['entropy'] -= sum(p*math.log(p)
                                                    for p in q.values())
                assert GRAMMARS[genre].prefix_probability(raw) > 0
            if row < 5 and i in (0, 1, 2, 3, 4, 32, 64, 128, 255):
                replay = document.next_distribution(words[:i], regime)
                assert set(replay) == set(q)
                assert all(abs(replay[t]-q[t]) < 1e-10 for t in q)
                checks['independent_prefix_replay'] += 1
            if label in ('source_answer', 'arithmetic_answer',
                         'step_number', 'bracket_close', 'section_number'):
                v3_prefix = to_v3(document._task_prefix(words[:i]))
                history, raw = _split(_base_prefix(v3_prefix))
                if label == 'source_answer':
                    speaker = visible_state(history)['claims'][(raw[3], raw[5])]
                    assert q == {speaker: 1.}
                elif label == 'arithmetic_answer':
                    _, problem = _split(v3_prefix)
                    assert q == {str(int(problem[2])+int(problem[4])): 1.}
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
        task_history, _ = _split(_base_prefix(to_v3(
            document._task_prefix(words))))
        visible_state(task_history)
        checks['semantic_document_replay'] += 1
    assert labels['background_word'] and checks['bracket_close']
    return dict(version='small_genre_pcfg_v7', regime=regime, n=n,
                length=length, seed=seed, vocab_size=len(document.VOCAB),
                enumeration=check_exhaustive(), labels=dict(labels),
                checks=dict(checks),
                background_prefix={k: dict(n=int(v['n']),
                     deterministic_fraction=v['deterministic']/v['n'],
                     mean_next_token_entropy_nats=v['entropy']/v['n'])
                     for k, v in bg_prefix.items()},
                caveat=('The small grammar proves exact ambiguity-aware '
                        'token oracle feasibility; its hand-built lexicon '
                        'and topic sentences are not natural-language realism.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime', choices=('task_rich', 'reference'),
                        default='task_rich')
    parser.add_argument('--n', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.regime, args.n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['checks'], flush=True)
