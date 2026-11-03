"""Independent small-CFG enumeration and joint-document oracle checks."""

import argparse
import collections
import json
import math
from pathlib import Path

import genre_document_v7 as document
import genre_document_v8 as stateful_document
from genre_document_v3 import _base_prefix
from genre_document_v4 import to_v3
from joint_document_v1 import _split, visible_state
from small_genre_grammar_v7 import GRAMMARS
from weighted_cfg_oracle import EOS, WeightedCFG
from stateful_genre_grammar_v8 import grammar as stateful_grammar


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


def check_context_edit():
    profiles_a = (('Alice', 'pilot'),)
    profiles_b = (('Bob', 'pilot'),)
    prefix = ('The', 'pilot', 'in', 'the', 'report', 'was')
    q_a = stateful_grammar('court', profiles_a).next_distribution(prefix)
    q_b = stateful_grammar('court', profiles_b).next_distribution(prefix)
    assert q_a == {'Alice': 1.} and q_b == {'Bob': 1.}
    claims_a = (('badge', 'rare', 'Alice'),)
    claims_b = (('badge', 'rare', 'Bob'),)
    claim_prefix = ('The', 'badge', 'was', 'described', 'as', 'rare', 'by')
    c_a = stateful_grammar('court', (), claims_a).next_distribution(
        claim_prefix)
    c_b = stateful_grammar('court', (), claims_b).next_distribution(
        claim_prefix)
    assert c_a == {'Alice': 1.} and c_b == {'Bob': 1.}
    profile_a = ('Document : grove . Alice is a pilot . '
                 'The pilot in the report was').split()
    profile_b = ('Document : grove . Bob is a pilot . '
                 'The pilot in the report was').split()
    joint_profile_a = stateful_document.next_distribution(profile_a)
    joint_profile_b = stateful_document.next_distribution(profile_b)
    assert abs(joint_profile_a['Alice']-q_a['Alice']) < 1e-12
    assert joint_profile_b == q_b
    claim_a = ('Document : grove . Alice is a pilot . '
               'Bob is a pilot . According to Alice , the badge is rare . '
               'The badge was described as rare by').split()
    claim_b = ('Document : grove . Alice is a pilot . '
               'Bob is a pilot . According to Bob , the badge is rare . '
               'The badge was described as rare by').split()
    joint_claim_a = stateful_document.next_distribution(claim_a)
    joint_claim_b = stateful_document.next_distribution(claim_b)
    assert abs(joint_claim_a['Alice']-c_a['Alice']) < 1e-12
    assert joint_claim_b == c_b
    return dict(profile_name_probability_before=q_a['Alice'],
                profile_name_probability_after=q_b.get('Alice', 0.),
                claim_speaker_probability_before=c_a['Alice'],
                claim_speaker_probability_after=c_b.get('Alice', 0.),
                exact_joint_document_counterfactuals_passed=4)


def audit(regime='task_rich', n=30, length=256, seed=81001,
          version='v7'):
    assert version in ('v7', 'v8')
    source = document if version == 'v7' else stateful_document
    labels = collections.Counter()
    checks = collections.Counter()
    bg_prefix = collections.defaultdict(lambda: collections.Counter())
    for row in range(n):
        words, sites, saved, genre = source.sample_sequence(
            seed+row, length, regime)
        trace = (document.PosteriorTrace(regime) if version == 'v7' else
                 document.PosteriorTrace(
                     regime, stateful_document.conditional_distribution,
                     stateful_document.TOKEN2ID))
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
                bg_grammar = (GRAMMARS[genre] if version == 'v7' else
                              stateful_grammar(
                                  genre, *stateful_document._visible_facts(
                                      words[:i])))
                assert bg_grammar.prefix_probability(raw) > 0
            if row < 5 and i in (0, 1, 2, 3, 4, 32, 64, 128, 255):
                replay = source.next_distribution(words[:i], regime)
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
    return dict(version=('small_genre_pcfg_v7' if version == 'v7'
                         else 'stateful_small_genre_pcfg_v8'), regime=regime,
                n=n, length=length, seed=seed, vocab_size=len(source.VOCAB),
                enumeration=check_exhaustive(), labels=dict(labels),
                checks=dict(checks),
                context_edit=(check_context_edit() if version == 'v8'
                              else None),
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
    parser.add_argument('--version', choices=('v7', 'v8'), default='v7')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.regime, args.n, version=args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['checks'], flush=True)
