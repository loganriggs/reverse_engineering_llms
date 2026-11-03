"""Replay the joint DGP, check semantic sites and supported edits."""

import argparse
import collections
import json
import random
from pathlib import Path

from joint_document_v1 import (DETERMINERS, _split, next_distribution, sample_sequence,
                               site_label, verb_components, visible_state)


def _supported(prefix):
    for i, token in enumerate(prefix):
        if next_distribution(prefix[:i]).get(token, 0.) <= 0:
            return False
    return True


def _independent_claim_answer(prefix):
    history, raw = _split(prefix)
    assert raw[:2] == ('Who', 'described') and raw[2] in DETERMINERS
    claims = {}
    for sentence in history:
        if sentence[0] in ('According', 'As'):
            claims[(sentence[5], sentence[7])] = (
                sentence[2] if sentence[0] == 'According' else sentence[1])
    return claims[(raw[3], raw[5])]


def _query_key_edit(prefix):
    history, raw = _split(prefix)
    state = visible_state(history)
    current = (raw[3], raw[5])
    speaker = state['claims'][current]
    alternatives = [(item, attr) for (item, attr), value in
                    state['claims'].items() if value != speaker]
    for item, attr in alternatives:
        edited = list(prefix)
        edited[-6] = item  # raw[3] of the 9-token partial query
        edited[-4] = attr  # raw[5]
        if _supported(edited):
            return edited
    return None


def audit(n, length, seed):
    families = collections.Counter()
    edits = collections.Counter()
    joint = collections.Counter()
    lexical = collections.Counter()
    lexical_docs = 0
    support_checks = 0
    for row in range(n):
        words, labels, recorded = sample_sequence(seed+row, length)
        seen = set()
        for i, (word, label, q) in enumerate(zip(words, labels, recorded,
                                                  strict=True)):
            fresh = next_distribution(words[:i])
            assert q == fresh
            assert label == site_label(words[:i])
            assert word in q and q[word] > 0
            assert abs(sum(q.values())-1) < 1e-10
            support_checks += 1
            families[label] += 1
            seen.add(label)
            prefix = words[:i]
            if label == 'source_answer':
                answer = _independent_claim_answer(prefix)
                assert q == {answer: 1.}
                edits['source_answers_replayed'] += 1
                changed = _query_key_edit(prefix)
                if changed:
                    new = next_distribution(changed)
                    assert len(new) == 1 and next(iter(new)) != answer
                    assert new == {_independent_claim_answer(changed): 1.}
                    edits['source_key_changes_answer'] += 1
            elif label == 'section_number':
                history, _ = _split(prefix)
                number = 1+sum(sentence[0] in ('Section', 'Part')
                               for sentence in history)
                assert q == {str(number): 1.}
                edits['section_numbers_replayed'] += 1
            elif label == 'bracket_close':
                history, raw = _split(prefix)
                opener = next(w for w in raw if w in ('(', '['))
                expected = ')' if opener == '(' else ']'
                assert q == {expected: 1.}
                changed = list(prefix)
                changed[len(prefix)-len(raw)+raw.index(opener)] = (
                    '[' if opener == '(' else '(')
                assert _supported(changed)
                assert next_distribution(changed) == {
                    ']' if opener == '(' else ')': 1.}
                edits['bracket_opener_changes_closer'] += 1
            elif label == 'mixed_verb':
                history, raw = _split(prefix)
                name = raw[4] if raw[0] in ('At', 'After') else raw[0]
                parts = verb_components(visible_state(history), name)
                independent = collections.defaultdict(float)
                for mass, distribution in parts.values():
                    for candidate, probability in distribution.items():
                        independent[candidate] += mass*probability
                independent = {w: p for w, p in independent.items() if p}
                assert set(independent) == set(q)
                assert max(abs(independent[w]-q[w]) for w in q) < 1e-10
                edits['verb_components_replayed'] += 1
        if {'mixed_verb', 'claim_linked_item', 'source_answer',
            'section_number', 'bracket_close'} <= seen:
            joint['all_five_families'] += 1
        if {'mixed_verb', 'source_answer', 'section_number'} <= seen:
            joint['verb_source_section'] += 1
        alpha = [word.lower() for word in words if word.isalpha()]
        if len(alpha) >= 100 and lexical_docs < 49:
            lexical.update(alpha[:100])
            lexical_docs += 1
    assert lexical_docs == 49
    return dict(version='joint_document_v1', seed=seed, n_streams=n,
                length=length, support_and_oracle_replay_checks=support_checks,
                sites=dict(families), paired_edits=dict(edits),
                joint_streams=dict(joint),
                first49_x100_alpha=dict(types=len(lexical),
                                        singletons=sum(v == 1 for v in lexical.values()),
                                        top10_share=sum(v for _, v in lexical.most_common(10))/4900,
                                        the_share=lexical['the']/4900,
                                        a_share=lexical['a']/4900,
                                        an_share=lexical['an']/4900),
                caveat='Structural exact-oracle pilot. No independent natural '
                       'form pass, no trained-model result, and query-key edits '
                       'test supported targets only when distinct speakers exist.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--seed', type=int, default=26001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.n, args.length, args.seed)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print({key: result[key] for key in ('sites', 'paired_edits',
                                       'joint_streams', 'first49_x100_alpha')},
          flush=True)
