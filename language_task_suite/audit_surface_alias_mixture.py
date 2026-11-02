"""Replay the exact prefix oracle and inspect alias/lexical rule collisions."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from surface_alias_mixture import ALIASES, AliasMixtureLanguage, alias_state, render
from surface_mixture_v2 import _frame, _indices, _split


def weighted_candidates(gen, candidates):
    total = sum(gen.weights[name] for name in candidates)
    out = defaultdict(float)
    for name, dist in candidates.items():
        for token, p in dist.items():
            out[token] += gen.weights[name]*p/total
    return dict(out)


def supported_prefix(gen, tokens, start):
    return all(gen.next_distribution(tokens[:i]).get(tokens[i], 0) > 0
               for i in range(start, len(tokens)))


def audit(n, length, seed, generator_kwargs=None):
    generator_kwargs = generator_kwargs or {}
    gen = AliasMixtureLanguage(seed=seed, **generator_kwargs)
    counts = Counter()
    mapping_pairs = Counter()
    examples = []
    for _ in range(n):
        ids, labels, generated_q = gen.sample_sequence(length)
        tokens = [gen.id2token[i] for i in ids]
        counts['tokens'] += len(tokens)
        for t, (token, label, stored) in enumerate(zip(tokens, labels, generated_q)):
            prefix = tokens[:t]
            q = gen.next_distribution(prefix)
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            assert set(q) == set(stored)
            assert max(abs(q[k]-stored[k]) for k in q) < 1e-12
            counts[label] += 1
            if label == 'alias_name':
                _, definition = _split(prefix)
                mapping_pairs[(token, definition[0])] += 1
            if label != 'mixed_verb':
                continue
            candidates = gen.candidates(prefix)
            expected = weighted_candidates(gen, candidates)
            assert set(expected) == set(q)
            assert max(abs(expected[k]-q[k]) for k in q) < 1e-12
            history, clause = _split(prefix)
            mapping = alias_state(history)
            canonical = gen._canonical_prefix(history, clause)
            _, normalized = _split(canonical)
            subject_pos = _indices(_frame(normalized))[0]
            subject = clause[subject_pos]
            counts['verb_alias_subject' if subject in ALIASES
                   else 'verb_canonical_subject'] += 1
            if subject in ALIASES:
                assert mapping[subject] == normalized[subject_pos]
                counts['alias_verb_with_induction' if 'induction' in candidates
                       else 'alias_verb_without_induction'] += 1
                if len({max(dist, key=dist.get) for dist in candidates.values()}) > 1:
                    counts['alias_verb_modal_conflict'] += 1
                if len(examples) < 5:
                    examples.append(dict(prefix=render(ids[:t]),
                                         alias=subject, canonical=mapping[subject],
                                         candidates=candidates, oracle=q))
            if subject not in ALIASES or len(mapping) < 2:
                continue
            start = sum(len(old) for old in history)
            subject_index = start+subject_pos
            support = gen.next_distribution(prefix[:subject_index])
            for other, referent in mapping.items():
                if (other == subject or referent == mapping[subject]
                        or support.get(other, 0) <= 0):
                    continue
                changed = prefix.copy()
                changed[subject_index] = other
                if not supported_prefix(gen, changed, subject_index):
                    continue
                q_after = gen.next_distribution(changed)
                assert set(q_after).issubset(gen.vocab)
                if q_after != q:
                    counts['supported_alias_key_edits'] += 1
                    counts['alias_key_oracle_tv_sum'] += .5*sum(
                        abs(q_after.get(tok, 0)-q.get(tok, 0))
                        for tok in gen.vocab)
                break
    assert counts['alias_name'] > 0 and counts['verb_alias_subject'] > 0
    assert counts['alias_verb_modal_conflict'] > 0
    people_per_alias = [len({person for alias_name, person in mapping_pairs
                             if alias_name == alias}) for alias in ALIASES]
    return dict(config=dict(n=n, length=length, seed=seed,
                            generator_kwargs=generator_kwargs),
                vocabulary_size=len(gen.vocab), counts=dict(counts),
                mapping_diversity=dict(distinct_alias_person_pairs=len(mapping_pairs),
                                       min_people_per_alias=min(people_per_alias),
                                       max_people_per_alias=max(people_per_alias)),
                examples=examples,
                caveat='Replay and independent weighted-mixture checks establish '
                       'prefix oracle consistency, not natural-text realism or '
                       'model competence. Alias-key edit counts are supported '
                       'through the complete edited prefix and change the '
                       'current verb oracle. They still need a model-side '
                       'counterfactual test.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--parenthetical-rate', type=float, default=.55)
    p.add_argument('--alias-fraction', type=float, default=.65)
    p.add_argument('--alias-use', type=float, default=.65)
    p.add_argument('--definition-start-rate', type=float, default=.20)
    p.add_argument('--alias-reuse-rate', type=float, default=.25)
    p.add_argument('--initial-alias-definitions', type=int, default=0)
    p.add_argument('--verb-rule-weights', type=json.loads)
    p.add_argument('--output', type=Path,
                   default=Path(__file__).with_name('surface_alias_audit.json'))
    a = p.parse_args()
    result = audit(a.n, a.length, a.seed,
                   dict(parenthetical_rate=a.parenthetical_rate,
                        alias_fraction=a.alias_fraction,
                        alias_use=a.alias_use,
                        definition_start_rate=a.definition_start_rate,
                        alias_reuse_rate=a.alias_reuse_rate,
                        initial_alias_definitions=a.initial_alias_definitions,
                        **({'weights': a.verb_rule_weights}
                           if a.verb_rule_weights is not None else {})))
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'])
