"""Replay v4's exact prefix oracle and independently verify retrieval answers."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from audit_surface_mixture_v2 import check_clause
from audit_surface_repetition import WORD, form_cues, score
from compare_realism import summary
from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import OBJECT_INDEX, SurfaceMixtureV3
from surface_mixture_v4 import SurfaceMixtureV4, _content, render


def reference_answer(history, obj):
    for raw in reversed(history):
        clause = _content(raw)
        if not clause or clause[0] in ('Where', 'How'):
            continue
        frame = _frame(clause)
        if clause[OBJECT_INDEX[frame]] == obj:
            return clause[_indices(frame)[2]]
    raise AssertionError(f'Unseen queried object {obj}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--hard-query-rate', type=float, default=None)
    p.add_argument('--counterbalanced-queries', action='store_true')
    p.add_argument('--question-rate', type=float, default=.05)
    p.add_argument('--output', default='language_task_suite/surface_mixture_v4_audit.json')
    a = p.parse_args()
    gen = SurfaceMixtureV4(seed=12001, hard_query_rate=a.hard_query_rate,
                           counterbalanced_queries=a.counterbalanced_queries,
                           question_rate=a.question_rate)
    narrative_oracle = SurfaceMixtureV3()
    ids, labels, qs = gen.sample_batch(a.n, a.length)
    counts = Counter()
    reuse = Counter()
    examples = [render(row) for row in ids[:8]]
    for row, labs, distributions in zip(ids, labels, qs):
        tokens = [gen.id2token[i] for i in row]
        history, _ = _split(tokens)
        for i, raw in enumerate(history):
            clause = _content(raw)
            if clause and clause[0] != 'Where':
                check_clause(clause)
            elif clause:
                assert clause[:3] == ['Where', 'did', 'the']
                assert clause[4:] == ['appear', '?', 'Answer', ':',
                                      reference_answer(history[:i],
                                                       clause[3]), '.']
        for t, (token, q, label) in enumerate(zip(tokens, distributions, labs)):
            assert q == gen.next_distribution(tokens[:t])
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            counts[label] += 1
            if label in ('mixed_verb', 'mixed_object'):
                before, raw = _split(tokens[:t])
                projected = [word for old in before
                             if _content(old) and _content(old)[0] != 'Where'
                             for word in _content(old)] + _content(raw)
                assert q == narrative_oracle.next_distribution(projected)
            if label != 'question_answer':
                continue
            before, raw = _split(tokens[:t])
            clause = _content(raw)
            assert clause[:3] == ['Where', 'did', 'the']
            assert len(clause) == 8
            answer = reference_answer(before, clause[3])
            assert q == {answer: 1.0}
            if a.counterbalanced_queries:
                latest_places = {}
                for old in before:
                    c = _content(old)
                    if c and c[0] != 'Where':
                        frame = _frame(c)
                        latest_places[c[OBJECT_INDEX[frame]]] = c[_indices(frame)[2]]
                assert len(set(latest_places.values())) >= 2
                query_q = gen.next_distribution(tokens[:t-5])
                assert query_q == {obj: 1/len(latest_places)
                                   for obj in latest_places}
                multiplicities = Counter(latest_places.values())
                reuse['key_blind_bayes_correct_sum'] += (
                    max(multiplicities.values())/len(latest_places))
                reuse['key_blind_entropy_sum'] += -sum(
                    (count/len(latest_places))*math.log(count/len(latest_places))
                    for count in multiplicities.values())
            narratives = [old for old in before if _content(old)
                          and _content(old)[0] != 'Where']
            latest_clause = _content(narratives[-1])
            latest_place = latest_clause[_indices(_frame(latest_clause))[2]]
            reuse['latest_narrative_place_baseline_correct'] += latest_place == answer
            last_matching_index = max(i for i, old in enumerate(narratives)
                                      if _content(old)[OBJECT_INDEX[
                                          _frame(_content(old))]] == clause[3])
            intervening = len(narratives)-1-last_matching_index
            reuse['query_not_latest_narrative'] += intervening >= 1
            reuse['query_at_least_two_narratives_back'] += intervening >= 2
            prior = [old for old in before if _content(old)
                     and _content(old)[0] != 'Where'
                     and _content(old)[OBJECT_INDEX[_frame(_content(old))]] == clause[3]]
            reuse['query_matches_2plus'] += len(prior) >= 2
            if len(prior) >= 2:
                earlier = _content(prior[-2])[_indices(_frame(_content(prior[-2])))[2]]
                reuse['changed_place_among_repeats'] += earlier != answer
    natural_rows = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    n = min(a.n, len(natural_rows))
    synthetic = [render(row) for row in ids[:n]]
    natural = [tokenizer.decode(row[:a.length].tolist(),
                                skip_special_tokens=False)
               for row in natural_rows[:n]]
    eligible = [i for i in range(n)
                if len(WORD.findall(synthetic[i])) >= 40
                and len(WORD.findall(natural[i])) >= 40]
    output = dict(config=vars(a), n_vocab=len(gen.vocab), slots=dict(counts),
                  answer_reuse=dict(reuse),
                  key_blind_prior=(dict(
                      bayes_accuracy=reuse['key_blind_bayes_correct_sum']/counts['question_answer'],
                      entropy_nats=reuse['key_blind_entropy_sum']/counts['question_answer'])
                      if a.counterbalanced_queries else None),
                  matched_surface=dict(n=n, synthetic=summary(synthetic),
                                       fineweb=summary(natural)),
                  form_cues=dict(synthetic=form_cues(synthetic),
                                 fineweb=form_cues(natural)),
                  word_matched=dict(n=len(eligible), words=40,
                                    synthetic=score([synthetic[i] for i in eligible], 40),
                                    fineweb=score([natural[i] for i in eligible], 40)),
                  examples=examples,
                  caveat='Exact oracle and last-mentioned-object answer were replayed '
                         'for every generated prefix. Surface statistics use different '
                         'tokenizers, and the questions are controlled toy syntax, not '
                         'an estimate of natural question-answer prevalence.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ('slots', 'answer_reuse',
                                             'form_cues', 'word_matched')}, indent=2))


if __name__ == '__main__':
    main()
