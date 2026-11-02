"""Audit v5's joint verb/object/retrieval/count prefix distributions."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from audit_surface_mixture_v2 import check_clause
from audit_surface_mixture_v4 import reference_answer
from audit_surface_repetition import WORD, form_cues, score
from compare_realism import summary
from surface_mixture_v2 import _frame, _indices, _split
from surface_mixture_v3 import SurfaceMixtureV3
from surface_mixture_v4 import _content
from surface_mixture_v5 import SurfaceMixtureV5, render


def independent_counts(history):
    out = Counter()
    for old in history:
        clause = _content(old)
        if clause and clause[0] not in ('Where', 'How'):
            out[clause[_indices(_frame(clause))[0]]] += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=3000)
    p.add_argument('--length', type=int, default=72)
    p.add_argument('--question-rate', type=float, default=.05)
    p.add_argument('--count-rate', type=float, default=.20)
    p.add_argument('--output', default='language_task_suite/surface_mixture_v5_audit.json')
    a = p.parse_args()
    gen = SurfaceMixtureV5(seed=12001, question_rate=a.question_rate,
                           count_rate=a.count_rate,
                           counterbalanced_queries=True)
    narrative_oracle = SurfaceMixtureV3()
    ids, labels, qs = gen.sample_batch(a.n, a.length)
    slots = Counter()
    count_stats = Counter()
    examples = [render(row) for row in ids[:8]]
    for row, labs, distributions in zip(ids, labels, qs):
        tokens = [gen.id2token[i] for i in row]
        history, _ = _split(tokens)
        for i, raw in enumerate(history):
            clause = _content(raw)
            if not clause:
                continue
            if clause[0] == 'Where':
                assert clause[:3] == ['Where', 'did', 'the']
                assert clause[4:] == ['appear', '?', 'Answer', ':',
                                      reference_answer(history[:i], clause[3]), '.']
            elif clause[0] == 'How':
                counts = independent_counts(history[:i])
                assert clause[:4] == ['How', 'many', 'times', 'was']
                assert clause[5:] == ['the', 'main', 'actor', '?', 'Answer', ':',
                                      str(counts[clause[4]]), '.']
            else:
                check_clause(clause)
        for t, (token, q, label) in enumerate(zip(tokens, distributions, labs)):
            assert q == gen.next_distribution(tokens[:t])
            assert abs(sum(q.values())-1) < 1e-12
            assert q.get(token, 0) > 0
            slots[label] += 1
            before, raw = _split(tokens[:t])
            content = _content(raw)
            if label in ('mixed_verb', 'mixed_object'):
                projected = [word for old in before
                             if _content(old) and _content(old)[0] not in ('Where', 'How')
                             for word in _content(old)] + content
                assert q == narrative_oracle.next_distribution(projected)
            elif label == 'question_answer':
                assert content[:3] == ['Where', 'did', 'the'] and len(content) == 8
                assert q == {reference_answer(before, content[3]): 1.0}
            elif label == 'count_answer':
                assert content[:4] == ['How', 'many', 'times', 'was']
                assert len(content) == 11
                counts = independent_counts(before)
                assert len(set(counts.values())) >= 2
                assert q == {str(counts[content[4]]): 1.0}
                name_q = gen.next_distribution(tokens[:t-7])
                assert name_q == {name: 1/len(counts) for name in counts}
                count_stats['answers_with_count_1'] += counts[content[4]] == 1
                count_stats['answers_with_count_2plus'] += counts[content[4]] >= 2
                count_stats['distinct_query_names_sum'] += len(counts)
                count_stats['distinct_answer_counts_sum'] += len(set(counts.values()))
                alternative = next(name for name in counts
                                   if counts[name] != counts[content[4]])
                edited = tokens[:t]
                edited[t-7] = alternative
                assert gen.next_distribution(edited) == {str(counts[alternative]): 1.0}
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
    output = dict(config=vars(a), n_vocab=len(gen.vocab), slots=dict(slots),
                  count_stats=dict(count_stats),
                  matched_surface=dict(n=n, synthetic=summary(synthetic),
                                       fineweb=summary(natural)),
                  form_cues=dict(synthetic=form_cues(synthetic),
                                 fineweb=form_cues(natural)),
                  word_matched=dict(n=len(eligible), words=40,
                                    synthetic=score([synthetic[i] for i in eligible], 40),
                                    fineweb=score([natural[i] for i in eligible], 40)),
                  examples=examples,
                  caveat='Every prefix was replayed; narrative mixtures were '
                         'checked against standalone v3 and count/recall answers '
                         'were independently recomputed. Surface frequency matches '
                         'do not establish natural task prevalence or realistic prose.')
    Path(a.output).write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: output[k] for k in ('slots', 'count_stats',
                                             'form_cues', 'word_matched')}, indent=2))


if __name__ == '__main__':
    main()
