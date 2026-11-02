"""Independent grammar and causal-edit checks for numeric score reports."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from audit_multi_genre_v2 import form_markers
from compare_realism import summary
from numeric_report_language import (ID2TOKEN, NumericReportLanguage,
                                     record_state, render)


def line_spans(words):
    starts, lines, current = [], [], []
    start = 3
    for at, word in enumerate(words[3:], start=3):
        if word == '\n':
            if current:
                starts.append(start)
                lines.append(current)
            current, start = [], at+1
        else:
            current.append(word)
    return starts, lines, current


def edit_pairs(words, answer_at, gen):
    prefix = words[:answer_at]
    starts, completed, query = line_spans(prefix)
    assert len(completed) % 3 == 2 and len(query) == 9
    first, second = completed[-2:]
    record_state((first, second))
    answer = next(iter(gen.next_distribution(prefix)[0]))
    pairs = {}
    # Swapping the two scores always flips the answer, including the 1/9
    # boundary where no single valid digit can cross the other score.
    edited = prefix.copy()
    edited[starts[-2]+2], edited[starts[-1]+2] = (
        edited[starts[-1]+2], edited[starts[-2]+2])
    edited[starts[-2]+3] = ('point' if edited[starts[-2]+2] == '1'
                           else 'points')
    edited[starts[-1]+3] = ('point' if edited[starts[-1]+2] == '1'
                           else 'points')
    pairs['score_assignment_swap'] = edited
    # Move each name to the other's record while scores remain fixed.
    edited = prefix.copy()
    edited[starts[-2]], edited[starts[-1]] = (
        edited[starts[-1]], edited[starts[-2]])
    pairs['person_binding'] = edited
    edited = prefix.copy()
    query_start = len(prefix)-len(query)
    edited[query_start+5], edited[query_start+7] = (
        edited[query_start+7], edited[query_start+5])
    pairs['question_order'] = edited
    edited = prefix.copy()
    edited[starts[-2]+1] = ('earned' if first[1] == 'scored' else 'scored')
    pairs['wording'] = edited
    expected = {'score_assignment_swap': True, 'person_binding': True,
                'question_order': False, 'wording': False}
    for kind, alternate in pairs.items():
        distribution, label = gen.next_distribution(alternate)
        assert label == 'score_comparison_answer' and len(distribution) == 1
        assert (next(iter(distribution)) != answer) == expected[kind]
    return pairs


def audit(n, length, seed):
    gen = NumericReportLanguage(seed=seed)
    counts, edits, comparisons, examples = Counter(), Counter(), Counter(), []
    synthetic = []
    for stream in range(n):
        ids, labels, distributions = gen.sample_sequence(length)
        words = [ID2TOKEN[i] for i in ids]
        if stream < 3:
            examples.append(render(ids))
        if stream < 480:
            synthetic.append(render(ids))
        for at, (word, label, q) in enumerate(zip(words, labels, distributions)):
            replay, replay_label = gen.next_distribution(words[:at])
            assert label == replay_label and q == replay and word in q
            assert abs(sum(q.values())-1) < 1e-12
            counts[label] += 1
            if label == 'score_comparison_answer':
                _, completed, query = line_spans(words[:at])
                records = completed[-2:]
                assert len(records) == 2 and len(query) == 9
                assert all(len(line) == 5 and line[2].isdigit()
                           and line[3] == ('point' if line[2] == '1' else 'points')
                           and line[4] == '.' for line in records)
                values = {line[0]: int(line[2]) for line in records}
                assert len(values) == 2 and query[5] in values and query[7] in values
                assert query[5] != query[7]
                expected = (max if query[2] == 'more' else min)(
                    values, key=lambda person: values[person])
                assert q == {expected: 1.}
                comparisons[query[2]] += 1
                comparisons['answer_first_query_name' if expected == query[5]
                            else 'answer_second_query_name'] += 1
                for kind in edit_pairs(words, at, gen):
                    edits[kind] += 1
    assert edits and counts['score_comparison_answer'] > 0
    rows = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    matched = min(n, 480, len(rows))
    natural = [tokenizer.decode(row[:length].tolist(),
                                skip_special_tokens=False)
               for row in rows[:matched]]
    return dict(config=dict(n=n, length=length, seed=seed),
                labels=dict(counts), edits=dict(edits),
                comparisons=dict(comparisons), examples=examples,
                surface=dict(source_rows=str(DEFAULT_ROWS),
                             source_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                             n=matched, fineweb=summary(natural),
                             synthetic=summary(synthetic[:matched]),
                             fineweb_markers=form_markers(natural),
                             synthetic_markers=form_markers(synthetic[:matched])),
                caveat='Audits the synthetic oracle, visible grammar and '
                       'supported causal edits. FineWeb and toy windows use '
                       'different tokenizers; surface statistics do not '
                       'establish task prevalence or natural-language realism.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--seed', type=int, default=20261102)
    p.add_argument('--output', type=Path,
                   default=Path(__file__).with_name('numeric_report_audit.json'))
    a = p.parse_args()
    result = audit(a.n, a.length, a.seed)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['labels'], result['edits'])
