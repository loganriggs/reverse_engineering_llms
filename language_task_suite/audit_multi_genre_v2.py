"""Oracle, independent record checks, interventions and surface audit."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_ROWS, DEFAULT_TOKENIZER
from compare_realism import summary
from eval_multi_genre_v2_edits import build_pairs
from multi_genre_v2_language import (CATALOG_FIELDS, CATALOG_OBJECTS,
                                     CATALOG_PEOPLE, CATALOG_STATUSES,
                                     MultiGenreV2Language, render)


def independent_catalog_check(words):
    assert words[:3] == ['Catalog', ':', '\n']
    chunks, current = [], []
    for word in words[3:]:
        if word == '\n':
            assert not current
        else:
            current.append(word)
            if word == '.':
                chunks.append(current)
                current = []
    records, queries = {}, []
    for clause in chunks:
        if clause[0] == 'The':
            assert (len(clause) == 9 and clause[1] in CATALOG_OBJECTS
                    and clause[8] == '.')
            obj = clause[1]
            assert obj not in records
            if clause[2] == 'belongs':
                assert clause[3] == 'to' and clause[5:7] == ['and', 'is']
                record = {'owner': clause[4], 'status': clause[7]}
            else:
                assert clause[2] == 'is' and clause[4:7] == [
                    'and', 'belongs', 'to']
                record = {'status': clause[3], 'owner': clause[7]}
            assert record['owner'] in CATALOG_PEOPLE
            assert record['status'] in CATALOG_STATUSES
            assert all(record[field] != prior[field]
                       for prior in records.values() for field in CATALOG_FIELDS)
            records[obj] = record
        else:
            if clause[0] == 'Who':
                assert (len(clause) == 10 and clause[1:3] == ['owns', 'the']
                        and clause[4:8] == ['?', 'The', 'answer', 'is'])
                field, obj, answer = 'owner', clause[3], clause[8]
            elif clause[0] == 'What':
                assert (len(clause) == 13
                        and clause[1:6] == ['is', 'the', 'status', 'of', 'the']
                        and clause[7:11] == ['?', 'The', 'answer', 'is'])
                field, obj, answer = 'status', clause[6], clause[11]
            else:
                assert (clause[0] == 'For' and len(clause) == 14
                        and clause[1] == 'the' and clause[3] == ','
                        and clause[4] in ('who', 'what')
                        and clause[5:7] == ['is', 'the']
                        and clause[8:12] == ['?', 'The', 'answer', 'is'])
                field = 'owner' if clause[4] == 'who' else 'status'
                assert clause[7] == field
                obj, answer = clause[2], clause[12]
            assert clause[-1] == '.' and obj in records
            assert records[obj][field] == answer
            queries.append((field, obj))
    return len(records), len(queries)


def form_markers(texts):
    markers = {'newline': '\n', 'question': '?', 'colon': ':',
               'quote': '"', 'parenthesis': '(', 'digit': None}
    return {name: sum(any(ch.isdigit() for ch in text) if marker is None
                      else marker in text for text in texts)/len(texts)
            for name, marker in markers.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--output', type=Path,
                   default=Path('language_task_suite/multi_genre_v2_audit.json'))
    a = p.parse_args()
    gen = MultiGenreV2Language(seed=5102)
    rows, labels, distributions = gen.sample_batch(a.n, a.length)
    genres, tasks = Counter(), Counter()
    catalog_records = catalog_queries = 0
    examples = []
    for row, labs, qs in zip(rows, labels, distributions):
        words = [gen.id2token[i] for i in row]
        genres[words[0]] += 1
        tasks.update(labs)
        if words[0] == 'Catalog':
            nr, nq = independent_catalog_check(words)
            catalog_records += nr
            catalog_queries += nq
            if len(examples) < 3:
                examples.append(render(row))
        for at, (word, label, q) in enumerate(zip(words, labs, qs)):
            replay, replay_label = gen.next_distribution(row[:at])
            assert q == replay and label == replay_label
            assert abs(sum(q.values())-1.) < 1e-12 and q.get(word, 0.) > 0
    edits = build_pairs(MultiGenreV2Language(seed=5103), 500, a.length-1)
    for pair in edits:
        changed = pair['kind'] in ('catalog_query_key',
                                   'catalog_relevant_value')
        assert (pair['q_before'] != pair['q_after']) == changed
    fineweb_rows = torch.load(DEFAULT_ROWS, map_location='cpu', weights_only=True)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    matched = min(480, a.n, len(fineweb_rows))
    natural = [tokenizer.decode(row[:a.length].tolist(),
                                skip_special_tokens=False)
               for row in fineweb_rows[:matched]]
    synthetic = [render(row) for row in rows[:matched]]
    result = dict(config={**vars(a), 'output': str(a.output)},
                  genres=dict(genres), labels=dict(tasks),
                  catalog_completed_records=catalog_records,
                  catalog_completed_queries=catalog_queries,
                  edit_counts=dict(Counter(pair['kind'] for pair in edits)),
                  matched_surface=dict(n=matched,
                                       fineweb=summary(natural),
                                       synthetic=summary(synthetic),
                                       fineweb_form_markers=form_markers(natural),
                                       synthetic_form_markers=form_markers(synthetic)),
                  catalog_surface=summary([render(row) for row in rows
                                           if gen.id2token[row[0]] == 'Catalog']),
                  examples=examples,
                  caveat='FineWeb windows and synthetic streams use different '
                         'tokenizers. Catalog is deliberately task-enriched; '
                         'matching coarse form cues would not establish natural '
                         'language realism or the prevalence of this dependency.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print({key: result[key] for key in ('genres', 'catalog_completed_records',
                                      'catalog_completed_queries',
                                      'edit_counts')})


if __name__ == '__main__':
    main()
