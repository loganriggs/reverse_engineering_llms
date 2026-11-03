"""Screen ToTTo as a grounded, corpus-derived document realization bank.

Reads the official zip without extracting it. Reports only aggregates; raw
Wikipedia-derived sentences remain in the local archive and memory.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from compare_v12_fineweb_fresh import first_words
from audit_surface_repetition import WORD


DEFAULT_ZIP = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
                   'totto/totto_data.zip')
TRAIN = 'totto_data/totto_train_data.jsonl'


def _highlight_values(row):
    values = []
    for r, c in row['highlighted_cells']:
        try:
            value = row['table'][r][c]['value'].strip()
        except (IndexError, KeyError, TypeError):
            return None
        if value:
            values.append(value)
    return values


def audit(source, seed, n_form=64, form_words=100):
    counts = Counter()
    pages = Counter()
    groups = defaultdict(list)
    with zipfile.ZipFile(source) as archive, archive.open(TRAIN) as stream:
        for line in stream:
            row = json.loads(line)
            counts['examples'] += 1
            page = row['table_page_title']
            section = row['table_section_title']
            pages[page] += 1
            annotations = row['sentence_annotations']
            if not annotations:
                counts['no_annotation'] += 1
                continue
            text = annotations[0]['final_sentence'].strip()
            words = len(WORD.findall(text))
            counts['sentence_words_sum'] += words
            counts['sentences_8_to_60'] += 8 <= words <= 60
            values = _highlight_values(row)
            if values is None:
                counts['invalid_cells'] += 1
                continue
            counts['highlighted_cells_sum'] += len(values)
            exact = sum(value.lower() in text.lower() for value in values)
            counts['highlighted_exact_sum'] += exact
            counts['all_highlights_exact'] += bool(values) and exact == len(values)
            if not 8 <= words <= 60 or not values or exact != len(values):
                continue
            if len(row['highlighted_cells']) != len(values):
                continue
            rows = frozenset(r for r, _ in row['highlighted_cells'])
            groups[(page, section)].append((text, rows))
    rng = random.Random(seed)
    options = [(key, rows) for key, rows in groups.items()
               if len(rows) >= 4]
    rng.shuffle(options)
    documents = []
    for key, rows in options:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        pieces, used_rows = [], set()
        for text, highlighted_rows in shuffled:
            if highlighted_rows & used_rows:
                continue
            used_rows.update(highlighted_rows)
            pieces.append(text)
            if len(WORD.findall(' '.join(pieces))) >= form_words:
                break
        if len(WORD.findall(' '.join(pieces))) < form_words:
            continue
        documents.append(first_words(' '.join(pieces), form_words))
        if len(documents) == n_form:
            break
    return dict(source=str(source), train_records=counts['examples'],
                counts=dict(counts), unique_page_titles=len(pages),
                pages_ge2=sum(value >= 2 for value in pages.values()),
                pages_ge4=sum(value >= 4 for value in pages.values()),
                pages_ge8=sum(value >= 8 for value in pages.values()),
                unique_page_sections=len(groups),
                groups_ge4=len(options),
                form_documents=len(documents),
                form=structure(documents) if documents else None,
                caveat='Exact highlighted-cell strings are not proof that '
                       'the sentence contains no other unsupported facts. '
                       'Concatenating distinct rows from one page section '
                       'does not guarantee discourse coherence. The form '
                       'panel is a screen, not an accepted DGP.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=DEFAULT_ZIP)
    p.add_argument('--seed', type=int, default=73183)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.source, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('records', result['train_records'],
          'pages', result['unique_page_titles'],
          'groups', result['unique_page_sections'],
          'form_n', result['form_documents'], flush=True)
    if result['form']:
        print('types', result['form']['word_types'],
              'repeat', result['form'][
                  'repeated_across_docs_trigram_occurrence_share'],
              flush=True)


if __name__ == '__main__':
    main()
