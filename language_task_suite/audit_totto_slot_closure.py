"""Upper-bound screen for fully bound corpus-attested ToTTo realizations.

All counts are heuristic feasibility measures. Even when every named entity
and number is bound, a sentence can assert an unsupported relation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import zipfile

import spacy

from audit_totto_document_realisations import DEFAULT_ZIP, TRAIN
from totto_table_description_dgp import check_visible, extract


WORD = r'[A-Za-z0-9]'
NUMBER = re.compile(r'(?<!\w)\d[\d,./-]*(?!\w)')


def _parts(example):
    lines = example.text.splitlines()
    header = next(line for line in lines if line.startswith('| '))
    value_line = next(line for line in lines if line.startswith('| ')
                      and line != header)
    schema = tuple(part.strip().casefold()
                   for part in header.strip('|').split('|'))
    values = tuple(part.strip() for part in value_line.strip('|').split('|'))
    assert len(schema) == len(values)
    return schema, values, example.text.split('Description: ', 1)[1]


def _proposals(example, schema, values, sentence):
    candidates = []
    # Only unique, long-enough row values have unambiguous column bindings.
    counts = Counter(value.casefold() for value in values)
    for header, value in zip(schema, values):
        if len(value) >= 3 and counts[value.casefold()] == 1:
            candidates.append((value, f'CELL:{header}'))
    title = example.page.strip()
    if len(title) >= 4:
        candidates.append((title, 'PAGE:full'))
    title_words = re.findall(r'[A-Za-z]{4,}', title)
    if title_words and title_words[-1].casefold() != title.casefold():
        candidates.append((title_words[-1], 'PAGE:last'))
    # Prefer long matches and avoid overlapping/ambiguous matches.
    candidates.sort(key=lambda row: len(row[0]), reverse=True)
    occupied = [False]*len(sentence)
    spans = []
    for value, label in candidates:
        pattern = re.compile(r'(?<!'+WORD+r')'+re.escape(value)+
                             r'(?!'+WORD+r')', re.I)
        for match in pattern.finditer(sentence):
            if any(occupied[match.start():match.end()]):
                continue
            spans.append((match.start(), match.end(), label))
            for i in range(match.start(), match.end()):
                occupied[i] = True
    spans.sort()
    form, last = [], 0
    for lo, hi, label in spans:
        form.extend((sentence[last:lo], '<'+label+'>'))
        last = hi
    form.append(sentence[last:])
    return spans, ''.join(form)


def _covered(lo, hi, spans):
    return any(a <= lo and hi <= b for a, b, _ in spans)


def audit(source, sample_n=3000, seed=75911):
    rng = random.Random(seed)
    sample = []
    schema_pages = defaultdict(set)
    strict_n = 0
    with zipfile.ZipFile(source) as archive, archive.open(TRAIN) as stream:
        for line in stream:
            example = extract(json.loads(line))
            if example is None:
                continue
            assert check_visible(example)
            strict_n += 1
            schema, values, sentence = _parts(example)
            schema_pages[schema].add(example.page)
            record = (example, schema, values, sentence)
            if len(sample) < sample_n:
                sample.append(record)
            else:
                j = rng.randrange(strict_n)
                if j < sample_n:
                    sample[j] = record
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger',
                                               'attribute_ruler', 'lemmatizer'])
    counts = Counter()
    forms = set()
    by_schema = Counter()
    for record, doc in zip(sample, nlp.pipe((row[3] for row in sample),
                                       batch_size=128), strict=True):
        example, schema, values, sentence = record
        spans, form = _proposals(example, schema, values, sentence)
        other_page = len(schema_pages[schema]) >= 2
        missing_entities = [ent for ent in doc.ents
                            if not _covered(ent.start_char, ent.end_char,
                                            spans)]
        missing_numbers = [match for match in NUMBER.finditer(sentence)
                           if not _covered(match.start(), match.end(), spans)]
        counts['sampled'] += 1
        counts['cross_page_schema'] += other_page
        counts['any_unbound_entity'] += bool(missing_entities)
        counts['any_unbound_number'] += bool(missing_numbers)
        counts['slot_closed'] += not missing_entities and not missing_numbers
        if not missing_entities and not missing_numbers and other_page:
            counts['slot_closed_cross_page_schema'] += 1
            forms.add(form)
            by_schema[schema] += 1
    return {
        'source': str(source), 'seed': seed,
        'strict_examples_in_full_train': strict_n,
        'counts': dict(counts),
        'distinct_closed_cross_page_forms_in_sample': len(forms),
        'closed_cross_page_schemas_in_sample': len(by_schema),
        'top_closed_schemas': [
            {'headers': list(schema), 'sample_count': count}
            for schema, count in by_schema.most_common(15)],
        'caveat': 'Named entities come from a fallible English NER model. '
                  'Page-title matching is heuristic. Slot closure is only '
                  'a necessary screen, not semantic validity, grammatical '
                  'transfer, or a full next-token oracle. No source prose '
                  'or substituted text is exported.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_ZIP)
    parser.add_argument('--sample-n', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=75911)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.sample_n, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'])


if __name__ == '__main__':
    main()
