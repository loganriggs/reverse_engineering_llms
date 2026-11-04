"""Screen WikiBio as a document-level empirical source for controlled rules.

This is a feasibility and form audit, not a semantic oracle. Infobox values
can diverge from the biography. Only literal visible-text mentions are counted
as candidate rule sites; no source prose is exported.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

from audit_document_structure_vs_fineweb import structure


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SUBJECT_VERBS = frozenset({
    'is', 'was', 'were', 'has', 'had', 'won', 'served', 'played', 'worked',
    'joined', 'became', 'made', 'began', 'started', 'studied', 'died',
    'married', 'founded', 'published', 'wrote', 'released', 'retired',
    'moved', 'returned', 'received', 'appeared', 'later', 'also',
})


def field_map(row):
    table = row['input_text']['table']
    result = {}
    for header, value in zip(table['column_header'], table['content']):
        value = value.strip().lower()
        if value and header not in result:
            result[header] = value
    return result


def mentions(text, value):
    if not value or len(value) < 3:
        return []
    return list(re.finditer(r'(?<![a-z])' + re.escape(value) +
                           r'(?![a-z])', text.lower()))


def candidate(row, max_words=160):
    """Return literal sites whose controlling values are visibly repeated."""
    fields = field_map(row)
    text = row['target_text'].strip().lower()
    name = fields.get('name', fields.get('article_title', '')).strip()
    if not name or not text.startswith(name) or len(name.split()) < 2:
        return None
    first = mentions(text, name)
    surname = name.split()[-1]
    if not surname.isalpha() or len(surname) < 4:
        return None
    surname_sites = mentions(text, surname)
    if not first or not surname_sites:
        return None
    first_end = first[0].end()
    later = [m for m in surname_sites if m.start() > first_end + 10 and
             len(WORD.findall(text[:m.start()])) <= max_words]
    if not later:
        return None
    year = re.search(r'\b(?:18|19|20)\d{2}\b',
                     fields.get('birth_date', ''))
    birth_year = year.group() if year else None
    year_sites = mentions(text, birth_year) if birth_year else []
    nationality = fields.get('nationality', '')
    nationality_sites = mentions(text, nationality) if nationality else []
    occupation = fields.get('occupation', '')
    occupation_sites = mentions(text, occupation) if occupation else []
    return {
        'text': text,
        'surname': surname,
        'first_surname_site_word': len(WORD.findall(text[:later[0].start()])),
        'birth_year_visible': bool(year_sites),
        'nationality_visible': bool(nationality_sites),
        'occupation_literal_visible': bool(occupation_sites),
        'multi_sentence': text.count('\n') >= 2,
        'words': len(WORD.findall(text)),
        'strict_editable': strict_editable(text, name, surname,
                                           surname_sites, max_words),
    }


def strict_editable(text, name, surname, surname_sites, max_words):
    """High-precision two-mention subject rewrite, not full semantic proof."""
    if len(surname_sites) != 2 or len(name.split()) not in (2, 3):
        return False
    first_line = text.split('\n', 1)[0]
    if not first_line.startswith(name) or len(mentions(first_line, surname)) != 1:
        return False
    second = surname_sites[1]
    if second.start() <= len(first_line) or len(WORD.findall(
            text[:second.start()])) > max_words:
        return False
    prior = text[:second.start()]
    if not prior.endswith('\n'):
        return False
    after = WORD.findall(text[second.end():second.end()+30])
    return bool(after and after[0] in SUBJECT_VERBS)


def first_words(text, n):
    words = list(WORD.finditer(text))
    return text[:words[n-1].end()] if len(words) >= n else None


def audit(parquet, *, scan_rows=20000, seed=74301, panel_size=512,
          natural_preview=None):
    counts = Counter()
    form_pool = []
    strict_form_pool = []
    first_site_words = []
    file = pq.ParquetFile(parquet)
    for batch in file.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            if counts['scanned'] >= scan_rows:
                break
            counts['scanned'] += 1
            site = candidate(row)
            if site is None:
                continue
            counts['surname_repeat_candidates'] += 1
            first_site_words.append(site['first_surname_site_word'])
            for key in ('birth_year_visible', 'nationality_visible',
                        'occupation_literal_visible', 'multi_sentence',
                        'strict_editable'):
                counts[key] += site[key]
            if site['strict_editable'] and site['birth_year_visible']:
                counts['strict_editable_with_birth_year'] += 1
            if site['birth_year_visible'] and site['nationality_visible']:
                counts['surname_year_nationality'] += 1
            crop = first_words(site['text'], 60)
            if crop is not None:
                form_pool.append(crop)
                if site['strict_editable']:
                    strict_form_pool.append(crop)
        if counts['scanned'] >= scan_rows:
            break
    rng = random.Random(seed)
    sample = rng.sample(form_pool, min(panel_size, len(form_pool)))
    strict_sample = rng.sample(strict_form_pool,
                               min(panel_size, len(strict_form_pool)))
    result = {
        'source': 'thoughtworks/wiki_bio train parquet; official WikiBio '
                  'paragraphs and infoboxes',
        'scan_rows': scan_rows,
        'counts': dict(counts),
        'surname_site_word_median': sorted(first_site_words)[
            len(first_site_words)//2] if first_site_words else None,
        'form_panel_size': len(sample),
        'form': structure(sample) if sample else None,
        'strict_form_panel_size': len(strict_sample),
        'strict_form': structure(strict_sample) if strict_sample else None,
        'caveat': 'Literal field mentions are not semantic alignment. '
                  'WikiBio text is lowercased/tokenized and infoboxes can '
                  'disagree with prose. No edits or model training yet.',
    }
    if natural_preview is not None:
        rows = json.loads(natural_preview.read_text())
        natural = [first_words(row['preview'], 60) for row in rows]
        natural = [x for x in natural if x]
        natural = rng.sample(natural, min(len(natural), len(sample)))
        result['natural_reference_form'] = structure(natural)
        result['natural_reference_n'] = len(natural)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--scan-rows', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=74301)
    parser.add_argument('--panel-size', type=int, default=512)
    parser.add_argument('--natural-preview', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.parquet, scan_rows=args.scan_rows,
                   seed=args.seed, panel_size=args.panel_size,
                   natural_preview=args.natural_preview)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['counts'], flush=True)


if __name__ == '__main__':
    main()
