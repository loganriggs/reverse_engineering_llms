"""Screen corpus-attested ToTTo realization forms for cross-page reuse.

This is a feasibility audit, not a generator: exact value slots alone do not
establish that the remaining human sentence is valid for a different row.
Raw source sentences and proposed substitutions are never written to JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import zipfile

from audit_totto_document_realisations import DEFAULT_ZIP, TRAIN
from totto_table_description_dgp import check_visible, extract


def _schema_and_form(example):
    lines = example.text.splitlines()
    header = next(line for line in lines if line.startswith('| '))
    schema = tuple(part.strip().casefold()
                   for part in header.strip('|').split('|'))
    start = example.text.index('Description: ') + len('Description: ')
    sentence = example.text[start:]
    sites = sorted(example.sites, key=lambda site: site.span[0])
    pieces, previous = [], start
    for site in sites:
        pieces.append(example.text[previous:site.span[0]])
        pieces.append('<'+site.column.casefold()+'>')
        previous = site.span[1]
    pieces.append(example.text[previous:])
    return schema, ''.join(pieces), sentence


def audit(source):
    counts = Counter()
    schema_pages = defaultdict(set)
    schema_forms = defaultdict(set)
    forms = defaultdict(list)
    with zipfile.ZipFile(source) as archive, archive.open(TRAIN) as stream:
        for line in stream:
            counts['records'] += 1
            example = extract(json.loads(line))
            if example is None:
                continue
            assert check_visible(example)
            counts['strict_examples'] += 1
            schema, form, _ = _schema_and_form(example)
            schema_pages[schema].add(example.page)
            schema_forms[schema].add(form)
            forms[form].append((schema, example.page))
            # Page identity is often not among the highlighted cell values.
            # A same-schema value swap then leaves the old entity in prose.
            title_words = re.findall(r'[A-Za-z]{4,}', example.page)
            if title_words and re.search(r'\b'+re.escape(title_words[-1])+r'\b',
                                         form, re.I):
                counts['forms_with_unbound_page_name'] += 1
            if re.search(r'\d', form):
                counts['forms_with_unbound_number'] += 1
    cross_page = {schema for schema, pages in schema_pages.items()
                  if len(pages) >= 2}
    frequent = {schema for schema, pages in schema_pages.items()
                if len(pages) >= 5}
    counts['distinct_schemas'] = len(schema_pages)
    counts['schemas_cross_page'] = len(cross_page)
    counts['schemas_five_pages'] = len(frequent)
    counts['distinct_delexicalized_forms'] = len(forms)
    counts['forms_used_on_multiple_pages'] = sum(
        len({page for _, page in items}) >= 2 for items in forms.values())
    counts['cross_page_schemas_with_ten_forms'] = sum(
        len(schema_forms[schema]) >= 10 for schema in cross_page)
    # Number of *source* examples whose schema permits at least one other
    # page. A separate target-validity audit is required before substitution.
    counts['examples_with_other_page_same_schema'] = sum(
        len(items) for form, items in forms.items()
        if items[0][0] in cross_page)
    top = sorted(((len(schema_pages[s]), len(schema_forms[s]), s)
                  for s in schema_pages), reverse=True)[:20]
    return {
        'source': str(source), 'counts': dict(counts),
        'top_schemas': [{'page_count': pages, 'form_count': nforms,
                         'headers': list(schema)}
                        for pages, nforms, schema in top],
        'caveat': 'A common header schema and exact value slots are necessary '
                  'but not sufficient for valid cross-row realization. '
                  'The residual sentence may contain unsupported facts, '
                  'agreement errors, or page-specific language. No bank '
                  'is approved by this screen.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_ZIP)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['counts'])


if __name__ == '__main__':
    main()
