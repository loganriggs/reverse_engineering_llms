"""Table-row context followed by a human-corrected ToTTo sentence.

This empirical DGP only labels exact, unique highlighted-cell mentions in
the corrected sentence. It does not claim every token has a symbolic oracle
or that every sentence fact is present in the selected row.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Site:
    column: str
    value: str
    span: tuple[int, int]
    row: int


@dataclass(frozen=True)
class Example:
    text: str
    sites: tuple[Site, ...]
    page: str
    section: str
    example_id: str


def _cell(table, r, c):
    try:
        return table[r][c]
    except (IndexError, TypeError):
        return None


def _aligned_header(table, r):
    """Use only an explicit, complete header row with matching cell count.

    Ragged Wiki tables often omit a repeated leading cell. Positional
    inheritance then assigns every later value to the wrong column.
    """
    width = len(table[r])
    for above in range(r-1, -1, -1):
        row = table[above]
        if any(cell.get('is_header') for cell in row):
            if len(row) != width or not all(
                    cell.get('is_header') and cell.get('value', '').strip()
                    for cell in row):
                return None
            return [cell['value'].strip() for cell in row]
    return None


def _mentions(text, value):
    return list(re.finditer(r'(?<!\w)' + re.escape(value) + r'(?!\w)',
                           text, re.I))


def extract(row, *, min_sites=2, max_columns=12,
            max_cell_chars=90):
    annotations = row.get('sentence_annotations') or []
    if not annotations:
        return None
    sentence = annotations[0].get('final_sentence', '').strip()
    table = row.get('table') or []
    highlights = row.get('highlighted_cells') or []
    if not sentence or len(highlights) < min_sites:
        return None
    grouped = {}
    for r, c in highlights:
        grouped.setdefault(r, set()).add(c)
    eligible_rows = [(r, columns) for r, columns in grouped.items()
                     if len(columns) >= min_sites and r < len(table)]
    if len(eligible_rows) != 1:
        return None
    r, highlighted_columns = eligible_rows[0]
    cells = table[r]
    if not 2 <= len(cells) <= max_columns:
        return None
    raw_headers = _aligned_header(table, r)
    if raw_headers is None:
        return None
    headers, values = [], []
    for c, cell in enumerate(cells):
        value = re.sub(r'\s+', ' ', cell.get('value', '')).strip()
        header = re.sub(r'\s+', ' ', raw_headers[c]).strip()
        if (not value or not header or len(value) > max_cell_chars or
                len(header) > max_cell_chars or '\n' in value or
                '|' in value or '|' in header or '#' in value or
                'http' in value.lower()):
            return None
        headers.append(header)
        values.append(value)
    sites = []
    if len({header.casefold() for header in headers}) != len(headers):
        return None
    for c in sorted(highlighted_columns):
        value = values[c]
        if (len(value) < 3 or len(_mentions(sentence, value)) != 1 or
                sum(x.casefold() == value.casefold()
                    for x in values) != 1):
            continue
        match = _mentions(sentence, value)[0]
        sites.append((c, value, match.span()))
    if len(sites) < min_sites:
        return None
    # A value that names another selected cell cannot be assigned a unique
    # target relation from this sentence alone.
    for i, (_, _, (lo, hi)) in enumerate(sites):
        if any(not (hi <= other[2][0] or other[2][1] <= lo)
               for j, other in enumerate(sites) if j != i):
            return None
    title = re.sub(r'\s+', ' ', row.get('table_page_title') or '').strip()
    section = re.sub(r'\s+', ' ', row.get('table_section_title') or '').strip()
    if not title:
        return None
    prefix = ('Page: ' + title + '\n' +
              ('Section: ' + section + '\n' if section else '') +
              '| ' + ' | '.join(headers) + ' |\n' +
              '| ' + ' | '.join(values) + ' |\n' +
              'Description: ')
    result_sites = tuple(Site(headers[c], value,
                              (len(prefix)+span[0], len(prefix)+span[1]), r)
                         for c, value, span in sites)
    return Example(prefix + sentence, result_sites, title, section,
                   str(row.get('example_id', '')))


def check_visible(example):
    """Check each target against the displayed row without source metadata."""
    lines = example.text.splitlines()
    header_line = next((line for line in lines
                        if line.startswith('| ') and len(line) > 3), None)
    value_line = next((line for line in lines
                       if line.startswith('| ') and line != header_line), None)
    if not header_line or not value_line:
        raise ValueError('Missing visible table')
    headers = [v.strip() for v in header_line.strip('|').split('|')]
    values = [v.strip() for v in value_line.strip('|').split('|')]
    if len(headers) != len(values):
        raise ValueError('Table shape mismatch')
    table = list(zip(headers, values))
    for site in example.sites:
        if (site.column, site.value) not in table:
            raise ValueError('Source cell missing from visible row')
        if example.text[slice(*site.span)].casefold() != site.value.casefold():
            raise ValueError('Sentence target is not the visible value')
    return True
