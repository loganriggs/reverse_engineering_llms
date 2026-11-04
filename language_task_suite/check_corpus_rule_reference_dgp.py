"""Synthetic invariants for source-level three-rule composition and chunks."""

from __future__ import annotations

import random
import re

from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_nonce_copy_dgp import codebook
from corpus_rule_reference_dgp import sample_document, chunk_document


TEXT = (
    'The ABC issued the first report. The team reviewed several records in the '
    'office (and checked the older notes carefully). Other people discussed '
    'the meeting, the schedule, and the next steps before the committee '
    'published a short list.\n'
    '1. The members collected the forms and checked each name.\n'
    '2. The members compared the dates and confirmed the location.\n'
    '3. The members filed the results after the meeting.\n'
    'A later section says ABC returned with an updated summary.')


def main():
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    pool = codebook(tokenizer)
    natural = sample_document(TEXT, random.Random(7), pool)
    assert natural.text == TEXT
    assert set(natural.eligible) == {'list', 'bracket', 'nonce',
                                     'literal_copy'}
    assert set(natural.applied) == {'list', 'bracket'}
    assert set(site['kind'] for chunk in chunk_document(natural, tokenizer)
               for site in chunk['sites']) == {'list', 'bracket'}

    natural_three = sample_document(TEXT, random.Random(7), pool,
                                    literal_copy_rate=1.)
    assert natural_three.text == TEXT
    assert set(natural_three.applied) == {'list', 'bracket', 'literal_copy'}
    assert set(site['kind'] for chunk in chunk_document(natural_three,
                                                        tokenizer)
               for site in chunk['sites']) == set(natural_three.applied)

    style_text = ('The color of the exhibit was recorded in a field report. '
                  'Several people read the report and discussed the result '
                  'with the staff before sending another note to the office. '
                  'The favorite item was mentioned again.')
    style_document = sample_document(style_text, random.Random(9), pool,
                                     list_rate=0., bracket_rate=0.,
                                     natural_style_rate=1.)
    assert style_document.text == style_text
    assert style_document.applied == ('natural_style',)
    assert any(site['kind'] == 'natural_style'
               for chunk in chunk_document(style_document, tokenizer)
               for site in chunk['sites'])
    mixed_style = sample_document(style_text.replace('favorite', 'favourite'),
                                  random.Random(9), pool, list_rate=0.,
                                  bracket_rate=0., natural_style_rate=1.)
    assert mixed_style.applied == ('natural_style',)
    assert mixed_style.sites[0].answer == 'favourite'
    assert any(site['kind'] == 'natural_style'
               for chunk in chunk_document(mixed_style, tokenizer)
               for site in chunk['sites'])

    shifted = sample_document(TEXT, random.Random(7), pool,
                              list_shift_rate=1.)
    assert {site.kind: site.answer for site in shifted.sites}['list'] == '4'
    changed = {i for i, (a, b) in enumerate(zip(TEXT, shifted.text,
                                                strict=True)) if a != b}
    assert changed == {match.start() for match in re.finditer(
        r'(?m)^[123]\.', TEXT)}

    square = sample_document(TEXT, random.Random(7), pool, square_rate=1.)
    bracket = next(site for site in square.sites if site.kind == 'bracket')
    changed = {i for i, (a, b) in enumerate(zip(TEXT, square.text,
                                                strict=True)) if a != b}
    assert changed == {bracket.source[0], bracket.target[0]}
    assert bracket.answer == ']'

    all_rules = sample_document(TEXT, random.Random(7), pool, nonce_rate=1.,
                                list_shift_rate=1., square_rate=1.)
    assert set(all_rules.applied) == {'list', 'bracket', 'nonce'}
    sites = [site for chunk in chunk_document(all_rules, tokenizer)
             for site in chunk['sites']]
    assert {site['kind'] for site in sites} == set(all_rules.applied)
    assert len(next(site for site in sites if site['kind'] == 'nonce')[
        'positions']) == 3
    print('reference DGP natural rule fixtures passed')


if __name__ == '__main__':
    main()
