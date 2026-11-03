"""Independent invariants for document-wide spelling-style transform."""

from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from corpus_spelling_tokenization import aligned_pair
from corpus_transformed_spelling_style import (
    occurrences, reverse_transform, strict_candidate, transform,
)


def check():
    filler = 'The report considered several different options. '*8
    source = ('The Color Center opened. The painter chose colour. '+filler+
              'Visitors reached the center. Later the organization '
              'discussed colors and theatres.')
    us, uk = transform(source, 'us'), transform(source, 'uk')
    assert us is not None and uk is not None
    assert (us.cue_root, us.target_root) == ('color', 'center')
    assert us.gap_words == uk.gap_words
    assert reverse_transform(us) == reverse_transform(uk) == source
    assert 'The Color Center' in us.text and 'The Color Center' in uk.text
    assert 'painter chose color' in us.text
    assert 'painter chose colour' in uk.text
    assert us.target_word == 'center' and uk.target_word == 'centre'
    assert us.text[:us.cue_span[0]] == uk.text[:uk.cue_span[0]]
    assert us.text[us.cue_span[1]:us.target_span[0]] == \
           uk.text[uk.cue_span[1]:uk.target_span[0]]
    assert len(occurrences('The Color Center.')) == 0
    assert strict_candidate('color '+filler+'colours') is None
    assert transform('color '+filler+'colours', 'uk') is None
    pair, reason = aligned_pair(source, Tokenizer.from_file(
        str(DEFAULT_TOKENIZER)), n_ctx=256)
    assert reason == 'aligned' and pair is not None
    us_row, uk_row = pair['us'], pair['uk']
    assert us_row['cue_root'] == 'color'
    assert us_row['target_root'] == 'center'
    assert us_row['target_id'] != uk_row['target_id']
    site = us_row['site_position']
    assert site == uk_row['site_position']
    assert (us_row['x'][:site+1] != uk_row['x'][:site+1]
            ).nonzero().flatten().tolist() == [us_row['cue_position']]
    try:
        transform(source, 'american')
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown style should fail')
    print('Spelling-style transform checks passed.')


if __name__ == '__main__':
    check()
