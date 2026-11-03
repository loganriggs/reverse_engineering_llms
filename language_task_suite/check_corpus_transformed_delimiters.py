"""Edge cases for the natural-parenthetical delimiter transform."""

import random

from corpus_transformed_delimiters import candidates, transform, validate


def check():
    source = ('The report (written by the local council) was published. '
              'It included a note (read by several residents) as well.')
    assert len(candidates(source)) == 2
    round_ex = transform(source, random.Random(0), style='round', pair_index=0)
    square_ex = transform(source, random.Random(0), style='square', pair_index=0)
    assert round_ex and square_ex
    assert validate(round_ex) and validate(square_ex)
    assert round_ex.text == source
    assert square_ex.answer == ']'
    assert square_ex.prefix[square_ex.open_index] == '['
    assert square_ex.text[square_ex.close_index] == ']'
    assert transform(source, random.Random(1), pair_index=0,
                     square_rate=0.).style == 'round'
    assert transform(source, random.Random(1), pair_index=0,
                     square_rate=1.).style == 'square'
    assert not candidates('A nested (phrase with (another set) inside) ends.')
    assert not candidates('A code (x = y + z) appears.')
    assert not candidates('A short (two words) appears.')
    print('Delimiter eligibility and intervention checks passed.')


if __name__ == '__main__':
    check()
