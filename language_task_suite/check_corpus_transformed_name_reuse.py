"""Focused edge-case checks for the natural-corpus name transform."""

import random

from corpus_transformed_name_reuse import candidates, reverse_transform, transform


def check():
    filler = 'the article described several local events and outcomes ' * 4
    valid = f'Sarah Palin spoke. {filler}Sarah Palin returned.'
    options, _ = candidates(valid)
    assert options and options[0][0] == 'Sarah'
    example = transform(valid, random.Random(7))
    assert example and reverse_transform(example) == valid
    assert example.sampled_name in example.prefix
    assert example.text[slice(*example.answer_span)] == example.sampled_name

    two_people = f'Henry David spoke. {filler}Henry Walke returned.'
    assert not candidates(two_people)[0]
    standalone = f'Sarah Palin spoke. {filler}Sarah returned.'
    assert not candidates(standalone)[0]
    assert candidates(standalone, identity_policy='first_name')[0]
    middle_name = (f'Benjamin Hingley spoke. {filler}'
                   'George Benjamin Hingley returned.')
    assert not candidates(middle_name)[0]

    collision = valid+' Tara spoke.'
    try:
        transform(collision, random.Random(7), forced_source='Sarah',
                  forced_target='Tara')
    except ValueError:
        pass
    else:
        raise AssertionError('Existing target name was accepted')
    print('Name-reuse identity and counterfactual checks passed.')


if __name__ == '__main__':
    check()
