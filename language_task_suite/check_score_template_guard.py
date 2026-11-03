"""Adversarial contract checks for candidate score-sentence guards."""

from corpus_score_sentence_renderer import ScoreTemplate, render
from score_template_guard import safe_for


def template(text, home_win=True):
    return ScoreTemplate(tuple(text.split()), home_win)


def check():
    good = template('The host {WINNER} defeated the visiting {LOSER} '
                    '{WIN_SCORE} - {LOSE_SCORE} .')
    assert safe_for(good, 112, 104)
    assert render(good, 'Boston Celtics', 'New York Knicks',
                  112, 104) == ('The host Boston Celtics defeated the '
                                'visiting New York Knicks 112 - 104 .').split()
    contradiction = template('The {LOSER} defeated the {WINNER} '
                             '{WIN_SCORE} - {LOSE_SCORE} .')
    assert not safe_for(contradiction, 112, 104)
    host_flip = template('The host {WINNER} defeated the {LOSER} '
                         '{WIN_SCORE} - {LOSE_SCORE} .', False)
    assert not safe_for(host_flip, 112, 104)
    narrow = template('The {WINNER} narrowly beat the {LOSER} '
                      '{WIN_SCORE} - {LOSE_SCORE} .')
    assert safe_for(narrow, 112, 109)
    assert not safe_for(narrow, 130, 100)
    extra_fact = template('The {WINNER} defeated the {LOSER} '
                          '{WIN_SCORE} - {LOSE_SCORE} for their fourth '
                          'straight win .')
    assert not safe_for(extra_fact, 112, 104)
    print('Score-role, margin, home/away and extra-claim checks passed.')


if __name__ == '__main__':
    check()
