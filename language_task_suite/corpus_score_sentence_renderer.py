"""Conservative, reversible score-sentence templates from game reports.

This uses the original corpus only as a surface-form prior. A template's
winner/loser/score placeholders have an exact source alignment; the rest of
its claims still require human review before general counterfactual use.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


ALLOWED_CAPITALIZED = frozenset({
    'The', 'A', 'An', 'On', 'In', 'At', 'After', 'Before', 'Despite',
    'With', 'By', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
    'Friday', 'Saturday', 'Sunday', 'NBA', 'Christmas',
    'Thanksgiving', 'All-Star', 'Game',
})
@dataclass(frozen=True)
class ScoreTemplate:
    tokens: tuple[str, ...]
    home_win: bool


def _sentences(tokens):
    sentence = []
    for token in tokens:
        sentence.append(token)
        if token in ('.', '!', '?'):
            yield sentence
            sentence = []


def _find(sentence, phrase):
    return [i for i in range(len(sentence)-len(phrase)+1)
            if sentence[i:i+len(phrase)] == phrase]


def _template(sentence, winner, loser, winning_score, losing_score,
              home_win, counts):
    if not 8 <= len(sentence) <= 42:
        return None
    win_tokens = winner.split()
    lose_tokens = loser.split()
    starts = _find(sentence, win_tokens), _find(sentence, lose_tokens)
    if any(len(found) != 1 for found in starts):
        return None
    pair = [winning_score, '-', losing_score]
    pair_starts = _find(sentence, pair)
    if len(pair_starts) != 1:
        return None
    counts['full_teams_and_winning_score_pair'] += 1
    if [word for word in sentence if word.isdigit()] != [
            winning_score, losing_score]:
        return None
    counts['no_extra_numeric_tokens'] += 1
    spans = [(starts[0][0], starts[0][0]+len(win_tokens),
              ('{WINNER}',)),
             (starts[1][0], starts[1][0]+len(lose_tokens),
              ('{LOSER}',)),
             (pair_starts[0], pair_starts[0]+3,
              ('{WIN_SCORE}', '-', '{LOSE_SCORE}'))]
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return None
    covered = {i for start, end, _ in spans for i in range(start, end)}
    if any(token[:1].isupper() and token not in ALLOWED_CAPITALIZED
           and i not in covered for i, token in enumerate(sentence)):
        return None
    counts['no_other_proper_names'] += 1
    output, cursor = [], 0
    for start, end, replacement in spans:
        output.extend(sentence[cursor:start])
        output.extend(replacement)
        cursor = end
    output.extend(sentence[cursor:])
    template = ScoreTemplate(tuple(output), home_win)
    assert render(template, winner, loser, winning_score,
                  losing_score) == sentence
    return template


def extract(rows):
    counts = Counter()
    templates = set()
    for row in rows:
        try:
            home_score = int(row['home_line']['TEAM-PTS'])
            away_score = int(row['vis_line']['TEAM-PTS'])
        except (KeyError, ValueError):
            continue
        if home_score == away_score:
            continue
        home_win = home_score > away_score
        home = row['home_city']+' '+row['home_name']
        away = row['vis_city']+' '+row['vis_name']
        winner, loser = (home, away) if home_win else (away, home)
        for sentence in _sentences(row['summary']):
            result = _template(sentence, winner, loser,
                               str(max(home_score, away_score)),
                               str(min(home_score, away_score)),
                               home_win, counts)
            if result is not None:
                templates.add(result)
    return tuple(sorted(templates, key=lambda row:
                        (row.home_win, row.tokens))), counts


def render(template, winner, loser, winning_score, losing_score):
    assert int(winning_score) > int(losing_score)
    values = {'{WINNER}': winner.split(),
              '{LOSER}': loser.split(),
              '{WIN_SCORE}': [str(winning_score)],
              '{LOSE_SCORE}': [str(losing_score)]}
    output = []
    for token in template.tokens:
        output.extend(values.get(token, [token]))
    return output
