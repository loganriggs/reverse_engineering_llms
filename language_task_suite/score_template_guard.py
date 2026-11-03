"""First conservative semantic guard for corpus-derived score sentences.

This admits only a simple result assertion. It deliberately sacrifices many
fluent templates whose extra claims would require additional world state.
"""

from __future__ import annotations

from corpus_score_sentence_renderer import ScoreTemplate


UNMODELLED = frozenset({
    'season', 'streak', 'overtime', 'playoff', 'injury', 'injured',
    'depleted', 'undermanned', 'short-handed', 'comeback', 'halftime',
    'quarter', 'lead', 'runs', 'run', 'record', 'straight', 'consecutive',
    'opening', 'opener', 'first', 'second', 'third', 'fourth', 'last',
    'previous', 'remaining', 'franchise', 'best', 'worst', 'momentum',
    'buzzer', 'homecourt', 'roadtrip', 'road', 'hot', 'cold', 'slump',
    'pesky', 'fouled', 'foul', 'and', 'but', 'while', 'however',
    'despite', 'although', 'though', 'because', 'after', 'before',
    'finally', 'eventually', 'preseason', 'postseason', 'boys',
    'teal', 'stars', 'star', 'shining', 'satisfying', 'probably',
    'rivial', 'what', 'battle', 'eighth', 'seed', 'contenders',
    'dramatic', 'dominating', 'divisional', 'rivals',
})
NEUTRAL_WIN = frozenset({'defeated', 'beat', 'topped', 'downed'})
CLOSE_WIN = frozenset({'edged', 'squeaked'})
LARGE_WIN = frozenset({'steamrolled', 'routed'})


def safe_for(template: ScoreTemplate, winning_score: int,
             losing_score: int):
    """Return whether a result claim is supported by the sampled plan.

    This is a conservative *candidate* guard, not a semantic parser.
    Human review of accepted templates remains necessary.
    """
    margin = int(winning_score)-int(losing_score)
    if margin <= 0:
        return False
    tokens = list(template.tokens)
    lower = [token.lower() for token in tokens]
    if len(tokens) > 27 or UNMODELLED.intersection(lower):
        return False
    win, lose = tokens.index('{WINNER}'), tokens.index('{LOSER}')
    for index, is_home in ((win, template.home_win),
                           (lose, not template.home_win)):
        context = lower[max(0, index-3):index]
        if ('host' in context and not is_home or
                'visiting' in context and is_home):
            return False
    if 'narrowly' in lower and margin > 6:
        return False
    if 'blew' in lower or 'blowout' in lower:
        if margin < 15:
            return False
    if 'cruised' in lower or 'easy' in lower or 'comfortably' in lower:
        if margin < 12:
            return False
    if ('easily' in lower or 'handily' in lower) and margin < 12:
        return False
    if 'hung' in lower and margin > 6:
        return False
    if win < lose:
        between = lower[win+1:lose]
        if NEUTRAL_WIN.intersection(between):
            return True
        if CLOSE_WIN.intersection(between):
            return margin <= 6
        if LARGE_WIN.intersection(between):
            return margin >= 15
        if 'took' in between and 'down' in between:
            return True
        if 'blew' in between and 'out' in between:
            return margin >= 15
        return False
    between = lower[lose+1:win]
    return ('to' in between and
            ('lost' in between or 'fell' in between) and
            not any(word in lower for word in ('heartbreaker', 'nail',
                                                'biter')))
