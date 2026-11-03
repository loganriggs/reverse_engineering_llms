"""Natural sports-report carrier with one controlled, visible margin rule.

This is an empirical-carrier bridge, not a full symbolic language DGP. Source
reports are sampled from a fixed corpus; only the inserted result sentence
and final margin site have explicit generative semantics.
"""

from __future__ import annotations

import random
import re
from datetime import datetime
from dataclasses import dataclass, replace
from functools import lru_cache

from corpus_score_sentence_renderer import _sentences, render
from score_template_guard import safe_for
from sports_document_dgp import (
    SCORE_PAIR, _detokenize, game_partition, resources,
)
from audit_rotowire_document_source import game_key, score_pair_present


NUMBER_WORDS = frozenset({
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
    'twenty', 'thirty', 'forty',
})
WEEKDAYS = frozenset({
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
    'Saturday', 'Sunday',
})
QUERY_FORMS = (
    'The final winning margin was {MARGIN} points.',
    'The winner finished {MARGIN} points ahead.',
    'The final gap between the teams was {MARGIN} points.',
    'In the final score, the winning side led by {MARGIN} points.',
)
COPY_QUERY_FORMS = (
    'The winning side scored {SCORE} points.',
    'The higher final total was {SCORE} points.',
    'The winner finished on {SCORE} points.',
    'The winning team ended with {SCORE} points.',
)


@dataclass(frozen=True)
class Plan:
    source: str
    source_game_key: tuple
    winner: str
    loser: str
    high: int
    low: int
    score_template: object
    carrier: tuple[str, ...]
    query_form: str
    query_kind: str = 'margin'


@dataclass(frozen=True)
class Site:
    start: int
    end: int
    answer: str
    candidates: tuple[str, ...]
    kind: str = 'margin'


@dataclass(frozen=True)
class Document:
    text: str
    site: Site
    plan: Plan


def _contains_score_pair(sentence, high, low):
    wanted = {(str(high), str(low)), (str(low), str(high))}
    return any(middle in ('-', 'to', '–') and (left, right) in wanted
               for left, middle, right in
               zip(sentence, sentence[1:], sentence[2:]))


def _first_sentence_asserts_result(row, home_score, away_score):
    """Conservative surface check for a full-team, directed result lead."""
    sentence = next(_sentences(row['summary']), [])
    expected_day = datetime.strptime(row['day'], '%m_%d_%y').strftime('%A')
    observed_days = WEEKDAYS.intersection(sentence)
    if observed_days and observed_days != {expected_day}:
        return False
    high, low = max(home_score, away_score), min(home_score, away_score)
    if not _contains_score_pair(sentence, high, low):
        return False
    home = (row['home_city']+' '+row['home_name']).split()
    away = (row['vis_city']+' '+row['vis_name']).split()
    winner, loser = (home, away) if home_score > away_score \
        else (away, home)
    def locations(phrase):
        return [i for i in range(len(sentence)-len(phrase)+1)
                if sentence[i:i+len(phrase)] == phrase]
    win_at, lose_at = locations(winner), locations(loser)
    if len(win_at) != 1 or len(lose_at) != 1:
        return False
    lower = [token.lower() for token in sentence]
    if win_at[0] < lose_at[0]:
        between = lower[win_at[0]+len(winner):lose_at[0]]
        return (any(word in between for word in
                    ('beat', 'defeated', 'topped', 'downed', 'edged',
                     'routed', 'outlasted')) or
                ('took' in between and 'down' in between) or
                ('blew' in between and 'out' in between))
    between = lower[lose_at[0]+len(loser):win_at[0]]
    return 'to' in between and ('fell' in between or 'lost' in between)


def _carrier(row, high, low, words):
    kept = []
    for sentence in _sentences(row['summary']):
        if not _contains_score_pair(sentence, high, low):
            kept.extend(sentence)
            if len(kept) >= words:
                break
    if not words <= len(kept) <= words+60:
        return None
    tokens = tuple(kept)
    margin = high-low
    if any(token in {str(high), str(low), str(margin),
                     str(high+1), str(margin+1)} for token in tokens):
        return None
    lower = [token.lower() for token in tokens]
    # A spelled-out numeric margin elsewhere in the source would make the
    # +1 score edit contradictory. This is intentionally overconservative.
    if any(token in NUMBER_WORDS and any(
            next_token.lower().startswith('point') for next_token in
            lower[i+1:i+4]) for i, token in enumerate(lower)):
        return None
    text = _detokenize(tokens)
    for a, b in ((high, low), (low, high)):
        if re.search(r'\b'+str(a)+r'\s*[-–]\s*'+str(b)+r'\b', text):
            return None
    if margin in (6, 11, 14):
        return None
    return tokens


@lru_cache(maxsize=8)
def eligible(source, group='train', words=100):
    if group not in ('train', 'heldout') or words < 25:
        raise ValueError('Invalid carrier group or length')
    rows = resources()[0][source][0]
    score_templates = resources()[0]['rotowire'][1]
    output = []
    for row in rows:
        key = game_key(row)
        if game_partition(key) != group:
            continue
        # Box-score/summary alignment is noisy, especially in SBNation.
        # Demand that the human report itself prints the supplied final score.
        if not score_pair_present(row):
            continue
        try:
            home_score = int(row['home_line']['TEAM-PTS'])
            away_score = int(row['vis_line']['TEAM-PTS'])
        except (KeyError, ValueError):
            continue
        if home_score == away_score:
            continue
        if not _first_sentence_asserts_result(row, home_score, away_score):
            continue
        high, low = max(home_score, away_score), min(home_score, away_score)
        home_win = home_score > away_score
        if not any(t.home_win == home_win and safe_for(t, high, low) and
                   safe_for(t, high+1, low) for t in score_templates):
            continue
        carrier = _carrier(row, high, low, words)
        if carrier is None:
            continue
        home = row['home_city']+' '+row['home_name']
        away = row['vis_city']+' '+row['vis_name']
        winner, loser = (home, away) if home_win else (away, home)
        output.append((key, winner, loser, high, low, carrier, home_win))
    return tuple(output)


def sample_plan(rng: random.Random, source='rotowire', group='train',
                carrier_words=100, query_entropy=1.,
                query_kind='margin'):
    if source not in ('rotowire', 'sbnation') or not 0 <= query_entropy <= 1:
        raise ValueError('Invalid source or query entropy')
    if query_kind not in ('margin', 'winning_score'):
        raise ValueError(query_kind)
    rows = eligible(source, group, carrier_words)
    if not rows:
        raise RuntimeError('No eligible carrier reports')
    key, winner, loser, high, low, carrier, home_win = rng.choice(rows)
    templates = resources()[0]['rotowire'][1]
    candidates = [t for t in templates if t.home_win == home_win and
                  safe_for(t, high, low) and safe_for(t, high+1, low)]
    score_template = rng.choice(candidates)
    weekday = datetime.strptime(key[0], '%m_%d_%y').strftime('%A')
    score_template = replace(score_template, tokens=tuple(
        weekday if token in WEEKDAYS else token
        for token in score_template.tokens))
    forms = QUERY_FORMS if query_kind == 'margin' else COPY_QUERY_FORMS
    query_form = rng.choice(forms) if rng.random() < query_entropy \
        else forms[0]
    return Plan(source, key, winner, loser, high, low,
                score_template, carrier, query_form, query_kind)


def realize(plan: Plan):
    if not safe_for(plan.score_template, plan.high, plan.low):
        raise ValueError('Unsafe score template for plan')
    result = _detokenize(render(plan.score_template, plan.winner,
                                plan.loser, plan.high, plan.low))
    carrier = _detokenize(plan.carrier).replace('*NEWPARAGRAPH*', '\n\n')
    marker = '{MARGIN}' if plan.query_kind == 'margin' else '{SCORE}'
    query = plan.query_form
    assert query.count(marker) == 1
    answer = str(plan.high-plan.low) if plan.query_kind == 'margin' \
        else str(plan.high)
    prefix, suffix = query.split(marker)
    text_before = result+' '+carrier+' '+prefix
    start = len(text_before)
    text = text_before+answer+suffix
    candidates = tuple(map(str, range(1, 61))) if plan.query_kind == 'margin' \
        else tuple(map(str, range(60, 161)))
    return Document(text, Site(start, start+len(answer), answer,
                               candidates, plan.query_kind), plan)


def visible_answer(prefix, kind='margin'):
    first_sentence = prefix.split('.')[0]
    pairs = SCORE_PAIR.findall(first_sentence)
    if len(pairs) != 1:
        raise ValueError('Expected one visible final-score pair')
    high, low = map(int, pairs[0])
    if high <= low:
        raise ValueError('Expected winning score first')
    if kind == 'margin':
        return str(high-low)
    if kind == 'winning_score':
        return str(high)
    raise ValueError(kind)


def visible_margin(prefix):
    return visible_answer(prefix, 'margin')


def increment_winning_score(plan):
    return replace(plan, high=plan.high+1)
