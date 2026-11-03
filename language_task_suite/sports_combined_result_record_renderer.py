"""Guarded corpus-derived result leads with parenthetical team records.

Only a score, two complete team names, and one or two immediately attached
box-score-matching W-L pairs are lifted from training-game source sentences.
The remaining score clause must pass the existing conservative result guard.
This is a restricted form bank, not a general semantic parser.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import random
import re

from audit_rotowire_document_source import game_key
from corpus_score_sentence_renderer import (
    ScoreTemplate, _find, _sentences, _template,
)
from score_template_guard import safe_for
from sports_document_dgp import game_partition, resources


WEEKDAYS = frozenset(('Monday', 'Tuesday', 'Wednesday', 'Thursday',
                      'Friday', 'Saturday', 'Sunday'))
SAFE_WORDS = frozenset({
    'the', 'a', 'defeated', 'beat', 'topped', 'downed', 'edged', 'took',
    'down', 'blew', 'out', 'to', 'lost', 'fell', 'on', 'by', 'score', 'of',
    'visiting', 'host', 'at', 'home', 'narrowly', 'easily', 'handily',
}) | frozenset(day.lower() for day in WEEKDAYS)
PUNCTUATION = frozenset(('.', ',', '-', ';', ':', '!', '?'))


@dataclass(frozen=True)
class CombinedTemplate:
    score: ScoreTemplate
    winner_record: bool
    loser_record: bool


def _row_records(row):
    out = {}
    for side in ('home', 'vis'):
        line = row[side+'_line']
        team = row[side+'_city']+' '+row[side+'_name']
        try:
            out[team] = (int(line['TEAM-WINS']),
                         int(line['TEAM-LOSSES']))
        except (KeyError, TypeError, ValueError):
            return None
    return out if len(out) == 2 else None


def _extract(sentence, winner, loser, high, low, home_win, records):
    if len(_find(sentence, [str(high), '-', str(low)])) != 1:
        return None
    spans, seen = [], []
    for team in (winner, loser):
        starts = _find(sentence, team.split())
        if len(starts) != 1:
            return None
        after = starts[0]+len(team.split())
        w, l = records[team]
        if sentence[after:after+5] == ['(', str(w), '-', str(l), ')']:
            spans.append((after, after+5))
            seen.append(team)
    if not seen:
        return None
    cleaned = [token for i, token in enumerate(sentence)
               if not any(start <= i < end for start, end in spans)]
    score = _template(cleaned, winner, loser, str(high), str(low),
                      home_win, Counter())
    if score is None or not safe_for(score, high, low):
        return None
    if any(token.lower() not in SAFE_WORDS
           for token in score.tokens
           if token not in PUNCTUATION and not token.startswith('{')):
        return None
    candidate = CombinedTemplate(score, winner in seen, loser in seen)
    assert render_tokens(candidate, winner, loser, high, low,
                         records[winner], records[loser]) == sentence
    return candidate


@lru_cache(maxsize=1)
def bank():
    counts = Counter()
    for row in resources()[0]['rotowire'][0]:
        if game_partition(game_key(row)) != 'train':
            continue
        records = _row_records(row)
        if not records:
            continue
        try:
            hs, vs = (int(row[side+'_line']['TEAM-PTS'])
                      for side in ('home', 'vis'))
        except (KeyError, TypeError, ValueError):
            continue
        if hs == vs:
            continue
        home = row['home_city']+' '+row['home_name']
        away = row['vis_city']+' '+row['vis_name']
        winner, loser = (home, away) if hs > vs else (away, home)
        for sentence in _sentences(row['summary']):
            template = _extract(sentence, winner, loser,
                                max(hs, vs), min(hs, vs), hs > vs,
                                records)
            if template is not None:
                counts[template] += 1
    if not counts:
        raise RuntimeError('No fact-matched combined result/record forms')
    return counts


def render_tokens(template, winner, loser, high, low,
                  winner_record, loser_record):
    values = {'{WINNER}': winner.split(), '{LOSER}': loser.split(),
              '{WIN_SCORE}': [str(high)], '{LOSE_SCORE}': [str(low)]}
    out = []
    for token in template.score.tokens:
        if token == '{WINNER}' and template.winner_record:
            w, l = winner_record
            out.extend(values[token]+['(', str(w), '-', str(l), ')'])
        elif token == '{LOSER}' and template.loser_record:
            w, l = loser_record
            out.extend(values[token]+['(', str(w), '-', str(l), ')'])
        else:
            out.extend(values.get(token, [token]))
    return out


def detokenize(tokens):
    text = ' '.join(tokens)
    for char in ('.', ',', '!', '?', ';', ':', ')'):
        text = text.replace(' '+char, char)
    return text.replace('( ', '(').replace(' - ', '-')


def render(plan, rng: random.Random):
    base = plan.base
    candidates = [(template, count) for template, count in bank().items()
                  if template.score.home_win == base.home_win and
                  safe_for(template.score, base.high_score,
                           base.low_score) and
                  safe_for(template.score, base.high_score+1,
                           base.low_score)]
    if not candidates:
        raise RuntimeError('No compatible combined result/record form')
    template = rng.choices([row[0] for row in candidates],
                           [row[1]**.5 for row in candidates])[0]
    expected_day = datetime.strptime(
        base.source_game_key[0], '%m_%d_%y').strftime('%A')
    score = ScoreTemplate(tuple(expected_day if token in WEEKDAYS else token
                                for token in template.score.tokens),
                          template.score.home_win)
    template = CombinedTemplate(score, template.winner_record,
                                template.loser_record)
    text = detokenize(render_tokens(
        template, base.winner, base.loser, base.high_score,
        base.low_score, plan.after[base.winner], plan.after[base.loser]))
    return text, template


def visible_records(sentence, teams):
    """Read parenthetical records attached to complete visible team names."""
    out = {}
    for team in teams:
        found = re.findall(re.escape(team)+r'\s*\((\d+)-(\d+)\)',
                           sentence)
        if len(found) > 1:
            raise ValueError('Duplicate team record')
        if found:
            out[team] = tuple(map(int, found[0]))
    return out


def strip_records(sentence):
    return re.sub(r'\s*\(\d+-\d+\)', '', sentence)


def parse_visible_result(sentence):
    """Infer winner and margin from one visible combined sentence only."""
    cleaned = strip_records(sentence)
    score_pair = re.search(r'\b(\d{2,3})-(\d{2,3})\b', cleaned)
    if score_pair is None:
        raise ValueError('Missing final score')
    high, low = map(int, score_pair.groups())
    teams = [team for team in resources()[1]
             if re.search(r'(?<!\w)'+re.escape(team)+r'(?!\w)', sentence)]
    if len(teams) != 2:
        raise ValueError('Expected two complete teams')
    records = visible_records(sentence, teams)
    def normalize_day(text):
        return re.sub(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|'
                      r'Saturday|Sunday)\b', '<DAY>', text)
    winners = set()
    for template in bank():
        if not safe_for(template.score, high, low):
            continue
        for winner, loser in (teams, teams[::-1]):
            if template.winner_record != (winner in records) or \
                    template.loser_record != (loser in records):
                continue
            rendered = detokenize(render_tokens(
                template, winner, loser, high, low,
                records.get(winner, (0, 0)), records.get(loser, (0, 0))))
            if normalize_day(rendered) == normalize_day(sentence):
                winners.add(winner)
    if len(winners) != 1:
        raise ValueError(f'Ambiguous combined result: {winners}')
    return high, low, next(iter(winners))
