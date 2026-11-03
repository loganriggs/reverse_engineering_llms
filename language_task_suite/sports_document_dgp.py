"""Sampled sports-document pilot with three visible, linked rule sites.

Score sentences are guarded corpus-derived forms. Player and recap forms are
small typed grammars. This is an experimental DGP, not an accepted realistic
training distribution or an exact full next-token oracle.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from audit_rotowire_document_source import load
from corpus_score_sentence_renderer import ScoreTemplate, extract, render
from score_template_guard import safe_for


ARCHIVE_DIR = Path('/home/loganriggs/Coding/neuron_circuits/'
                   'artifacts/rotowire')
PLAYER = re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+$')
SCORE_PAIR = re.compile(r'\b(\d{2,3})\s*-\s*(\d{2,3})\b')
PLAYER_FACT = re.compile(
    r'\b([A-Z][a-z]+ [A-Z][a-z]+) '
    r'(?:scored|finished with|posted|contributed|added|had) '
    r'(\d+) points\b')
PAREN_FACT = re.compile(r'\b([A-Z][a-z]+ [A-Z][a-z]+) '
                        r'\((\d+) points\)')
SCORING_FACT = re.compile(r'\b([A-Z][a-z]+ [A-Z][a-z]+) '
                          r'scoring (\d+) points\b')
MARKERS = ('WINNER_ANSWER', 'MARGIN_ANSWER', 'TOP_PLAYER_ANSWER')


@dataclass(frozen=True)
class Plan:
    home: str
    away: str
    home_win: bool
    high_score: int
    low_score: int
    players: tuple[str, ...]
    points: tuple[int, ...]
    score_template: ScoreTemplate
    style_source: str
    surface_seed: int

    @property
    def winner(self):
        return self.home if self.home_win else self.away

    @property
    def loser(self):
        return self.away if self.home_win else self.home

    @property
    def top_player(self):
        return self.players[self.points.index(max(self.points))]


@dataclass(frozen=True)
class Site:
    kind: str
    start: int
    end: int
    answer: str
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    text: str
    sites: tuple[Site, ...]
    plan: Plan


@lru_cache(maxsize=1)
def resources():
    corpora = {}
    teams = set()
    players = Counter()
    for name in ('rotowire', 'sbnation'):
        train = load(ARCHIVE_DIR/f'{name}.tar.bz2', name)['train']
        templates, _ = extract(train)
        corpora[name] = (train, templates)
        for row in train:
            teams.add(row['home_city']+' '+row['home_name'])
            teams.add(row['vis_city']+' '+row['vis_name'])
            if name == 'rotowire':
                for player in row['box_score']['PLAYER_NAME'].values():
                    if PLAYER.fullmatch(player):
                        players[player] += 1
    return corpora, tuple(sorted(teams)), players


def sample_plan(rng, sbnation_rate=.25, player_exponent=.5,
                margin_limit=25):
    if not 0 <= sbnation_rate <= 1 or player_exponent < 0 \
            or not 2 <= margin_limit <= 35:
        raise ValueError('Invalid sports DGP controls')
    corpora, _, players = resources()
    source = 'sbnation' if rng.random() < sbnation_rate else 'rotowire'
    train, templates = corpora[source]
    # TEAM_CITY is the only team affiliation carried in these box scores.
    # Same-city games are ambiguous under that field, so exclude them.
    row = rng.choice(train)
    while row['home_city'] == row['vis_city']:
        row = rng.choice(train)
    home = row['home_city']+' '+row['home_name']
    away = row['vis_city']+' '+row['vis_name']
    home_win = rng.random() < .55
    high = rng.randint(100, 130)
    low = high-rng.randint(1, margin_limit)
    options = [template for template in templates
               if template.home_win == home_win and
               safe_for(template, high, low) and
               safe_for(template, high+1, low)]
    if not options:
        raise RuntimeError('No guarded score realization for sampled world')
    template = rng.choice(options)
    # Sample three players from the winning side and two from the losing
    # side. Each synthetic point total is 8..32, so their respective sums
    # cannot exceed the sampled team scores (high>=100, low>=65).
    box = row['box_score']
    winner_city = row['home_city'] if home_win else row['vis_city']
    loser_city = row['vis_city'] if home_win else row['home_city']
    chosen = []
    for city, count in ((winner_city, 3), (loser_city, 2)):
        roster = tuple(dict.fromkeys(
            name for key, name in box['PLAYER_NAME'].items()
            if box['TEAM_CITY'].get(key) == city and
            PLAYER.fullmatch(name)))
        if len(roster) < count:
            raise RuntimeError('Sampled game has too few usable players')
        remaining = list(roster)
        for _ in range(count):
            weights = [max(1, players[name])**player_exponent
                       for name in remaining]
            name = rng.choices(remaining, weights)[0]
            chosen.append(name)
            remaining.remove(name)
    rng.shuffle(chosen)
    points = tuple(rng.sample(range(8, 33), 5))
    return Plan(home, away, home_win, high, low,
                tuple(chosen), points, template, source,
                rng.randrange(2**63))


def _detokenize(tokens):
    text = ' '.join(tokens)
    for punctuation in ('.', ',', '!', '?', ';', ':'):
        text = text.replace(' '+punctuation, punctuation)
    return text.replace(' - ', '-')


STAT_FORMS = (
    '{name} scored {points} points.',
    '{name} finished with {points} points.',
    '{name} posted {points} points.',
    '{name} contributed {points} points.',
)
PAIR_FORMS = (
    '{a} scored {pa} points, while {b} added {pb} points.',
    '{a} finished with {pa} points; {b} posted {pb} points.',
    '{a} ({pa} points) and {b} ({pb} points) were both noted.',
    '{a} contributed {pa} points, with {b} scoring {pb} points.',
)
RECAP_FORMS = {
    'winner': (
        'The winning team was WINNER_ANSWER.',
        'The result gave the victory to WINNER_ANSWER.',
        'In the final result, WINNER_ANSWER won.',
    ),
    'margin': (
        'The winning margin was MARGIN_ANSWER points.',
        'That left a margin of MARGIN_ANSWER points.',
        'The two final totals differed by MARGIN_ANSWER points.',
    ),
    'top_player': (
        'Among the five players listed, TOP_PLAYER_ANSWER scored the most.',
        'The highest scorer in this group was TOP_PLAYER_ANSWER.',
        'Of the players named here, TOP_PLAYER_ANSWER had the most points.',
    ),
}


def realize(plan, form_entropy=1., paragraph_rate=.12,
            pair_rate=1.):
    if not 0 <= form_entropy <= 1 or not 0 <= paragraph_rate <= 1 \
            or not 0 <= pair_rate <= 1:
        raise ValueError('Invalid surface controls')
    if len(plan.players) != 5 or len(plan.points) != 5 or \
            len(set(plan.players)) != 5 or len(set(plan.points)) != 5:
        raise ValueError('Five distinct player names and point totals required')
    if not safe_for(plan.score_template,
                    plan.high_score, plan.low_score):
        raise ValueError('Score template incompatible with sampled scores')
    rng = random.Random(plan.surface_seed)
    sentences = [_detokenize(render(
        plan.score_template, plan.winner, plan.loser,
        plan.high_score, plan.low_score))]
    order = list(range(5))
    rng.shuffle(order)
    while order:
        index = order.pop()
        if order and rng.random() < pair_rate:
            second = order.pop()
            form = rng.choice(PAIR_FORMS) if rng.random() < form_entropy \
                else PAIR_FORMS[0]
            sentences.append(form.format(
                a=plan.players[index], pa=plan.points[index],
                b=plan.players[second], pb=plan.points[second]))
        else:
            form = rng.choice(STAT_FORMS) if rng.random() < form_entropy \
                else STAT_FORMS[0]
            sentences.append(form.format(name=plan.players[index],
                                         points=plan.points[index]))
    recap_order = list(RECAP_FORMS)
    rng.shuffle(recap_order)
    for kind in recap_order:
        choices = RECAP_FORMS[kind]
        form = rng.choice(choices) if rng.random() < form_entropy \
            else choices[0]
        sentences.append(form)
    text = sentences[0]
    for sentence in sentences[1:]:
        text += ('\n\n' if rng.random() < paragraph_rate else ' ')+sentence
    values = {
        'WINNER_ANSWER': ('winner', plan.winner,
                          (plan.home, plan.away)),
        'MARGIN_ANSWER': ('margin',
                          str(plan.high_score-plan.low_score),
                          tuple(map(str, range(1, 36)))),
        'TOP_PLAYER_ANSWER': ('top_player', plan.top_player,
                              plan.players),
    }
    sites, pieces, cursor, length = [], [], 0, 0
    for match in re.finditer('|'.join(MARKERS), text):
        chunk = text[cursor:match.start()]
        pieces.append(chunk)
        length += len(chunk)
        kind, answer, candidates = values[match.group()]
        sites.append(Site(kind, length, length+len(answer),
                          answer, candidates))
        pieces.append(answer)
        length += len(answer)
        cursor = match.end()
    pieces.append(text[cursor:])
    return Document(''.join(pieces), tuple(sites), plan)


def _visible_score(prefix):
    return _parse_score_sentence(prefix.split('.')[0]+'.')


@lru_cache(maxsize=100_000)
def _parse_score_sentence(first):
    pair = SCORE_PAIR.search(first)
    if pair is None:
        raise ValueError('Missing score pair')
    high, low = map(int, pair.groups())
    _, teams, _ = resources()
    mentioned = [team for team in teams
                 if re.search(r'(?<!\w)'+re.escape(team)+r'(?!\w)', first)]
    if len(mentioned) != 2:
        raise ValueError(f'Expected two complete teams, found {mentioned}')
    corpora, _, _ = resources()
    winners = set()
    for _, templates in corpora.values():
        for template in templates:
            if not safe_for(template, high, low):
                continue
            for winner, loser in (mentioned, mentioned[::-1]):
                if _detokenize(render(template, winner, loser,
                                      high, low)) == first:
                    winners.add(winner)
    if len(winners) != 1:
        raise ValueError(f'Ambiguous visible result: {winners}')
    return high, low, next(iter(winners))


def answer_from_visible_prefix(prefix, kind):
    if kind in ('winner', 'margin'):
        high, low, winner = _visible_score(prefix)
        return winner if kind == 'winner' else str(high-low)
    if kind == 'top_player':
        rows = [(name, int(points))
                for pattern in (PLAYER_FACT, PAREN_FACT, SCORING_FACT)
                for name, points in pattern.findall(prefix)]
        if len(rows) != 5 or len({name for name, _ in rows}) != 5:
            raise ValueError('Expected five visible player-point facts')
        return max(rows, key=lambda item: item[1])[0]
    raise ValueError(kind)


def edit(plan, field, value):
    if field == 'swap_teams':
        return replace(plan, home=plan.away, away=plan.home)
    if field == 'high_score':
        return replace(plan, high_score=value)
    if field == 'points':
        return replace(plan, points=tuple(value))
    raise ValueError(field)
