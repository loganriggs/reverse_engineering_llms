"""Versioned sports DGP arm with a four-quarter trajectory and five rules.

This is a semantic/causal fixture, not an accepted natural-language surface.
It reuses the V3 score/player document, inserts a visible quarter table, and
adds halftime-leader and comeback answers.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
import random
import re

from audit_rotowire_document_source import game_key
from corpus_score_sentence_renderer import ScoreTemplate, render
from score_template_guard import safe_for
from sports_document_dgp import (
    Document, SCORE_PAIR, Site, _detokenize,
    answer_from_visible_prefix as base_answer,
    edit as base_edit, game_partition, realize as base_realize,
    resources, sample_plan as base_sample_plan,
)


QUARTER = re.compile(r'Q([1-4]): (.+?) (\d+), (.+?) (\d+)\.')
HALF_FORMS = (
    'At halftime, HALFTIME_ANSWER was ahead.',
    'The halftime lead belonged to HALFTIME_ANSWER.',
    'Through two quarters, HALFTIME_ANSWER held the lead.',
    'The team leading after the second quarter was HALFTIME_ANSWER.',
)
COMEBACK_FORMS = (
    'The eventual winner trailed at halftime: COMEBACK_ANSWER.',
    'Did the winner come from behind after halftime? COMEBACK_ANSWER.',
    'A halftime deficit for the winner: COMEBACK_ANSWER.',
    'The winner was behind after two quarters: COMEBACK_ANSWER.',
)
WEEKDAYS = frozenset(('Monday', 'Tuesday', 'Wednesday', 'Thursday',
                      'Friday', 'Saturday', 'Sunday'))


@dataclass(frozen=True)
class Plan:
    base: object
    home_quarters: tuple[int, ...]
    away_quarters: tuple[int, ...]
    donor_game_key: tuple

    @property
    def halftime_leader(self):
        h, a = sum(self.home_quarters[:2]), sum(self.away_quarters[:2])
        if h == a:
            raise ValueError('Halftime tie excluded from this rule arm')
        return self.base.home if h > a else self.base.away

    @property
    def comeback(self):
        return 'yes' if self.halftime_leader != self.base.winner else 'no'


@lru_cache(maxsize=6)
def donors(source, group):
    rows = resources()[0][source][0]
    output = []
    for row in rows:
        if group != 'all' and game_partition(game_key(row)) != group:
            continue
        try:
            home = tuple(int(row['home_line'][f'TEAM-PTS_QTR{i}'])
                         for i in range(1, 5))
            away = tuple(int(row['vis_line'][f'TEAM-PTS_QTR{i}'])
                         for i in range(1, 5))
            hs = int(row['home_line']['TEAM-PTS'])
            vs = int(row['vis_line']['TEAM-PTS'])
        except (KeyError, TypeError, ValueError):
            continue
        high, low = max(hs, vs), min(hs, vs)
        if sum(home) != hs or sum(away) != vs or min(home+away) < 0 \
                or not (100 <= high <= 130 and 1 <= high-low <= 25):
            continue
        winning, losing = (home, away) if hs > vs else (away, home)
        half_margin = sum(winning[:2])-sum(losing[:2])
        if half_margin == 0 or abs(half_margin) > 8:
            continue
        # Permit a conservative same-final-score edit that reverses
        # halftime leadership without implausible negative/huge quarters.
        leader = winning if half_margin > 0 else losing
        delta = abs(half_margin)+1
        if leader[1] <= delta or leader[2]+delta > 45:
            continue
        output.append((high, low, winning, losing, game_key(row)))
    return tuple(output)


def sample_plan(rng: random.Random, source_group='train',
                sbnation_rate=.25, **base_controls):
    base = base_sample_plan(rng, source_group=source_group,
                            sbnation_rate=sbnation_rate, **base_controls)
    candidates = donors(base.style_source, source_group)
    if not candidates:
        raise RuntimeError('No quarter donors for style/group')
    templates = resources()[0][base.style_source][1]
    expected_day = datetime.strptime(
        base.source_game_key[0], '%m_%d_%y').strftime('%A')
    for _ in range(100):
        high, low, winning, losing, donor_key = rng.choice(candidates)
        options = [
            ScoreTemplate(tuple(expected_day if token in WEEKDAYS else token
                                for token in t.tokens), t.home_win)
            for t in templates if t.home_win == base.home_win
            and safe_for(t, high, low) and safe_for(t, high+1, low)]
        if options:
            changed = replace(base, high_score=high, low_score=low,
                              score_template=rng.choice(options))
            home, away = (winning, losing) if base.home_win \
                else (losing, winning)
            return Plan(changed, home, away, donor_key)
    raise RuntimeError('Could not choose compatible quarter score template')


def _table(plan):
    return ' '.join(
        f'Q{i}: {plan.base.home} {h}, {plan.base.away} {a}.'
        for i, (h, a) in enumerate(zip(plan.home_quarters,
                                       plan.away_quarters), 1))


def realize(plan: Plan, **surface_controls):
    if len(plan.home_quarters) != 4 or len(plan.away_quarters) != 4 \
            or sum(plan.home_quarters) != (
                plan.base.high_score if plan.base.home_win
                else plan.base.low_score) \
            or sum(plan.away_quarters) != (
                plan.base.low_score if plan.base.home_win
                else plan.base.high_score):
        raise ValueError('Quarter scores must reconstruct final score')
    base = base_realize(plan.base, **surface_controls)
    # Put the trajectory between evidence and the first scored recap.
    first_site = min(site.start for site in base.sites)
    insert_at = base.text.rfind('.', 0, first_site)+1
    insert = ' '+_table(plan)
    text = base.text[:insert_at]+insert+base.text[insert_at:]
    sites = [replace(site, start=site.start+len(insert),
                     end=site.end+len(insert))
             if site.start >= insert_at else site for site in base.sites]
    rng = random.Random(plan.base.surface_seed ^ 0x4A19D1)
    for kind, form, answer, candidates in (
            ('halftime_leader', rng.choice(HALF_FORMS),
             plan.halftime_leader, (plan.base.home, plan.base.away)),
            ('comeback', rng.choice(COMEBACK_FORMS),
             plan.comeback, ('yes', 'no'))):
        marker = 'HALFTIME_ANSWER' if kind == 'halftime_leader' \
            else 'COMEBACK_ANSWER'
        pre, post = form.split(marker)
        text += ' '+pre
        start = len(text)
        text += answer
        sites.append(Site(kind, start, len(text), answer, candidates))
        text += post
    # Legacy rich-stat forms always pluralize these units. Fix only this
    # versioned arm and update every later scored-site offset.
    for match in reversed(list(re.finditer(r'\b1 (assists|rebounds)\b', text))):
        replacement = '1 '+match.group(1)[:-1]
        text = text[:match.start()]+replacement+text[match.end():]
        sites = [replace(site, start=site.start-1, end=site.end-1)
                 if site.start >= match.end() else site for site in sites]
    return Document(text, tuple(sorted(sites, key=lambda s: s.start)), plan)


def _visible_quarters(prefix):
    rows = QUARTER.findall(prefix)
    if len(rows) != 4 or [int(row[0]) for row in rows] != [1, 2, 3, 4]:
        raise ValueError('Expected four ordered visible quarter rows')
    home, away = rows[0][1], rows[0][3]
    if any(row[1] != home or row[3] != away for row in rows):
        raise ValueError('Quarter team names disagree')
    h = [int(row[2]) for row in rows]
    a = [int(row[4]) for row in rows]
    if sum(h[:2]) == sum(a[:2]):
        raise ValueError('Halftime tie')
    return home if sum(h[:2]) > sum(a[:2]) else away


def _visible_score(prefix):
    pair = SCORE_PAIR.search(prefix)
    if pair is None:
        raise ValueError('No final score pair in visible text')
    high, low = map(int, pair.groups())
    start = prefix.rfind('.', 0, pair.start())+1
    end = prefix.find('.', pair.end())
    if end < 0:
        raise ValueError('Incomplete final-score sentence')
    sentence = prefix[start:end+1].strip()
    def normalize_day(text):
        return re.sub(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|'
                      r'Saturday|Sunday)\b', '<DAY>', text)
    wanted = normalize_day(sentence)
    corpora, teams, _ = resources()
    mentioned = [team for team in teams
                 if re.search(r'(?<!\w)'+re.escape(team)+r'(?!\w)',
                              sentence)]
    if len(mentioned) != 2:
        raise ValueError('Expected two visible teams')
    winners = set()
    for _, templates in corpora.values():
        for template in templates:
            if not safe_for(template, high, low):
                continue
            for winner, loser in (mentioned, mentioned[::-1]):
                rendered = _detokenize(render(template, winner, loser,
                                              high, low))
                if normalize_day(rendered) == wanted:
                    winners.add(winner)
    if len(winners) != 1:
        raise ValueError('Ambiguous visible winner')
    return high, low, next(iter(winners))


def answer_from_visible_prefix(prefix, kind):
    if kind in ('winner', 'margin'):
        high, low, winner = _visible_score(prefix)
        return winner if kind == 'winner' else str(high-low)
    if kind == 'top_player':
        return base_answer(prefix, kind)
    leader = _visible_quarters(prefix)
    if kind == 'halftime_leader':
        return leader
    if kind == 'comeback':
        return 'yes' if leader != _visible_score(prefix)[2] else 'no'
    raise ValueError(kind)


def edit(plan: Plan, field, value=None):
    if field in ('swap_teams', 'points'):
        return replace(plan, base=base_edit(plan.base, field, value))
    if field == 'high_score':
        if value != plan.base.high_score+1:
            raise ValueError('Only +1 winning score edit supported')
        # Modify the winning side's fourth quarter, leaving halftime intact.
        if plan.base.home_win:
            home = plan.home_quarters[:-1]+(plan.home_quarters[-1]+1,)
            return replace(plan, base=base_edit(plan.base, field, value),
                           home_quarters=home)
        away = plan.away_quarters[:-1]+(plan.away_quarters[-1]+1,)
        return replace(plan, base=base_edit(plan.base, field, value),
                       away_quarters=away)
    if field == 'flip_halftime':
        half = sum(plan.home_quarters[:2])-sum(plan.away_quarters[:2])
        if half == 0:
            raise ValueError('Halftime tie')
        delta = abs(half)+1
        if half > 0:
            quarters = list(plan.home_quarters)
            quarters[1] -= delta
            quarters[2] += delta
            return replace(plan, home_quarters=tuple(quarters))
        quarters = list(plan.away_quarters)
        quarters[1] -= delta
        quarters[2] += delta
        return replace(plan, away_quarters=tuple(quarters))
    raise ValueError(field)
