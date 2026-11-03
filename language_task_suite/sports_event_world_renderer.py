"""First text pilot over event-grounded game worlds.

Two visible score snapshots precede each run claim, making its two numeric
targets computable from the document prefix. This is a controlled pilot, not
an accepted natural-language distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import re

from corpus_score_sentence_renderer import render as render_score
from score_template_guard import safe_for
from sports_document_dgp import Site, _detokenize, _parse_score_sentence, resources
from sports_event_world_dgp import EventWorld


ORDINAL = ('', 'first', 'second', 'third', 'fourth')
SNAPSHOT1 = re.compile(r'At one point in the '
                       r'(first|second|third|fourth) quarter, the score '
                       r'was (.+?) (\d+), (.+?) (\d+)\.$')
SNAPSHOT2 = re.compile(r'Later that quarter, the score was '
                       r'(.+?) (\d+), (.+?) (\d+)\.$')
UNSUPPORTED_TIME_WORDS = frozenset({
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday',
    'sunday', 'christmas', 'thanksgiving', 'yesterday', 'tonight',
    'night', 'afternoon', 'morning',
})
FALLBACK_RESULT = re.compile(r'(.+?) beat (.+?) (\d+)-(\d+)\.$')


@dataclass(frozen=True)
class RenderedDocument:
    text: str
    sites: tuple[Site, ...]
    world: EventWorld
    sentences: tuple[str, ...]


def parse_result_sentence(sentence):
    match = FALLBACK_RESULT.fullmatch(sentence)
    if match:
        winner, loser, high, low = match.groups()
        if winner in resources()[1] and loser in resources()[1] and \
                int(high) > int(low):
            return int(high), int(low), winner
    return _parse_score_sentence(sentence)


def score_sentence(world, rng):
    home_score, away_score = world.final_score
    home_win = home_score > away_score
    winner = world.home if home_win else world.away
    loser = world.away if home_win else world.home
    high, low = sorted(world.final_score, reverse=True)
    templates = [template for template in resources()[0][world.source][1]
                 if template.home_win == home_win and
                 safe_for(template, high, low) and
                 not any(token.lower() in UNSUPPORTED_TIME_WORDS
                         for token in template.tokens)]
    sentence = (_detokenize(render_score(rng.choice(templates), winner,
                                         loser, high, low)) if templates
                else f'{winner} beat {loser} {high}-{low}.')
    assert parse_result_sentence(sentence) == (high, low, winner)
    return sentence


def player_sentence(world, rng):
    return rng.choice((
        f'{world.top_player} scored {world.top_player_points} points.',
        f'{world.top_player} finished with {world.top_player_points} points.',
        f'{world.top_player} posted {world.top_player_points} points.',
    ))


def halftime_sentence(world, rng):
    home, away = world.period_end_scores[1]
    if home == away:
        return f'The teams were tied {home}-{away} at halftime.'
    leader = world.home if home > away else world.away
    trailer = world.away if home > away else world.home
    high, low = sorted((home, away), reverse=True)
    return rng.choice((
        f'{leader} led {trailer} {high}-{low} at halftime.',
        f'At halftime, {leader} held a {high}-{low} lead over {trailer}.',
    ))


def period_sentence(world, period, rng):
    if period not in (1, 3):
        raise ValueError('Only first and third quarter forms implemented')
    home, away = world.period_end_scores[period-1]
    label = ORDINAL[period]
    if home == away:
        return f'The {label} quarter ended tied {home}-{away}.'
    leader = world.home if home > away else world.away
    trailer = world.away if home > away else world.home
    high, low = sorted((home, away), reverse=True)
    return rng.choice((
        f'{leader} led {trailer} {high}-{low} after the {label} quarter.',
        f'After the {label} quarter, {leader} held a {high}-{low} lead.',
    ))


def run_sentences(world, rng):
    run = world.interval
    if run.period > 4:
        return None
    first = (f'At one point in the {ORDINAL[run.period]} quarter, the '
             f'score was {world.home} {run.start_score[0]}, '
             f'{world.away} {run.start_score[1]}.')
    second = (f'Later that quarter, the score was {world.home} '
              f'{run.end_score[0]}, {world.away} {run.end_score[1]}.')
    team = world.home if run.side == 'home' else world.away
    third = rng.choice((
        f'{team} went on a {run.points_for}-{run.points_against} '
        'run over that stretch.',
        f'Over that stretch, {team} went on a '
        f'{run.points_for}-{run.points_against} run.',
        f'The {team} put together a {run.points_for}-{run.points_against} '
        'run during that stretch.',
    ))
    return first, second, third


def run_values_from_visible_prefix(prefix):
    """Compute run scores from visible sentences and the named run team."""
    sentences = re.findall(r'[^.!?]*[.!?]', prefix)
    if len(sentences) < 2:
        raise ValueError('Missing visible score snapshots')
    first = SNAPSHOT1.fullmatch(sentences[-2].strip())
    second = SNAPSHOT2.fullmatch(sentences[-1].strip())
    if first is None or second is None:
        raise ValueError('Missing expected score snapshot pair')
    _, home, start_home, away, start_away = first.groups()
    home2, end_home, away2, end_away = second.groups()
    if (home, away) != (home2, away2):
        raise ValueError('Snapshot teams differ')
    # Find the incomplete sentence by its position after the last terminal.
    terminal = max(prefix.rfind('.'), prefix.rfind('!'), prefix.rfind('?'))
    partial = prefix[terminal+1:].lstrip()
    partial = re.sub(r'\d+-$', '', partial)
    team_match = re.search(r'(?:Over that stretch, )?(?:The )?(.+?) '
                           r'(?:went on|put together) a\s*$', partial)
    if team_match is None:
        raise ValueError('Run team absent before target')
    team = team_match.group(1)
    delta = (int(end_home)-int(start_home),
             int(end_away)-int(start_away))
    if min(delta) < 0:
        raise ValueError('Negative visible score change')
    if team == home:
        return delta
    if team == away:
        return delta[::-1]
    raise ValueError('Run team not in snapshots')


def render_world(world: EventWorld, rng: random.Random,
                 run_rate=None, halftime_rate=.55, period_rate=.55,
                 paragraph_rate=.12):
    if run_rate is None:
        run_rate = .04 if world.source == 'rotowire' else .22
    if not all(0 <= value <= 1 for value in
               (run_rate, halftime_rate, period_rate, paragraph_rate)):
        raise ValueError('Rates must lie in [0,1]')
    result = score_sentence(world, rng)
    player = player_sentence(world, rng)
    other = [player]
    if rng.random() < halftime_rate:
        other.append(halftime_sentence(world, rng))
    for period in (1, 3):
        if rng.random() < period_rate:
            other.append(period_sentence(world, period, rng))
    if rng.random() < run_rate:
        block = run_sentences(world, rng)
        if block:
            other.extend(block)
    # Only the two score snapshots and their run statement are kept together.
    # Elsewhere the act order is sampled rather than fixed.
    if len(other) >= 4 and SNAPSHOT1.fullmatch(other[-3]):
        run_block = tuple(other[-3:])
        other = other[:-3]
    else:
        run_block = None
    rng.shuffle(other)
    units = other + ([run_block] if run_block else [])
    rng.shuffle(units)
    lead_probability = .75 if world.source == 'rotowire' else .15
    score_slot = 0 if rng.random() < lead_probability else \
        rng.randrange(len(units)+1)
    units.insert(score_slot, result)
    sentences = tuple(sentence for unit in units for sentence in
                      (unit if isinstance(unit, tuple) else (unit,)))
    pieces = []
    for sentence in sentences:
        if pieces:
            pieces.append('\n\n' if rng.random() < paragraph_rate else ' ')
        pieces.append(sentence)
    text = ''.join(pieces)
    sites = []
    if run_block:
        final = run_block[-1]
        start = text.index(final)
        pair = re.search(r'\b(\d+)-(\d+) run\b', final)
        assert pair
        for kind, match in (('run_for', 1), ('run_against', 2)):
            a, b = start+pair.start(match), start+pair.end(match)
            sites.append(Site(kind, a, b, text[a:b],
                              tuple(map(str, range(0, 41)))))
    return RenderedDocument(text, tuple(sites), world, sentences)
