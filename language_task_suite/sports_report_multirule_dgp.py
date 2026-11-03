"""Natural-report carrier with three controlled, jointly sampled rule sites.

The empirical carrier has no exact LM oracle. The inserted score and player
statements and three answer sites do have visible, independently testable
rules. Corpus-derived tally clauses are drawn only from the train game group.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import random
import re

from audit_boxscore_relation_inventory import FACT, _field
from audit_player_stat_grammar_bank import accepted as tally_accepted
from audit_rotowire_document_source import game_key
from corpus_score_sentence_renderer import _sentences, render
from sports_document_dgp import PLAYER, _detokenize, resources
from sports_report_carrier_dgp import sample_plan as sample_carrier_plan, visible_answer


QUERY_FORMS = {
    'winning_score': (
        'The winning side scored {ANSWER} points.',
        'The higher final total was {ANSWER} points.',
        'The winner finished on {ANSWER} points.',
    ),
    'margin': (
        'The final winning margin was {ANSWER} points.',
        'The winner finished {ANSWER} points ahead.',
        'The final gap between the teams was {ANSWER} points.',
    ),
    'player_points': (
        '{PLAYER} finished with {ANSWER} points.',
        '{PLAYER} recorded {ANSWER} points in the game.',
        'The point total for {PLAYER} was {ANSWER}.',
    ),
}


@dataclass(frozen=True)
class Plan:
    carrier_plan: object
    player_key: str
    player: str
    points: int
    stat_template: str
    stat_fields: tuple[str, ...]
    query_forms: tuple[tuple[str, str], ...]


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
def _rows():
    return {game_key(row): row for row in resources()[0]['rotowire'][0]}


@lru_cache(maxsize=1)
def _stat_templates():
    """Only approved full tally clauses from train-group games are used."""
    from sports_document_dgp import game_partition
    forms = {}
    for row in resources()[0]['rotowire'][0]:
        if game_partition(game_key(row)) != 'train':
            continue
        for sentence in _sentences(row['summary']):
            result = tally_accepted(sentence, row)
            if result is None or 'PTS' not in result['fields'] or \
                    result['text'].split(' ', 2)[2].lower().startswith('also '):
                continue
            text = result['text']
            name = text[:text.index(' ', text.index(' ')+1)]
            # The original number style is intentionally not preserved:
            # every resampled value is rendered as a digit, then rechecked.
            body = text[len(name):]
            matches = list(FACT.finditer(body))
            fields = []
            for match in reversed(matches):
                field = _field(match['unit'])
                if field is None:
                    break
                fields.append(field)
                body = body[:match.start()]+'{'+field+'} '+match['unit']+body[match.end():]
            else:
                fields = tuple(reversed(fields))
                template = '{PLAYER}'+body
                forms.setdefault((template, fields), None)
    result = tuple(forms)
    if len(result) < 20:
        raise RuntimeError('Too few grounded tally forms')
    return result


def template_digest():
    return hashlib.sha256(repr(_stat_templates()).encode()).hexdigest()


def _player_options(carrier_plan):
    row = _rows()[carrier_plan.source_game_key]
    box = row['box_score']
    carrier = _detokenize(carrier_plan.carrier)
    options = []
    for key, name in box['PLAYER_NAME'].items():
        points = box['PTS'].get(key)
        if not PLAYER.fullmatch(name) or not str(points).isdigit() or \
                not 4 <= int(points) <= 49 or \
                re.search(r'\b'+re.escape(name.split()[-1])+r'\b', carrier, re.I):
            continue
        city = box['TEAM_CITY'].get(key)
        peers = [int(value) for other, value in box['PTS'].items()
                 if box['TEAM_CITY'].get(other) == city and str(value).isdigit()]
        # A +1 edit cannot change a team-high claim about another player.
        if not peers or int(points)+1 >= max(peers):
            continue
        options.append((key, name, int(points)))
    return options


def sample_plan(rng: random.Random, group='train', carrier_words=100):
    for _ in range(200):
        base = sample_carrier_plan(rng, source='rotowire', group=group,
                                   carrier_words=carrier_words,
                                   query_kind='winning_score')
        options = _player_options(base)
        if not options:
            continue
        key, name, points = rng.choice(options)
        box = _rows()[base.source_game_key]['box_score']
        compatible = [(form, fields) for form, fields in _stat_templates()
                      if all(str(box[field].get(key)).isdigit() for field in fields)]
        if not compatible:
            continue
        form, fields = rng.choice(compatible)
        kinds = list(QUERY_FORMS)
        rng.shuffle(kinds)
        queries = tuple((kind, rng.choice(QUERY_FORMS[kind])) for kind in kinds)
        return Plan(base, key, name, points, form, fields, queries)
    raise RuntimeError('No game with a grounded unmentioned player')


def _answer(plan, kind):
    if kind == 'winning_score':
        return str(plan.carrier_plan.high)
    if kind == 'margin':
        return str(plan.carrier_plan.high-plan.carrier_plan.low)
    if kind == 'player_points':
        return str(plan.points)
    raise ValueError(kind)


def realize(plan: Plan):
    base = plan.carrier_plan
    row = _rows()[base.source_game_key]
    box = row['box_score']
    from score_template_guard import safe_for
    if not safe_for(base.score_template, base.high, base.low):
        raise ValueError('Score template incompatible with edit')
    score = _detokenize(render(base.score_template, base.winner,
                               base.loser, base.high, base.low))
    carrier = _detokenize(base.carrier).replace('*NEWPARAGRAPH*', '\n\n')
    values = {field: int(box[field][plan.player_key]) for field in plan.stat_fields}
    values['PTS'] = plan.points
    stat = plan.stat_template.format(PLAYER=plan.player, **values)
    text = score+' '+carrier+' '+stat
    sites = []
    for kind, form in plan.query_forms:
        prefix, suffix = form.split('{ANSWER}')
        prefix = prefix.replace('{PLAYER}', plan.player)
        suffix = suffix.replace('{PLAYER}', plan.player)
        answer = _answer(plan, kind)
        text += ' '+prefix
        start = len(text)
        text += answer+suffix
        candidates = tuple(map(str, range(60, 161))) if kind == 'winning_score' \
            else tuple(map(str, range(1, 61))) if kind == 'margin' \
            else tuple(map(str, range(0, 61)))
        sites.append(Site(kind, start, start+len(answer), answer, candidates))
    return Document(text, tuple(sites), plan)


def increment_score(plan):
    return replace(plan, carrier_plan=replace(plan.carrier_plan,
                                               high=plan.carrier_plan.high+1))


def increment_player(plan):
    return replace(plan, points=plan.points+1)


def visible_site_answer(doc, site):
    """Recompute from visible source text, without reading hidden plan values."""
    prefix = doc.text[:site.start]
    if site.kind in ('winning_score', 'margin'):
        return visible_answer(prefix, site.kind)
    tail = prefix[prefix.rfind('.')+1:]
    question = re.fullmatch(
        r'\s*(?:([A-Z][a-z]+ [A-Z][a-z]+) '
        r'(?:finished with|recorded)|'
        r'The point total for ([A-Z][a-z]+ [A-Z][a-z]+) was)\s+',
        tail)
    if question is None:
        raise ValueError('Cannot parse visible player question')
    name = question[1] or question[2]
    matches = list(re.finditer(r'(?<!\w)'+re.escape(name)+r'(?=\s)', prefix))
    source = []
    for match in matches:
        end = prefix.find('.', match.end())
        if end >= 0:
            source.append(prefix[match.start():end+1])
    if not source:
        raise ValueError('Expected a completed visible player source')
    # The first mention is the inserted source; completed answer sentences
    # can mention the same name later in the document.
    facts = [(int(m['number']), _field(m['unit'])) for m in FACT.finditer(source[0])
             if m['number'].isdigit()]
    points = [n for n, field in facts if field == 'PTS']
    if len(points) != 1:
        raise ValueError('Expected one visible player-point fact')
    return str(points[0])
