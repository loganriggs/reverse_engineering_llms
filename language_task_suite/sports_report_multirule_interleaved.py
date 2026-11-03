"""Place unchanged controlled facts and queries among carrier sentences.

This is a paired placement experiment against sports_report_multirule_dgp:
same sampled game, human carrier, factual sentence, question forms and answers.
Only sentence order changes. It is not a new claim of natural discourse.
"""

from __future__ import annotations

import hashlib
import random

from corpus_score_sentence_renderer import _sentences, render
from score_template_guard import safe_for
from sports_document_dgp import _detokenize
from sports_report_multirule_dgp import (
    Document, Site, _answer, _rows,
    realize as fixed_realize,
)


def _stable_seed(plan):
    identity = (plan.carrier_plan.source_game_key, plan.player,
                plan.stat_template, plan.query_forms,
                plan.carrier_plan.carrier)
    return int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:8],
                          'big')


def placement(plan):
    carrier_sentences = list(_sentences(plan.carrier_plan.carrier))
    if not carrier_sentences:
        raise ValueError('Need at least one carrier sentence')
    n = len(carrier_sentences)
    rng = random.Random(_stable_seed(plan))
    stat_slot = rng.randint(0, max(0, n-2))
    question_slots = {}
    for kind, _ in plan.query_forms:
        lower = stat_slot+1 if kind == 'player_points' else 1
        question_slots[kind] = rng.randint(lower, n)
    return stat_slot, question_slots


def realize(plan):
    base = plan.carrier_plan
    if not safe_for(base.score_template, base.high, base.low):
        raise ValueError('Score template incompatible with plan')
    score = _detokenize(render(base.score_template, base.winner,
                               base.loser, base.high, base.low))
    row = _rows()[base.source_game_key]
    box = row['box_score']
    values = {field: int(box[field][plan.player_key]) for field in plan.stat_fields}
    values['PTS'] = plan.points
    stat = plan.stat_template.format(PLAYER=plan.player, **values)
    carrier = [_detokenize(sentence).replace('*NEWPARAGRAPH*', '\n\n')
               for sentence in _sentences(base.carrier)]
    stat_slot, question_slots = placement(plan)
    rng = random.Random(_stable_seed(plan) ^ 0x517E)
    scheduled = {slot: [] for slot in range(len(carrier)+1)}
    for kind, form in plan.query_forms:
        scheduled[question_slots[kind]].append((kind, form))
    for slot in scheduled:
        rng.shuffle(scheduled[slot])

    items = [('score', score)]
    for slot in range(len(carrier)+1):
        if slot == stat_slot:
            items.append(('stat', stat))
        items.extend(scheduled[slot])
        if slot < len(carrier):
            items.append(('carrier', carrier[slot]))
    # The player question must follow a completed source sentence.
    kinds = [item[0] for item in items]
    assert kinds.index('stat') < kinds.index('player_points')
    text = ''
    sites = []
    for kind, content in items:
        if text:
            text += ' '
        if kind in ('score', 'stat', 'carrier'):
            text += content
            continue
        prefix, suffix = content.split('{ANSWER}')
        prefix = prefix.replace('{PLAYER}', plan.player)
        suffix = suffix.replace('{PLAYER}', plan.player)
        answer = _answer(plan, kind)
        text += prefix
        start = len(text)
        text += answer+suffix
        candidates = tuple(map(str, range(60, 161))) if kind == 'winning_score' \
            else tuple(map(str, range(1, 61))) if kind == 'margin' \
            else tuple(map(str, range(0, 61)))
        sites.append(Site(kind, start, start+len(answer), answer, candidates))
    return Document(text, tuple(sites), plan)


def realize_mixed(plan):
    """Stable 50/50 placement assignment, unchanged under supported edits."""
    return realize(plan) if _stable_seed(plan) & 1 else fixed_realize(plan)
