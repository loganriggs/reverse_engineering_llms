"""Variable sports fact/answer plan with grounded pregame record context.

This is a controlled DGP arm, not a natural-language realism claim. It uses
corpus-derived result and player-stat forms, then samples which evidence and
which scored rules to include. Record priors are sampled from box-score W-L
values and treated as synthetic *pregame* state; outcome updates are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random
import re

from audit_rotowire_document_source import game_key
from sports_document_dgp import (
    Document, Site, SCORE_PAIR,
    answer_from_visible_prefix as base_answer,
    edit as base_edit, game_partition, realize as base_realize,
    resources, sample_plan as base_sample_plan,
)


SENTENCE = re.compile(r'[^.]+\.', re.S)
RECORD = re.compile(r'(?:The ([^.]+?) entered with a (\d+)-(\d+) record|'
                    r'Before tipoff, ([^.]+?) had a (\d+)-(\d+) record)\.')
POST_RECORD = re.compile(r'(?:After the result, ([^.]+?) stood at '
                         r'(\d+)-(\d+)|The game left ([^.]+?) with a '
                         r'(\d+)-(\d+) record)\.')
RULE_RATES = {
    'rotowire': {'winner': .42, 'margin': .30, 'top_player': .38},
    'sbnation': {'winner': .25, 'margin': .22, 'top_player': .28},
}


@dataclass(frozen=True)
class Plan:
    base: object
    home_before: tuple[int, int]
    away_before: tuple[int, int]

    @property
    def after(self):
        return {
            self.base.home: (self.home_before[0]+int(self.base.home_win),
                             self.home_before[1]+int(not self.base.home_win)),
            self.base.away: (self.away_before[0]+int(not self.base.home_win),
                             self.away_before[1]+int(self.base.home_win)),
        }


@lru_cache(maxsize=6)
def record_pool(source, group):
    out = []
    for row in resources()[0][source][0]:
        if group != 'all' and game_partition(game_key(row)) != group:
            continue
        for field in ('home_line', 'vis_line'):
            line = row[field]
            try:
                wins, losses = int(line['TEAM-WINS']), int(line['TEAM-LOSSES'])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= wins <= 80 and 0 <= losses <= 80 and 8 <= wins+losses <= 80:
                out.append((wins, losses))
    if not out:
        raise RuntimeError('No empirical record prior')
    return tuple(out)


def sample_plan(rng, **kwargs):
    base = base_sample_plan(rng, **kwargs)
    pool = record_pool(base.style_source, kwargs.get('source_group', 'all'))
    return Plan(base, rng.choice(pool), rng.choice(pool))


def edit(plan, field, value):
    return Plan(base_edit(plan.base, field, value),
                plan.home_before, plan.away_before)


def _base_sentences(doc):
    rows = []
    for match in SENTENCE.finditer(doc.text):
        raw = match.group()
        trim = len(raw)-len(raw.lstrip())
        sentence = raw.strip()
        own = [site for site in doc.sites
               if match.start() <= site.start < match.end()]
        if len(own) > 1:
            raise ValueError('Multiple sites in one sentence')
        site = own[0] if own else None
        offset = site.start-match.start()-trim if site else None
        kind = ('answer' if site else 'score' if SCORE_PAIR.search(sentence)
                else 'player')
        rows.append((kind, sentence, site, offset))
    score_count, player_count, answer_count = CounterKinds(rows)
    if score_count != 1 or not 3 <= player_count <= 5 or \
            answer_count != 3:
        raise ValueError('Unexpected base sentence inventory')
    return rows


def CounterKinds(rows):
    return tuple(sum(kind == wanted for kind, *_ in rows)
                 for wanted in ('score', 'player', 'answer'))


def _record_sentence(team, before, rng):
    w, l = before
    return (f'The {team} entered with a {w}-{l} record.'
            if rng.randrange(2) == 0 else
            f'Before tipoff, {team} had a {w}-{l} record.')


def _post_record_sentence(team, after, rng):
    w, l = after
    return (f'After the result, {team} stood at {w}-{l}.'
            if rng.randrange(2) == 0 else
            f'The game left {team} with a {w}-{l} record.')


def realize(plan, empirical_stat_rate=1., paragraph_rate=.12,
            rule_rate_scale=1., record_rate=.65,
            post_record_rate=.5, combined_record_rate=0.,
            pair_rate=0.):
    if not (0 <= empirical_stat_rate <= 1 and
            0 <= paragraph_rate <= 1 and
            0 <= rule_rate_scale <= 1 and 0 <= record_rate <= 1 and
            0 <= post_record_rate <= 1 and
            0 <= combined_record_rate <= 1 and
            0 <= pair_rate <= 1):
        raise ValueError('Rates must be within [0,1]')
    base = base_realize(plan.base, pair_rate=pair_rate, stat_richness=1.,
                        paragraph_rate=0., discourse_entropy=0.,
                        lead_stat_rate=0.,
                        empirical_stat_rate=empirical_stat_rate)
    rows = _base_sentences(base)
    score = next(row for row in rows if row[0] == 'score')
    stats = [row for row in rows if row[0] == 'player']
    answers = {row[2].kind: row for row in rows if row[0] == 'answer'}
    rng = random.Random(plan.base.surface_seed ^ 0x51C0A17)
    combined_rng = random.Random(plan.base.surface_seed ^ 0xC0B1E)
    if (plan.base.style_source == 'rotowire' and
            combined_rng.random() < combined_record_rate):
        from sports_combined_result_record_renderer import render
        score = ('score', render(plan, combined_rng)[0], None, None)
    rates = RULE_RATES[plan.base.style_source]
    chosen = {kind for kind, rate in rates.items()
              if rng.random() < rate*rule_rate_scale}
    if not chosen:
        chosen.add(rng.choice(tuple(rates)))
    selected_stats = stats if 'top_player' in chosen else rng.sample(
        stats, rng.randint(2, min(4, len(stats))))
    rng.shuffle(selected_stats)
    records = []
    post_records = []
    for team, before in ((plan.base.home, plan.home_before),
                         (plan.base.away, plan.away_before)):
        if rng.random() < record_rate:
            records.append(('record', _record_sentence(team, before, rng),
                            None, None))
            if rng.random() < post_record_rate:
                post_records.append((
                    'post_record',
                    _post_record_sentence(team, plan.after[team], rng),
                    None, None))
    # The result lead probability is approximately the human primary-result
    # frequency by style. Other acts may precede it, and site selection is
    # independent of wording; this is a sampled plan, not a fixed permutation.
    lead_result = rng.random() < (.75 if plan.base.style_source == 'rotowire'
                                  else .15)
    other = selected_stats+records
    rng.shuffle(other)
    score_slot = 0 if lead_result else rng.randrange(1, min(5, len(other)+1))
    skeleton = other[:score_slot]+[score]+other[score_slot:]
    for row in post_records:
        team = next(team for team in (plan.base.home, plan.base.away)
                    if team in row[1])
        pre_index = next(i for i, item in enumerate(skeleton)
                         if item[0] == 'record' and team in item[1])
        earliest = max(skeleton.index(score), pre_index)+1
        skeleton.insert(rng.randrange(earliest, len(skeleton)+1), row)
    # Place each selected recap after the evidence it requires. This varies
    # both rule prevalence and placement, unlike appending all three at end.
    for kind in sorted(chosen):
        if kind == 'top_player':
            earliest = max(i for i, row in enumerate(skeleton)
                           if row[0] == 'player')+1
        else:
            earliest = skeleton.index(score)+1
        slot = rng.randint(earliest, len(skeleton))
        skeleton.insert(slot, answers[kind])
    pieces, sites, length = [], [], 0
    for kind, sentence, site, offset in skeleton:
        if pieces:
            separator = '\n\n' if rng.random() < paragraph_rate else ' '
            pieces.append(separator)
            length += len(separator)
        if site:
            sites.append(Site(site.kind, length+offset,
                              length+offset+len(site.answer),
                              site.answer, site.candidates))
        pieces.append(sentence)
        length += len(sentence)
    return Document(''.join(pieces), tuple(sorted(sites,
                    key=lambda site: site.start)), plan)


def record_facts_from_visible(text):
    """Parse the two unambiguous W-L form families from visible text."""
    found = {'before': {}, 'after': {}}
    for name, pattern in (('before', RECORD), ('after', POST_RECORD)):
        for match in pattern.finditer(text):
            team, w, l = match.group(1, 2, 3) if match.group(1) else \
                match.group(4, 5, 6)
            if team in found[name]:
                raise ValueError('Duplicate record for team and phase')
            found[name][team] = (int(w), int(l))
    from sports_combined_result_record_renderer import visible_records
    for sentence in SENTENCE.findall(text):
        if not SCORE_PAIR.search(sentence):
            continue
        teams = [team for team in resources()[1]
                 if re.search(r'(?<!\w)'+re.escape(team)+r'(?!\w)',
                              sentence)]
        if len(teams) != 2:
            continue
        for team, record in visible_records(sentence, teams).items():
            if team in found['after'] and found['after'][team] != record:
                raise ValueError('Conflicting postgame records')
            found['after'][team] = record
    return found


def answer_from_visible_prefix(prefix, kind):
    if kind in ('winner', 'margin'):
        sentences = [match.group().strip() for match in SENTENCE.finditer(prefix)
                     if SCORE_PAIR.search(match.group()) and
                     not RECORD.fullmatch(match.group().strip()) and
                     not POST_RECORD.fullmatch(match.group().strip())]
        if len(sentences) != 1:
            raise ValueError('Expected one visible result sentence')
        if re.search(r'\(\d+-\d+\)', sentences[0]):
            from sports_combined_result_record_renderer import parse_visible_result
            high, low, winner = parse_visible_result(sentences[0])
            return winner if kind == 'winner' else str(high-low)
        return base_answer(sentences[0], kind)
    if kind == 'top_player':
        return base_answer(prefix, kind)
    raise ValueError(kind)
