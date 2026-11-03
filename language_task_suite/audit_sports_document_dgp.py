"""Semantic, edit and form audit for the sampled sports-document candidate."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key
from compare_v12_fineweb_fresh import first_words
from project_report_dgp import realize as report_realize
from project_report_dgp import sample_plan as report_plan
from sports_document_dgp import (
    answer_from_visible_prefix, edit, game_partition, realize, resources,
    sample_plan,
)


def _answers(doc):
    answers = {}
    previous = 0
    for site in doc.sites:
        assert previous <= site.start < site.end <= len(doc.text)
        assert doc.text[site.start:site.end] == site.answer
        assert site.answer in site.candidates
        observed = answer_from_visible_prefix(
            doc.text[:site.start], site.kind)
        assert observed == site.answer, (site.kind, observed, site.answer)
        answers[site.kind] = observed
        previous = site.end
    assert set(answers) == {'winner', 'margin', 'top_player'}
    return answers


def _window(doc, site):
    return ' '.join(doc.text[:site.start].split()[-25:]+
                    doc.text[site.start:].split()[:16])


def _rich_facts(doc):
    """Independently check all rendered numeric facts against sampled state."""
    counts = Counter()
    for i, name in enumerate(doc.plan.players):
        start = re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)', doc.text)
        assert start is not None, name
        later_names = [match.start() for other in doc.plan.players
                       if other != name
                       for match in [re.search(r'(?<!\w)'+re.escape(other)+
                                               r'(?!\w)',
                                               doc.text[start.end():])]
                       if match is not None]
        next_name = start.end()+min(later_names) if later_names else len(doc.text)
        next_period = doc.text.find('.', start.end())
        end = min(next_name, next_period if next_period >= 0 else len(doc.text))
        segment = doc.text[start.end():end]
        for metric, expected in (('points', doc.plan.points[i]),
                                 ('rebounds', doc.plan.rebounds[i]),
                                 ('assists', doc.plan.assists[i])):
            values = re.findall(r'\b(\d+) '+metric+r'\b', segment)
            if metric == 'points':
                assert len(values) == 1, (name, segment)
            assert all(int(value) == expected for value in values), \
                (name, metric, values, expected)
            counts[metric] += len(values)
    assert counts['points'] == 5
    assert counts['rebounds']+counts['assists'] > 0
    return counts


def semantic(seed, n, sbnation_rate, form_entropy, pair_rate,
             discourse_entropy=0., lead_stat_rate=0., game_group='all',
             stat_richness=0.):
    rng = random.Random(seed)
    counts = Counter()
    examples = []
    for i in range(n):
        plan = sample_plan(rng, sbnation_rate=sbnation_rate,
                           source_group=game_group)
        assert game_group == 'all' or \
            game_partition(plan.source_game_key) == game_group
        doc = realize(plan, form_entropy=form_entropy,
                      pair_rate=pair_rate,
                      discourse_entropy=discourse_entropy,
                      lead_stat_rate=lead_stat_rate,
                      stat_richness=stat_richness)
        original = _answers(doc)
        if stat_richness:
            counts.update(_rich_facts(doc))
        counts['plans'] += 1
        counts[f'style/{plan.style_source}'] += 1
        counts[f'recap_order/{"-".join(s.kind for s in doc.sites)}'] += 1
        if len(examples) < 2:
            examples.append(doc.text)
        changed = realize(edit(plan, 'swap_teams', True),
                          form_entropy=form_entropy,
                          pair_rate=pair_rate,
                          discourse_entropy=discourse_entropy,
                          lead_stat_rate=lead_stat_rate,
                          stat_richness=stat_richness)
        after = _answers(changed)
        assert after['winner'] != original['winner']
        assert after['margin'] == original['margin']
        assert after['top_player'] == original['top_player']
        counts['team_swap_selective'] += 1
        changed = realize(edit(plan, 'high_score',
                               plan.high_score+1),
                          form_entropy=form_entropy,
                          pair_rate=pair_rate,
                          discourse_entropy=discourse_entropy,
                          lead_stat_rate=lead_stat_rate,
                          stat_richness=stat_richness)
        after = _answers(changed)
        assert int(after['margin']) == int(original['margin'])+1
        assert after['winner'] == original['winner']
        assert after['top_player'] == original['top_player']
        counts['score_edit_selective'] += 1
        alt_index = next(j for j, points in enumerate(plan.points)
                         if points != max(plan.points))
        new_points = list(plan.points)
        new_points[alt_index] = max(plan.points)+1
        changed = realize(edit(plan, 'points', new_points),
                          form_entropy=form_entropy,
                          pair_rate=pair_rate,
                          discourse_entropy=discourse_entropy,
                          lead_stat_rate=lead_stat_rate,
                          stat_richness=stat_richness)
        after = _answers(changed)
        assert after['top_player'] == plan.players[alt_index]
        assert after['winner'] == original['winner']
        assert after['margin'] == original['margin']
        counts['player_edit_selective'] += 1
    return dict(counts=dict(counts), examples=examples)


def form(seed, n, words, sbnation_rate, form_entropy, pair_rate,
         discourse_entropy=0., lead_stat_rate=0., game_group='all',
         stat_richness=0.):
    rng = random.Random(seed)
    rows = []
    windows = {kind: [] for kind in ('winner', 'margin', 'top_player')}
    draws = 0
    while len(rows) < n:
        draws += 1
        if draws > 100*n:
            raise RuntimeError('Too few long sports documents')
        doc = realize(sample_plan(rng, sbnation_rate=sbnation_rate,
                                  source_group=game_group),
                      form_entropy=form_entropy, pair_rate=pair_rate,
                      discourse_entropy=discourse_entropy,
                      lead_stat_rate=lead_stat_rate,
                      stat_richness=stat_richness)
        crop = first_words(doc.text, words)
        if len(crop.split()) < words:
            continue
        rows.append(crop)
        for site in doc.sites:
            windows[site.kind].append(_window(doc, site))
    return dict(draws=draws, panel=structure(rows),
                rule_windows={kind: structure(texts)
                              for kind, texts in windows.items()})


def natural_sports(seed, n, words):
    corpora, _, _ = resources()
    result = {}
    for name, (rows, _) in corpora.items():
        by_game = {}
        for row in rows:
            if len(row['summary']) >= words:
                by_game.setdefault(game_key(row), row)
        choices = random.Random(seed).sample(list(by_game.values()), n)
        result[name] = structure([
            first_words(' '.join(row['summary']), words)
            for row in choices])
    return result


def report_reference(seed, n, words):
    rng = random.Random(seed)
    return structure([first_words(report_realize(report_plan(rng)).text,
                                  words) for _ in range(n)])


def audit(args):
    semantic_panels = {}
    form_panels = {}
    for style in (0., 1.):
        for entropy in (0., 1.):
            for pair in (0., 1.):
                key = (f'sbnation{style:g}_entropy{entropy:g}'
                       f'_pair{pair:g}')
                semantic_panels[key] = semantic(
                    args.seed+int(style)*100+int(entropy)*10+int(pair),
                    args.plans, style, entropy, pair,
                    args.discourse_entropy, args.lead_stat_rate,
                    args.game_group, args.stat_richness)
                form_panels[key] = form(
                    args.seed+10000+int(style)*100+int(entropy)*10+
                    int(pair), args.form_n, args.words, style,
                    entropy, pair, args.discourse_entropy,
                    args.lead_stat_rate, args.game_group,
                    args.stat_richness)
    return dict(config=dict(seed=args.seed,
                            plans_per_condition=args.plans,
                            form_n=args.form_n, form_words=args.words,
                            discourse_entropy=args.discourse_entropy,
                            lead_stat_rate=args.lead_stat_rate,
                            game_group=args.game_group,
                            stat_richness=args.stat_richness),
                semantic=semantic_panels, form=form_panels,
                natural_sports=natural_sports(args.seed+20000,
                                              args.form_n, args.words),
                linked_report=report_reference(args.seed+30000,
                                               args.form_n, args.words),
                caveat='Sports DGP is a first sampled multi-rule candidate. '
                       'The source archives supply guarded score sentences '
                       'and entity vocabularies; all later clauses remain '
                       'synthetic. Exact oracle is limited to marked sites. '
                       'Form comparators are independent same-size panels '
                       'but not matched for content or writing context.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=82041)
    p.add_argument('--plans', type=int, default=300)
    p.add_argument('--form-n', type=int, default=64)
    p.add_argument('--words', type=int, default=55)
    p.add_argument('--discourse-entropy', type=float, default=0.)
    p.add_argument('--lead-stat-rate', type=float, default=0.)
    p.add_argument('--stat-richness', type=float, default=0.)
    p.add_argument('--game-group', choices=('all', 'train', 'heldout'),
                   default='all')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    for name, row in result['form'].items():
        panel = row['panel']
        print(name, panel['word_types'],
              panel['repeated_across_docs_trigram_occurrence_share'])
    for name, panel in result['natural_sports'].items():
        print('natural', name, panel['word_types'],
              panel['repeated_across_docs_trigram_occurrence_share'])
    panel = result['linked_report']
    print('linked_report', panel['word_types'],
          panel['repeated_across_docs_trigram_occurrence_share'])


if __name__ == '__main__':
    main()
