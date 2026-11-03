"""Paired-world semantic and form audit for corpus-derived stat predicates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from audit_boxscore_relation_inventory import FACT, NUMBER_WORDS, _field
from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key
from compare_v12_fineweb_fresh import first_words
from sports_document_dgp import (answer_from_visible_prefix, edit, realize,
                                 game_partition, resources, sample_plan)
from sports_empirical_stat_renderer import bank


def answers(doc):
    result = {}
    for site in doc.sites:
        assert doc.text[site.start:site.end] == site.answer
        got = answer_from_visible_prefix(doc.text[:site.start], site.kind)
        assert got == site.answer, (site.kind, got, site.answer)
        result[site.kind] = got
    assert set(result) == {'winner', 'margin', 'top_player'}
    return result


def numeric_facts(doc):
    """Check every numeric stat in each first player clause against its plan."""
    counts = Counter()
    for index, name in enumerate(doc.plan.players):
        match = re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)', doc.text)
        assert match is not None
        stop = doc.text.find('.', match.end())
        assert stop >= 0
        clause = doc.text[match.end():stop]
        observed = {}
        for fact in FACT.finditer(clause):
            field = _field(fact['unit'])
            raw = fact['number'].lower()
            value = int(raw) if raw.isdigit() else NUMBER_WORDS[raw]
            assert field in ('PTS', 'REB', 'AST')
            assert field not in observed
            observed[field] = value
            expected = {'PTS': doc.plan.points[index],
                        'REB': doc.plan.rebounds[index],
                        'AST': doc.plan.assists[index]}[field]
            assert value == expected, (name, field, value, expected)
            counts['field/'+field] += 1
        assert 'PTS' in observed, (name, clause)
    assert counts['field/PTS'] == 5
    return counts


def panel(seed, n, source, group, words):
    rng = random.Random(seed)
    crops = {'baseline': [], 'empirical': []}
    counts = Counter()
    for _ in range(n):
        plan = sample_plan(rng, sbnation_rate=int(source == 'sbnation'),
                           source_group=group)
        docs = {
            'baseline': realize(plan, pair_rate=0., stat_richness=1.,
                                discourse_entropy=1., lead_stat_rate=.25),
            'empirical': realize(plan, pair_rate=0., stat_richness=1.,
                                 discourse_entropy=1., lead_stat_rate=.25,
                                 empirical_stat_rate=1.),
        }
        for label, doc in docs.items():
            original = answers(doc)
            counts.update(numeric_facts(doc))
            counts['plans/'+label] += 1
            for field, value in (
                    ('swap_teams', True), ('high_score', plan.high_score+1)):
                changed = realize(edit(plan, field, value), pair_rate=0.,
                                  stat_richness=1., discourse_entropy=1.,
                                  lead_stat_rate=.25,
                                  empirical_stat_rate=int(label == 'empirical'))
                after = answers(changed)
                if field == 'swap_teams':
                    assert after['winner'] != original['winner']
                    assert after['margin'] == original['margin']
                else:
                    assert int(after['margin']) == int(original['margin'])+1
                    assert after['winner'] == original['winner']
                assert after['top_player'] == original['top_player']
                numeric_facts(changed)
                counts['edit/'+label+'/'+field] += 1
            altered = list(plan.points)
            loser = next(i for i, value in enumerate(altered)
                         if value != max(altered))
            altered[loser] = max(altered)+1
            changed = realize(edit(plan, 'points', altered), pair_rate=0.,
                              stat_richness=1., discourse_entropy=1.,
                              lead_stat_rate=.25,
                              empirical_stat_rate=int(label == 'empirical'))
            after = answers(changed)
            assert after['top_player'] == plan.players[loser]
            assert all(after[k] == original[k] for k in ('winner', 'margin'))
            numeric_facts(changed)
            counts['edit/'+label+'/points'] += 1
            crop = first_words(doc.text, words)
            if len(crop.split()) >= words:
                crops[label].append(crop)
    n_pairs = min(map(len, crops.values()))
    human_by_game = {}
    for row in resources()[0][source][0]:
        if game_partition(game_key(row)) == group:
            crop = first_words(' '.join(row['summary']), words)
            if len(crop.split()) >= words:
                human_by_game.setdefault(game_key(row), crop)
    human = random.Random(seed+70000).sample(list(human_by_game.values()),
                                             n_pairs)
    return {'counts': dict(counts), 'paired_crops': n_pairs,
            'form': {label: structure(texts[:n_pairs])
                     for label, texts in crops.items()} |
                    {'human_reference': structure(human)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=37201)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--words', type=int, default=55)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {'config': {'seed': args.seed, 'n': args.n,
                         'words': args.words, 'bank_forms': len(bank())},
              'arms': {},
              'caveat': 'Paired sampled worlds; the two renderers change only '
                        'player-stat wording and sentence splitting. This '
                        'does not validate complete-document naturalness.'}
    for source in ('rotowire', 'sbnation'):
        for group in ('train', 'heldout'):
            key = source+'_'+group
            result['arms'][key] = panel(args.seed+len(result['arms'])*100,
                                        args.n, source, group, args.words)
            row = result['arms'][key]
            print(key, row['paired_crops'],
                  row['form']['baseline']['repeated_across_docs_trigram_occurrence_share'],
                  row['form']['empirical']['repeated_across_docs_trigram_occurrence_share'],
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
