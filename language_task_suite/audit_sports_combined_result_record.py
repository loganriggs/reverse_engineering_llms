"""RotoWire combined result/record form, semantics and intervention audit."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random

from audit_document_structure_vs_fineweb import structure
from audit_rotowire_document_source import game_key
from compare_v12_fineweb_fresh import first_words
from sports_combined_result_record_renderer import bank, visible_records
from sports_document_actplan_v1 import (
    answer_from_visible_prefix, edit, realize, sample_plan,
)
from sports_document_dgp import game_partition, realize as base_realize, resources


def score_sentence(doc):
    rows = [sentence.strip()+'.' for sentence in doc.text.split('.')
            if doc.plan.base.home in sentence and doc.plan.base.away in sentence]
    if len(rows) != 1:
        raise ValueError('Expected one result sentence')
    return rows[0]


def checked(doc):
    sentence = score_sentence(doc)
    records = visible_records(sentence,
                              (doc.plan.base.home, doc.plan.base.away))
    assert records
    assert all(value == doc.plan.after[team]
               for team, value in records.items())
    seen = {}
    for site in doc.sites:
        assert doc.text[site.start:site.end] == site.answer
        assert answer_from_visible_prefix(doc.text[:site.start],
                                          site.kind) == site.answer
        seen[site.kind] = site.answer
    assert seen
    return seen, records


def draw(rng, group):
    plan = sample_plan(rng, source_group=group, sbnation_rate=0.)
    return plan, realize(plan, record_rate=0., combined_record_rate=1.)


def semantic(seed, n, group):
    rng = random.Random(seed)
    counts = Counter()
    generated = []
    for _ in range(n):
        plan, doc = draw(rng, group)
        assert game_partition(plan.base.source_game_key) == group
        original, records = checked(doc)
        counts['documents'] += 1
        counts['inline_records'] += len(records)
        for kind in original:
            counts['site/'+kind] += 1
        if len(generated) < 2:
            generated.append(doc.text)
        for field, value in (('swap_teams', True),
                             ('high_score', plan.base.high_score+1)):
            changed, new_records = checked(realize(
                edit(plan, field, value), record_rate=0.,
                combined_record_rate=1.))
            assert set(changed) == set(original)
            if field == 'swap_teams':
                if 'winner' in original:
                    assert changed['winner'] != original['winner']
                for kind in ('margin', 'top_player'):
                    if kind in original:
                        assert changed[kind] == original[kind]
            else:
                if 'margin' in original:
                    assert int(changed['margin']) == int(original['margin'])+1
                for kind in ('winner', 'top_player'):
                    if kind in original:
                        assert changed[kind] == original[kind]
            counts['edit/'+field] += 1
        if 'top_player' in original:
            points = list(plan.base.points)
            index = next(i for i, p in enumerate(points) if p != max(points))
            points[index] = max(points)+1
            changed, _ = checked(realize(
                edit(plan, 'points', points), record_rate=0.,
                combined_record_rate=1.))
            assert changed['top_player'] == plan.base.players[index]
            counts['edit/points'] += 1
        w, l = plan.home_before
        changed_plan = replace(plan, home_before=(w+1, l))
        changed, new_records = checked(realize(
            changed_plan, record_rate=0., combined_record_rate=1.))
        assert changed == original
        if plan.base.home in records:
            assert new_records[plan.base.home][0] == \
                records[plan.base.home][0]+1
        counts['edit/record_prior'] += 1
    return {'counts': dict(counts), 'generated_examples': generated}


def form(seed, n, words, group):
    rng = random.Random(seed)
    crops = {name: [] for name in ('fixed', 'variable_no_record',
                                   'combined_record')}
    windows = {kind: {name: [] for name in crops}
               for kind in ('winner', 'margin', 'top_player')}
    attempts = 0
    while len(crops['fixed']) < n:
        attempts += 1
        if attempts > 200*n:
            raise RuntimeError('Insufficient jointly eligible documents')
        plan = sample_plan(rng, source_group=group, sbnation_rate=0.)
        docs = {
            'fixed': base_realize(plan.base, pair_rate=0.,
                                  stat_richness=1.,
                                  discourse_entropy=1.,
                                  lead_stat_rate=.25,
                                  empirical_stat_rate=1.),
            'variable_no_record': realize(plan, record_rate=0.),
            'combined_record': realize(plan, record_rate=0.,
                                       combined_record_rate=1.),
        }
        texts = {name: first_words(doc.text, words)
                 for name, doc in docs.items()}
        if min(len(text.split()) for text in texts.values()) < words:
            continue
        for name, text in texts.items():
            crops[name].append(text)
        common = set.intersection(*(set(site.kind for site in doc.sites)
                                    for doc in docs.values()))
        for kind in common:
            for name, doc in docs.items():
                site = next(site for site in doc.sites if site.kind == kind)
                windows[kind][name].append(' '.join(
                    doc.text[:site.start].split()[-25:]+
                    doc.text[site.start:].split()[:16]))
    human_by_game = {}
    for row in resources()[0]['rotowire'][0]:
        if game_partition(game_key(row)) != group:
            continue
        text = first_words(' '.join(row['summary']), words)
        if len(text.split()) >= words:
            human_by_game.setdefault(game_key(row), text)
    human = random.Random(seed+30000).sample(
        list(human_by_game.values()), n)
    return {'attempts': attempts,
            'panels': {name: structure(texts) for name, texts in crops.items()}
                      | {'human': structure(human)},
            'rule_windows': {kind: {name: structure(texts)
                                   for name, texts in groups.items()}
                             for kind, groups in windows.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=38401)
    parser.add_argument('--semantic-n', type=int, default=100)
    parser.add_argument('--form-n', type=int, default=64)
    parser.add_argument('--words', type=int, default=55)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = {'config': vars(args) | {'output': str(args.output)},
           'bank': {'distinct': len(bank()),
                    'train_occurrences': sum(bank().values())},
           'arms': {},
           'caveat': 'Training-game form bank only; heldout game worlds. '
                     'Source sentences are not saved. A guarded form is '
                     'not by itself proof of realistic full documents.'}
    for group in ('train', 'heldout'):
        out['arms'][group] = {
            'semantic': semantic(args.seed+int(group == 'heldout')*100,
                                 args.semantic_n, group),
            'form': form(args.seed+10000+int(group == 'heldout')*100,
                         args.form_n, args.words, group),
        }
        p = out['arms'][group]['form']['panels']
        print(group, out['arms'][group]['form']['attempts'],
              {name: round(m[
                  'repeated_across_docs_trigram_occurrence_share'], 3)
               for name, m in p.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
