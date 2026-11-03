"""Compare semantic-plan coherence controls without claiming naturalness."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from document_proposition_plan import FORMS, sample_plan, validate


def summarize(plan):
    nodes, moves = plan['nodes'], plan['moves']
    focus = next(move['entity'] for move in moves if move['kind'] == 'intro')
    comparisons, measures, records, attributes, recalls = (
        set(), set(), set(), set(), set())
    repeats = Counter()
    focal, topical = 0, 0
    for index, move in enumerate(moves):
        kind = move['kind']
        node = nodes[move['refs'][0]] if move['refs'] else None
        if kind in ('update', 'measure', 'record', 'step'):
            topical += 1
            focal += node['entity'] == focus
        if kind == 'compare':
            pair = tuple(sorted((node['entity'], node['other'])))
            signature = (pair, node['attr'], tuple(sorted(node['deps'])))
            repeats['comparison'] += signature in comparisons
            comparisons.add(signature)
        elif kind == 'measure':
            signature = node['entity'], node['unit']
            repeats['measure'] += signature in measures
            measures.add(signature)
        elif kind == 'record':
            signature = node['scope'], node['entity'], node['key']
            repeats['record'] += signature in records
            records.add(signature)
        elif kind == 'attribute':
            signature = node['deps'][0]
            repeats['attribute'] += signature in attributes
            attributes.add(signature)
        elif kind in ('revisit', 'question'):
            signature = move['refs'][0]
            repeats['recall'] += signature in recalls
            recalls.add(signature)
            repeats['early_recall'] += index < 6
        if index and kind == moves[index-1]['kind'] and kind != 'step':
            repeats['adjacent_same_move'] += 1
    return dict(repeats), focal, topical


def audit(seed=31191, per_form=500):
    rows = []
    for style in ('independent', 'threaded'):
        for form_index, form in enumerate(FORMS):
            repetitions = Counter()
            unique_sequences = set()
            focal = topical = 0
            move_counts = Counter()
            for i in range(per_form):
                rng = random.Random(seed + form_index*100_000 + i)
                plan = sample_plan(rng, form=form, n_moves=11,
                                   coupling_rate=.7, question_rate=1.,
                                   planning_style=style)
                assert validate(plan)
                unique_sequences.add(tuple(
                    move['kind'] for move in plan['moves']))
                move_counts.update(move['kind'] for move in plan['moves'])
                counts, focus_count, topical_count = summarize(plan)
                repetitions.update(counts)
                focal += focus_count
                topical += topical_count
            rows.append(dict(style=style, form=form, plans=per_form,
                             unique_move_sequences=len(unique_sequences),
                             repeated_events=dict(repetitions),
                             focus_share=focal/topical if topical else 0.,
                             topical_moves=topical,
                             move_counts=dict(move_counts)))
    focus_sweep = []
    for form in ('report', 'procedure', 'commercial'):
        for focus_rate in (0., .5, 1.):
            focal = topical = 0
            for i in range(per_form):
                rng = random.Random(seed + 7_000_000 +
                                    FORMS.index(form)*100_000 + i)
                plan = sample_plan(rng, form=form, n_moves=11,
                                   coupling_rate=.7,
                                   planning_style='threaded',
                                   focus_rate=focus_rate)
                _, focus_count, topical_count = summarize(plan)
                focal += focus_count
                topical += topical_count
            focus_sweep.append(dict(form=form, focus_rate=focus_rate,
                                    focus_share=focal/topical,
                                    topical_moves=topical))
    return dict(seed=seed, per_form=per_form, rows=rows,
                focus_sweep=focus_sweep,
                caveat='Constructed coherence diagnostics, not a measure '
                       'of natural-language realism.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['rows']:
        print(row['style'], row['form'],
              row['unique_move_sequences'], row['focus_share'],
              row['repeated_events'])


if __name__ == '__main__':
    main()
