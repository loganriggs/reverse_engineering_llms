"""Graph legality, move diversity, and DGP-control checks for plan sampler."""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path

from document_proposition_plan import FORMS, sample_plan, validate


def check_rejected(plan):
    corrupted = copy.deepcopy(plan)
    comparison = next(node for node in corrupted['nodes']
                      if node['kind'] == 'comparison')
    comparison['value'] = ('different' if comparison['value'] == 'same'
                           else 'same')
    try:
        validate(corrupted)
    except AssertionError:
        return True
    raise AssertionError('False comparison was accepted')


def audit(seed=8817, per_form=200):
    rng = random.Random(seed)
    by_form = {}
    total_sequences = set()
    for form in FORMS:
        rows = []
        for coupling in (0., 1.):
            counts, openings, sequences = Counter(), Counter(), set()
            for _ in range(per_form):
                plan = sample_plan(
                    rng, form=form, coupling_rate=coupling,
                    question_rate=1.)
                assert validate(plan)
                sequence = tuple(move['kind'] for move in plan['moves'])
                openings[plan['opening_role']] += 1
                sequences.add((plan['opening_role'], sequence))
                total_sequences.add((form, plan['opening_role'], sequence))
                counts.update(sequence)
                if any(node['kind'] == 'comparison'
                       for node in plan['nodes']):
                    assert check_rejected(plan)
            rows.append(dict(coupling_rate=coupling,
                             plans=per_form,
                             unique_sequences=len(sequences),
                             openings=dict(openings),
                             moves=dict(counts)))
        by_form[form] = rows
    all_report = {form: float(form == 'report') for form in FORMS}
    for _ in range(50):
        assert sample_plan(rng, form_weights=all_report)['form'] == 'report'
    comparison_sweep = []
    for panel_seed in (4014, 4015):
        for shared_rate in (0., .35, .6, 1.):
            panel_rng = random.Random(panel_seed)
            labels = Counter()
            for _ in range(1000):
                plan = sample_plan(panel_rng, coupling_rate=.7,
                                   shared_attribute_rate=shared_rate)
                labels.update(plan['nodes'][move['refs'][0]]['value']
                              for move in plan['moves']
                              if move['kind'] == 'compare')
            total = sum(labels.values())
            comparison_sweep.append(dict(seed=panel_seed,
                                         shared_attribute_rate=shared_rate,
                                         labels=dict(labels),
                                         majority_baseline=max(
                                             labels.values())/total))
    return dict(seed=seed, per_form_per_coupling=per_form,
                plans=2*per_form*len(FORMS),
                unique_form_sequence_pairs=len(total_sequences),
                by_form=by_form,
                comparison_sweep=comparison_sweep,
                form_prior_override_checks=50,
                caveat='Plan-level truth and dependency checks only. '
                       'No surface language, model learning, or realism '
                       'claim follows from move-sequence diversity.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('plans', result['plans'], 'unique',
          result['unique_form_sequence_pairs'])
    for form, rows in result['by_form'].items():
        print(form, [(row['coupling_rate'],
                      row['moves'].get('attribute', 0),
                      row['moves'].get('revisit', 0),
                      row['moves'].get('question', 0))
                     for row in rows])


if __name__ == '__main__':
    main()
