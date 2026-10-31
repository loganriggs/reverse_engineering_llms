"""Paired-trial accuracy intervals for active internal-intervention selection."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_STEM = 'active_intervention_graph_selection_s1201_t1000'
STRATEGIES = ('active', 'activity', 'input_norm', 'random')


def run(draws=10000, seed=6201, stem=DEFAULT_STEM):
    report = json.loads((HERE / f'{stem}.json').read_text())
    with gzip.open(HERE / f'{stem}_records.json.gz', 'rt') as stream:
        records = json.load(stream)
    assert len(records) == 2 * 3 * report['trials'] * len(report['sizes']) * 4
    lookup = {(r['teacher'], r['noise_fraction'], r['trial'],
               r['n_internal'], r['strategy']): r for r in records}
    assert len(lookup) == len(records)
    rng = np.random.default_rng(seed)
    summaries = {}
    for teacher in ('shared', 'duplicate'):
        for noise in (.5, 1., 2.):
            for n in (1, 4, 16):
                values = {strategy: np.array([
                    lookup[teacher, noise, trial, n, strategy]['correct']
                    for trial in range(report['trials'])], dtype=np.float64)
                    for strategy in STRATEGIES}
                indices = rng.integers(report['trials'],
                                       size=(draws, report['trials']))
                row = {}
                for control in STRATEGIES[1:]:
                    delta = values['active'] - values[control]
                    samples = delta[indices].mean(axis=1)
                    row[control] = dict(
                        active_minus_control_accuracy=float(delta.mean()),
                        paired_bootstrap_95=np.quantile(
                            samples, [.025, .975]).tolist(),
                        active_correct_control_wrong=int(np.sum(delta == 1)),
                        control_correct_active_wrong=int(np.sum(delta == -1)))
                summaries[f'{teacher}__noise{noise}__n{n}'] = row
    result = dict(description='Within-trial paired bootstrap comparing active graph-disagreement selection to high-output-activity, high-input-norm, and random selection from identical Gaussian pools. Trials vary teacher parameters and observation/intervention noise.', draws=draws,
                  seed=seed, source=stem, summary=summaries)
    path = HERE / f'{stem}_analysis.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    for key, row in summaries.items():
        print(key, 'active-activity',
              round(row['activity']['active_minus_control_accuracy'], 3),
              [round(v, 3) for v in row['activity']['paired_bootstrap_95']])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem', default=DEFAULT_STEM)
    run(stem=parser.parse_args().stem)
