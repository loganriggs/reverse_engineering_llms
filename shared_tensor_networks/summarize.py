"""Descriptive summaries; no hypothesis selection or significance claims."""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).parent
    records = {}
    summaries = []
    sizes = [8,16,32,64,128,256,512,1024]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for col, run in enumerate(['e1_sweep', 'e1_matched']):
        rows = list(csv.DictReader((root/'results'/run/'results.csv').open()))
        records[run] = {(r['teacher'], int(r['seed']), int(r['n']), r['student']):r for r in rows}
        for ri, teacher in enumerate(['tied','untied']):
            ax = axes[ri, col]
            for student in ['tied', 'untied_width', 'untied_budget']:
                values = [[float(records[run][teacher, seed, n, student]['exact_gaussian_mse']) for seed in range(5)] for n in sizes]
                median = np.median(values, axis=1)
                low, high = np.min(values, axis=1), np.max(values, axis=1)
                line, = ax.plot(sizes, median, marker='o', label=student)
                ax.fill_between(sizes, low, high, alpha=.12, color=line.get_color())
            for n in sizes:
                entry = dict(run=run, teacher=teacher, n=n)
                for student in ['tied','untied_width','untied_budget']:
                    entry[student+'_median'] = statistics.median(float(records[run][teacher, seed, n, student]['exact_gaussian_mse']) for seed in range(5))
                for other in ['untied_width','untied_budget']:
                    t = [float(records[run][teacher, seed, n, 'tied']['exact_gaussian_mse']) for seed in range(5)]
                    u = [float(records[run][teacher, seed, n, other]['exact_gaussian_mse']) for seed in range(5)]
                    entry['tied_wins_vs_'+other] = sum(a<b for a,b in zip(t,u))
                    entry['paired_tied_minus_'+other] = [a-b for a,b in zip(t,u)]
                summaries.append(entry)
            ax.axhline(1., color='black', linestyle=':', linewidth=1, label='zero predictor')
            ax.set_xscale('log', base=2); ax.set_yscale('log')
            ax.set_title(teacher+' teacher / '+('independent right branch' if col == 0 else 'matched initial function (width pair)'))
            ax.grid(alpha=.2)
            ax.set_xlabel('Training examples'); ax.set_ylabel('Exact Gaussian MSE')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Medians and full ranges across five paired seeds; ranges are not confidence intervals')
    fig.tight_layout()
    (root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures/e1_population_error.png', dpi=150)
    # Unchanged arms must reproduce exactly; only width-matched untied initialization changes.
    max_init_error = 0.
    for key, row in records['e1_sweep'].items():
        if key[-1] in ['tied','untied_budget']:
            other = records['e1_matched'][key]
            for metric in ['train_mse','test_mse','exact_gaussian_mse','coefficient_relative_error']:
                assert row[metric] == other[metric], (key,metric)
        if key[-1] == 'tied':
            other = records['e1_matched'][key[:-1]+('untied_width',)]
            max_init_error = max(max_init_error, abs(float(row['init_output_rms'])-float(other['init_output_rms'])))
    assert max_init_error == 0
    result = dict(summary=summaries, unchanged_arm_exact_replays=160, matched_initial_output_rms_max_error=max_init_error)
    (root/'results/summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
