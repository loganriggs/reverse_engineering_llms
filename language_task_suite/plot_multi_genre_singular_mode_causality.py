"""Plot score energy versus causal task damage for direct mode deletions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PANELS = [json.loads((HERE/name).read_text()) for name in (
    'multi_genre_layer0_token_tensor_mode_ablation.json',
    'multi_genre_layer0_token_tensor_mode_ablation_replication.json')]
MODELS = (('multi_genre_models_v1', 'Model seed 0, head 1'),
          ('multi_genre_models_seed1', 'Model seed 1, head 3'))


def values(panel, model_name, task):
    data = panel['results'][model_name]
    baseline = data['full_table_baseline']['groups'][task]['oracle_kl']
    rows = {row['singular_index']: row for row in data['rows']
            if row['lag_start'] == 0 and row['lag_end'] == 70}
    damage = np.array([rows[i]['groups'][task]['oracle_kl']-baseline
                       for i in range(3)])
    energy = np.array([rows[i]['score_energy_removed'] for i in range(3)])
    return energy, damage


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex='col',
                             constrained_layout=True)
    colors = {'FAQ recall': '#2456a6', 'Policy action': '#b85d27'}
    x = np.arange(3)
    for col, (name, title) in enumerate(MODELS):
        energy, _ = values(PANELS[0], name, 'faq_place_recall')
        axes[0, col].bar(x, energy*100, color='#757575')
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel('Score energy removed (%)')
        axes[0, col].set_ylim(0, 85)
        for label, task in (('FAQ recall', 'faq_place_recall'),
                            ('Policy action', 'policy_mixed_action')):
            damage = np.stack([values(panel, name, task)[1]
                               for panel in PANELS])
            mean = damage.mean(0)
            low = mean-damage.min(0)
            high = damage.max(0)-mean
            axes[1, col].errorbar(x, mean, yerr=[low, high], marker='o',
                                  linewidth=1.8, capsize=3, label=label,
                                  color=colors[label])
        axes[1, col].axhline(0, color='#999999', linewidth=.8)
        axes[1, col].set_ylabel('Increase in oracle KL (nats)')
        axes[1, col].set_xticks(x, ['Top', 'Second', 'Third'])
        axes[1, col].set_xlabel('Deleted singular term at every lag')
        axes[1, col].set_ylim(-.025, .68)
    axes[1, 0].legend(loc='upper right')
    fig.suptitle('First-layer score energy does not rank causal importance')
    out = HERE/'multi_genre_singular_mode_causality.png'
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == '__main__':
    main()
