"""Compare head rankings from weight-only scores with held-out ablations."""

import argparse
import json
from pathlib import Path


def pairwise_agreement(predicted, observed):
    concordant = discordant = tied = 0
    for i in range(len(predicted)):
        for j in range(i+1, len(predicted)):
            a = predicted[i]-predicted[j]
            b = observed[i]-observed[j]
            if a == 0 or b == 0:
                tied += 1
            elif a*b > 0:
                concordant += 1
            else:
                discordant += 1
    return dict(concordant=concordant, discordant=discordant,
                tied=tied, agreement_fraction=(concordant/
                                                (concordant+discordant)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=Path,
                   default=Path(__file__).parent / 'results')
    p.add_argument('--seeds', default='71411,71412,71413,71414,71415')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = []
    for seed in (int(v) for v in args.seeds.split(',')):
        behavior_path = args.results_dir / (
            f'corpus_delimiter_bilinear_site50_truenone_seed{seed}'
            '_44000_52000.json' if seed != 71411 else
            'corpus_delimiter_bilinear_site50_truenone_44000_52000.json')
        behavior = json.loads(behavior_path.read_text())['results']['2']['test']
        probe = json.loads((args.results_dir /
            f'corpus_delimiter_weight_vs_causal_heads_truenone_seed{seed}.json'
                            ).read_text())
        path = json.loads((args.results_dir /
            f'corpus_delimiter_full_weight_path_seed{seed}.json'
                           ).read_text())
        second_heads = [row for row in probe['head_ablations']
                        if row['layer_index'] == 2]
        first_whole = next(row for row in
                           probe['attention_layer_ablations']
                           if row['layer_index'] == 0)
        observed = dict(
            accuracy=[row['drop_both_two_way'] for row in second_heads],
            margin=[row['drop_mean_margin_shift'] for row in second_heads],
            matched_accuracy=[row['matched_drop_both_two_way']
                              for row in second_heads],
            matched_margin=[row['matched_drop_mean_margin_shift']
                            for row in second_heads])
        weight = probe['data_free_first_layer_qk'][
            'second_attention_after_first_mlp']['space']
        methods = dict(
            qk_rms=weight['difference_rms'],
            qk_relative_rms=weight['relative_difference_rms'],
            full_path_uniform=[row['mean_signed_drop'] for row in
                               path['uniform_vocabulary']['heads']],
            full_path_train_site=[row['mean_signed_drop'] for row in
                                  path['training_task_site_queries']['heads']],
            qk_ov_readout_uniform=[row['mean_signed_drop'] for row in
                                   path['ov_readout_uniform']['heads']],
            qk_ov_readout_train_site=[row['mean_signed_drop'] for row in
                                      path['ov_readout_training_task_site']['heads']])
        comparisons = {}
        for name, values in methods.items():
            comparisons[name] = {
                target: pairwise_agreement(values, outcome)
                for target, outcome in observed.items()}
            comparisons[name]['top_head'] = max(range(len(values)),
                                                 key=lambda i: values[i])
        eligible = (behavior['both_two_way'] >= .7 and
                    behavior['positive_shift_fraction'] >= .9 and
                    abs(first_whole['drop_both_two_way']) <= .03)
        rows.append(dict(seed=seed, behavior=behavior['both_two_way'],
                         positive_shift=behavior['positive_shift_fraction'],
                         first_attention_whole_drop=(
                             first_whole['drop_both_two_way']),
                         matched_pure_pairs=(probe['matched_pure_test']['n']),
                         eligible_for_approximation=eligible,
                         causal_second_head_drops=observed,
                         methods=methods, comparisons=comparisons,
                         causal_top_head=max(range(4),
                                             key=lambda i: observed['accuracy'][i]),
                         matched_causal_top_head=max(
                             range(4),
                             key=lambda i: observed['matched_accuracy'][i])))
    summary = {}
    for method in rows[0]['methods']:
        selected = [row for row in rows
                    if row['eligible_for_approximation']]
        summary[method] = dict(
            eligible_seeds=len(selected),
            top_head_matches=sum(
                row['comparisons'][method]['top_head'] ==
                row['causal_top_head'] for row in selected),
            matched_top_head_matches=sum(
                row['comparisons'][method]['top_head'] ==
                row['matched_causal_top_head'] for row in selected),
            accuracy_pairwise_agreement=(
                sum(row['comparisons'][method]['accuracy']['concordant']
                    for row in selected) /
                sum(row['comparisons'][method]['accuracy']['concordant']+
                    row['comparisons'][method]['accuracy']['discordant']
                    for row in selected)),
            margin_pairwise_agreement=(
                sum(row['comparisons'][method]['margin']['concordant']
                    for row in selected) /
                sum(row['comparisons'][method]['margin']['concordant']+
                    row['comparisons'][method]['margin']['discordant']
                    for row in selected)),
            matched_accuracy_pairwise_agreement=(
                sum(row['comparisons'][method]['matched_accuracy']['concordant']
                    for row in selected) /
                sum(row['comparisons'][method]['matched_accuracy']['concordant']+
                    row['comparisons'][method]['matched_accuracy']['discordant']
                    for row in selected)),
            matched_margin_pairwise_agreement=(
                sum(row['comparisons'][method]['matched_margin']['concordant']
                    for row in selected) /
                sum(row['comparisons'][method]['matched_margin']['concordant']+
                    row['comparisons'][method]['matched_margin']['discordant']
                    for row in selected)))
    result = dict(seeds=rows, summary=summary,
                  gate='held-out paired two-way >=.70, correct shift >=.90, '
                       'whole first-attention ablation <=.03 absolute '
                       'accuracy change',
                  warning='Five seeds, four heads each. Pairwise ranking '
                          'is diagnostic, not a circuit-recovery proof. '
                          'Full-path calculation uses a two-token/zero-'
                          'middle context and zero first attention. Matched '
                          'causal subset has pure closer tokens and the '
                          'same leading-space opener tokens as weight score.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
