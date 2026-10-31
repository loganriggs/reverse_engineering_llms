"""Summarize weak-rule reduction gates and fixed-function gauge outcomes."""

import json

import numpy as np

from dual_joint_function_canonical import HERE


MEASURES=('uniform_weight_only','empirical_usage',
          'half_uniform_empirical','output_balanced_weight_only')
PRIORS=('svd','key_l1','all_pair_edits','exclude_D_edits')
THRESHOLD=.01


def average(rows,key):
    return float(np.mean([key(x) for x in rows]))


def run():
    reduction=json.loads((HERE/'four_count_weak_rule_reduction.json').read_text())
    gauges=json.loads((HERE/'four_count_weak_rule_gauges.json').read_text())
    rows=reduction['records']
    assert len(rows)==10*2*2*len(MEASURES)
    assert len(gauges['rows'])==10*len(MEASURES)*len(PRIORS)
    summaries={}
    for calibration in (1.,.1):
        for measure in MEASURES:
            selected=[x for x in rows if x['calibration']==calibration and
                      x['measure']==measure]
            assert len(selected)==20
            by_rank={rank:[x for x in selected if x['rank']==rank] for rank in (3,4)}
            result={}
            for rank,panel in by_rank.items():
                result[str(rank)]=dict(
                    full_joint_accuracy=average(panel,lambda x:x['task']['full_joint_accuracy']),
                    weak_D_accuracy=average(panel,lambda x:x['task']['weak_D_accuracy']),
                    rare_heldout_joint_accuracy=average(panel,
                        lambda x:x['task']['rare_heldout_joint_accuracy']),
                    broad_relative_table_mse=average(panel,
                        lambda x:x['broad_relative_table_mse']),
                    all_pair_global_edit_error=average(panel,
                        lambda x:x['all_pair_edits']['global_relative_mse']),
                    all_pair_max_output_edit_error=average(panel,
                        lambda x:x['all_pair_edits']['max_output_relative_mse']),
                    D_excluded_global_edit_error=average(panel,
                        lambda x:x['D_family_withheld_edits']['global_relative_mse']),
                    max_output_relative_table_error=average(panel,
                        lambda x:x['max_output_relative_table_mse']))
            gates={}
            metrics={
                'global_all_pair_edit':lambda x:x['all_pair_edits']['global_relative_mse'],
                'global_D_excluded_edit':lambda x:x['D_family_withheld_edits']['global_relative_mse'],
                'max_output_all_pair_edit':lambda x:x['all_pair_edits']['max_output_relative_mse'],
                'max_output_table':lambda x:x['max_output_relative_table_mse']}
            for label,metric in metrics.items():
                chosen=[]
                for seed in range(10):
                    pool={x['rank']:x for x in selected if x['seed']==seed}
                    assert metric(pool[4])<THRESHOLD
                    chosen.append(3 if metric(pool[3])<THRESHOLD else 4)
                gates[label]=dict(rank3_count=chosen.count(3),rank4_count=chosen.count(4))
            summaries[f'{calibration:g}:{measure}']=dict(ranks=result,rank_gates=gates)
    gauge_summaries={}
    for prior in PRIORS:
        part=[x for x in gauges['rows'] if x['prior']==prior]
        assert len(part)==40
        gauge_summaries[prior]=dict(
            mean_variable_alignment=average(part,lambda x:x['mean_variable_alignment']),
            min_variable_alignment=float(min(x['mean_variable_alignment'] for x in part)),
            mean_D_alignment=average(part,lambda x:x['D_alignment']),
            D_accuracy_after_matched_axis_removal=average(part,
                lambda x:x['D_removal']['D_accuracy']),
            other_three_joint_accuracy_after_D_axis_removal=average(part,
                lambda x:x['D_removal']['other_three_joint_accuracy']),
            mean_random_rotation_alignment=average(part,
                lambda x:x['random_control']['mean_alignment']),
            mean_random_rotation_D_accuracy_after_axis_removal=average(part,
                lambda x:x['random_control']['D_after_removal_accuracy']),
            mean_random_rotation_other_three_accuracy_after_axis_removal=average(part,
                lambda x:x['random_control']['other_three_after_removal_accuracy']))
    model_checks={}
    for calibration in (1.,.1):
        part=[x for x in reduction['model_checks'] if x['calibration']==calibration]
        model_checks[str(calibration)]=dict(
            mean_median_D_margin=average(part,
                lambda x:x['median_calibrated_D_logit_margin']),
            expected_D_accuracy_with_gaussian_logit_noise_sigma_0_5=average(part,
                lambda x:x['expected_D_accuracy_under_independent_gaussian_logit_noise']['0.5']),
            expected_D_accuracy_with_gaussian_logit_noise_sigma_1=average(part,
                lambda x:x['expected_D_accuracy_under_independent_gaussian_logit_noise']['1.0']))
    output=dict(reduction_source='four_count_weak_rule_reduction.json',
        gauge_source='four_count_weak_rule_gauges.json',
        rank_gate_threshold=THRESHOLD,measure_summaries=summaries,
        gauge_summaries=gauge_summaries,model_checks=model_checks)
    (HERE/'four_count_weak_rule_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print('weak D cuts')
    for measure in MEASURES:
        row=summaries[f'0.1:{measure}']
        print(measure,row['ranks']['3']['weak_D_accuracy'],row['ranks']['3'][
            'all_pair_global_edit_error'],row['rank_gates'])
    print('gauge summaries',gauge_summaries)


if __name__=='__main__':run()
