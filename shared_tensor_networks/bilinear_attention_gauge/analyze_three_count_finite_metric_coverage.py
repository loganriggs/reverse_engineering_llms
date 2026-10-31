"""Summarize finite usage-metric fits with model-seed clustered intervals."""

import json

import numpy as np

from dual_joint_function_canonical import HERE


BOOTSTRAPS = 20000
SEED = 20261031
MEASURES = ('empirical', 'floor_0.05', 'floor_0.5')


def mean_ci(values, draws):
    values = np.asarray(values)
    sampled = values[draws].mean(axis=1)
    return dict(mean=float(values.mean()),
                model_seed_bootstrap_ci95=np.quantile(sampled, [.025, .975]).tolist())


def run():
    source = json.loads((HERE/'three_count_finite_metric_coverage.json').read_text())
    records = source['records']
    seeds = list(range(source['model_seeds']))
    sizes = source['sequence_sample_sizes']
    repeats = source['data_repeats']
    expected = len(seeds)*2 + len(seeds)*len(sizes)*repeats*2*len(MEASURES)
    assert len(records) == expected
    indexed = {(r['seed'], r['sample_size'], r['repeat'], r['rank'], r['measure']):r
               for r in records}
    assert len(indexed) == len(records)
    draws = np.random.default_rng(SEED).integers(len(seeds), size=(BOOTSTRAPS,len(seeds)))
    summaries = {}
    for n in sizes:
        by_measure = {}
        for rank in (2,3):
            for measure in MEASURES:
                panels = [[indexed[(seed,n,repeat,rank,measure)]
                           for repeat in range(repeats)] for seed in seeds]
                rare = [np.mean([x['task']['rare_heldout']['all_four_accuracy']
                                 for x in group]) for group in panels]
                full = [np.mean([x['task']['all']['all_four_accuracy']
                                 for x in group]) for group in panels]
                table = [np.mean([x['full_relative_table_mse'] for x in group])
                         for group in panels]
                by_measure[f'{rank}:{measure}'] = dict(
                    rare_heldout_accuracy=mean_ci(rare,draws),
                    full_task_accuracy=mean_ci(full,draws),
                    broad_relative_table_mse=mean_ci(table,draws),
                    max_all_pair_edit_error=float(max(max(x['all_pair_edit_error_by_query'])
                        for group in panels for x in group)))
        reference = [indexed[(seed,None,None,2,'uniform_weight_only')]
                     for seed in seeds]
        for measure in MEASURES:
            rare = np.asarray([np.mean([indexed[(seed,n,repeat,2,measure)]
                  ['task']['rare_heldout']['all_four_accuracy']
                  for repeat in range(repeats)]) for seed in seeds])
            uniform_rare = np.asarray([x['task']['rare_heldout']['all_four_accuracy']
                                       for x in reference])
            by_measure[f'2:{measure}']['rare_minus_uniform_rank2'] = mean_ci(
                rare-uniform_rare,draws)
        for other in ('floor_0.05','floor_0.5'):
            difference = np.asarray([np.mean([
                indexed[(seed,n,repeat,2,other)]['task']['rare_heldout']['all_four_accuracy']-
                indexed[(seed,n,repeat,2,'empirical')]['task']['rare_heldout']['all_four_accuracy']
                for repeat in range(repeats)]) for seed in seeds])
            by_measure[f'2:{other}']['rare_minus_empirical_rank2'] = mean_ci(
                difference,draws)
        coverage = [indexed[(seed,n,repeat,2,'empirical')]
                    for seed in seeds for repeat in range(repeats)]
        summaries[str(n)] = dict(
            n_panels=len(coverage),
            no_rare_query_panels=sum(x['rare_query_count']==0 for x in coverage),
            mean_covered_rare_key_cells=float(np.mean([x['covered_rare_key_cells']
                                                       for x in coverage])),
            measures=by_measure)
    rank_selection=[]
    for n in sizes:
        for seed in seeds:
            for repeat in range(repeats):
                for measure in MEASURES:
                    for threshold in (.01,.05):
                        candidate = next(rank for rank in (2,3) if max(
                            indexed[(seed,n,repeat,rank,measure)]
                            ['all_pair_edit_error_by_query']) <= threshold)
                        rank_selection.append(dict(seed=seed,sample_size=n,
                            repeat=repeat,measure=measure,
                            threshold=threshold,selected_rank=candidate))
    assert len(rank_selection)==len(seeds)*len(sizes)*repeats*len(MEASURES)*2
    result=dict(source='three_count_finite_metric_coverage.json',
        model_seed_bootstraps=BOOTSTRAPS,bootstrap_seed=SEED,
        summaries=summaries,
        answer_free_all_pair_edit_rank_selection=dict(
            n_choices=len(rank_selection),
            rank_counts={str(rank):sum(x['selected_rank']==rank for x in rank_selection)
                         for rank in (2,3)},
            thresholds=(.01,.05)),
        rank3_worst_rare_heldout_accuracy=float(min(
            x['task']['rare_heldout']['all_four_accuracy'] for x in records
            if x['rank']==3 and x['measure'] in MEASURES)))
    (HERE/'three_count_finite_metric_coverage_analysis.json').write_text(
        json.dumps(result,indent=2)+'\n')
    print('rank choices',result['answer_free_all_pair_edit_rank_selection'])
    print('rank3 worst',result['rank3_worst_rare_heldout_accuracy'])
    for n in sizes:
        x=summaries[str(n)]
        print(n,'coverage',x['mean_covered_rare_key_cells'],
              'no rare query',x['no_rare_query_panels'],
              'rank2 empirical rare',x['measures']['2:empirical']['rare_heldout_accuracy'],
              'floor .5 minus empirical',
              x['measures']['2:floor_0.5']['rare_minus_empirical_rank2'])


if __name__=='__main__':
    run()
