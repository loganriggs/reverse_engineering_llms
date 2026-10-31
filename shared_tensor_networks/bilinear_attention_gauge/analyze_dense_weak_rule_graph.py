"""Audit functional shared-atom recovery after dense-code gauge fitting."""

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from dual_joint_function_canonical import HERE
from train_dense_four_count_weak import MIX, SIGNS


MEASURES=('uniform_weight_only','empirical_usage',
          'half_uniform_empirical','output_balanced_weight_only')
PRIORS=('svd','key_l1','all_pair_edits','same_raw_D_bit_edits')
EXPECTED=np.asarray([MIX[rule]!=0 for rule in (0,1,2,3,1,2,0,3)])
RAW=SIGNS.T/np.linalg.norm(SIGNS.T,axis=1,keepdims=True)
THRESHOLDS=(.05,.1,.2)
BOOTSTRAPS=20000


def graph_scores(left,dictionary):
    normalized=dictionary/np.linalg.norm(dictionary,axis=1,keepdims=True)
    correlation=normalized@RAW.T
    row,column=linear_sum_assignment(-np.abs(correlation))
    mapping={int(rule):int(axis) for axis,rule in zip(row,column)}
    scores=np.asarray([abs(correlation[mapping[rule],rule]) for rule in range(4)])
    coefficients=np.stack([left[:,mapping[rule]]*
        np.linalg.norm(dictionary[mapping[rule]])*
        np.sign(correlation[mapping[rule],rule]) for rule in range(4)],axis=1)
    graph={}
    for threshold in THRESHOLDS:
        member=np.abs(coefficients)>=threshold*np.max(np.abs(coefficients),axis=1,
                                                       keepdims=True)
        graph[str(threshold)]=dict(cell_accuracy=float(np.mean(member==EXPECTED)),
            exact_row_fraction=float(np.mean(np.all(member==EXPECTED,axis=1))),
            false_positive_cells=int(np.sum(member&~EXPECTED)),
            false_negative_cells=int(np.sum(~member&EXPECTED)))
    D_ratios=np.mean(np.abs(coefficients[[3,7]])/
                     np.abs(coefficients[[3,7],3,None]),axis=0)
    matrix=left@dictionary
    projected=(matrix@SIGNS/16)@SIGNS.T
    return dict(mean_raw_atom_alignment=float(scores.mean()),
        raw_atom_alignment_by_bit=scores.tolist(),
        graph=graph,D_output_shared_atom_ratios=D_ratios.tolist(),
        relative_residual_outside_raw_bit_span=float(
            np.sum((matrix-projected)**2)/np.sum(matrix**2)))


def model_seed_ci(rows,metric):
    per_seed=np.array([np.mean([metric(x) for x in rows if x['seed']==seed])
                       for seed in range(10)])
    rng=np.random.default_rng(20261031)
    indices=rng.integers(10,size=(BOOTSTRAPS,10))
    return dict(mean=float(per_seed.mean()),
        model_seed_bootstrap_ci95=np.quantile(per_seed[indices].mean(axis=1),
                                             [.025,.975]).tolist())


def run():
    reduction=json.loads((HERE/'dense_weak_rule_reduction.json').read_text())
    gauges=json.loads((HERE/'dense_weak_rule_gauges.json').read_text())
    factors=np.load(HERE/'dense_weak_rule_gauge_factors.npz')
    rows=[];random_rows=[]
    for seed in range(10):
        for measure_index,measure in enumerate(MEASURES):
            for prior in PRIORS:
                prefix=f's{seed}_{measure}_{prior}'
                left=factors[prefix+'_left']
                dictionary=factors[prefix+'_dictionary']
                score=graph_scores(left,dictionary)
                saved=next(x for x in gauges['rows'] if x['seed']==seed and
                           x['measure']==measure and x['prior']==prior)
                rows.append(dict(seed=seed,measure=measure,prior=prior,
                    **score,heldout_D_flip_response_error=saved['component'][
                        'D_flip_output_response_relative_error'],
                    other_three_accuracy_after_D_axis_removal=saved['component'][
                        'other_three_joint_accuracy_after_removal'],
                    D_accuracy_after_D_axis_removal=saved['component'][
                        'D_accuracy_after_removal']))
            left=factors[f's{seed}_{measure}_svd_left']
            dictionary=factors[f's{seed}_{measure}_svd_dictionary']
            rng=np.random.default_rng(20261101+seed*100+measure_index)
            for draw in range(32):
                raw=rng.standard_normal((4,4))
                q,r=np.linalg.qr(raw)
                q=q*np.sign(np.diag(r))[None,:]
                random_rows.append(dict(seed=seed,measure=measure,draw=draw,
                    **graph_scores(left@q.T,q@dictionary)))
    assert len(rows)==160 and len(random_rows)==1280
    summary={}
    for prior in PRIORS:
        part=[x for x in rows if x['prior']==prior]
        summary[prior]=dict(
            raw_atom_alignment=model_seed_ci(part,
                lambda x:x['mean_raw_atom_alignment']),
            graph_cell_accuracy_at_0_1=model_seed_ci(part,
                lambda x:x['graph']['0.1']['cell_accuracy']),
            graph_exact_row_fraction_at_0_1=model_seed_ci(part,
                lambda x:x['graph']['0.1']['exact_row_fraction']),
            graph_cell_accuracy_at_0_05=float(np.mean([
                x['graph']['0.05']['cell_accuracy'] for x in part])),
            graph_cell_accuracy_at_0_2=float(np.mean([
                x['graph']['0.2']['cell_accuracy'] for x in part])),
            D_output_shared_atom_ratios=np.mean([
                x['D_output_shared_atom_ratios'] for x in part],axis=0).tolist(),
            heldout_D_flip_response_error=float(np.mean([
                x['heldout_D_flip_response_error'] for x in part])),
            other_three_accuracy_after_D_axis_removal=float(np.mean([
                x['other_three_accuracy_after_D_axis_removal'] for x in part])))
    summary['random_rotation']=dict(
        raw_atom_alignment=model_seed_ci(random_rows,
            lambda x:x['mean_raw_atom_alignment']),
        graph_cell_accuracy_at_0_1=model_seed_ci(random_rows,
            lambda x:x['graph']['0.1']['cell_accuracy']),
        graph_exact_row_fraction_at_0_1=model_seed_ci(random_rows,
            lambda x:x['graph']['0.1']['exact_row_fraction']))
    # An oracle coding projection is an evaluation of whether the trained
    # function really contains the designed graph; it is not used by gauges.
    native_graph=[]
    cuts=np.load(HERE/'dense_weak_rule_reduction_cuts.npz')
    for seed in range(10):
        matrix=cuts[f's{seed}_c0.1_r4_uniform_weight_only']
        coefficients=matrix@SIGNS/16
        support=np.abs(coefficients)>=.1*np.max(np.abs(coefficients),axis=1,keepdims=True)
        projection=coefficients@SIGNS.T
        native_graph.append(dict(seed=seed,
            expected_graph_cell_accuracy=float(np.mean(support==EXPECTED)),
            relative_residual_outside_raw_bit_span=float(
                np.sum((matrix-projection)**2)/np.sum(matrix**2)),
            D_output_shared_atom_ratios=np.mean(np.abs(coefficients[[3,7]])/
                np.abs(coefficients[[3,7],3,None]),axis=0).tolist()))
    result=dict(description='Post-fit known-code audit of shared atom DAG recovery in ten dense-coded weak-rule attention models; all labels and expected graph are used only for evaluation, never gauge fitting.',
        direct_D_flip_edges_withheld_from_same_bit_prior=True,
        expected_membership=EXPECTED.tolist(),thresholds=THRESHOLDS,
        rows=rows,random_rows=random_rows,summary=summary,
        oracle_coding_projection=native_graph)
    (HERE/'dense_weak_rule_graph_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for prior,item in summary.items():
        print(prior,'raw',round(item['raw_atom_alignment']['mean'],4),
              'graph',round(item['graph_cell_accuracy_at_0_1']['mean'],4),
              'exact rows',round(item['graph_exact_row_fraction_at_0_1']['mean'],4))
    print('oracle graph',np.mean([x['expected_graph_cell_accuracy'] for x in native_graph]),
          'residual',np.mean([x['relative_residual_outside_raw_bit_span']
                              for x in native_graph]))


if __name__=='__main__':run()
