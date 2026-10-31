"""Reduction-metric and answer-free rank gates for a weak competent output."""

import json
from itertools import combinations

import numpy as np
import torch
from scipy.special import ndtr

from analyze_three_count_language import head_tables, logits
from dual_joint_function_canonical import HERE
from three_count_metric import weighted_fit
from train_four_count_weak_rule import FourCountAttention, examples
from train_three_count_language import PAD


CALIBRATIONS=(1.,.1)
RANKS=(3,4)
ALL_PAIRS=np.asarray(tuple(combinations(range(8),2)))
NO_D_PAIRS=np.asarray(tuple(combinations(range(6),2)))


def svd(matrix,rank):
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    return (u[:,:rank]*s[:rank])@v[:rank]


def weights(tokens):
    query=tokens[:,-1]-8
    keys=tokens[:,:-1]
    valid=keys!=PAD
    counts=np.zeros((2,8))
    np.add.at(counts,(np.broadcast_to(query[:,None],keys.shape)[valid],
                      keys[valid]),1)
    assert np.all(counts>0)
    cells=np.repeat(counts,4,axis=0)
    return cells/cells.sum()


def balanced_svd(matrix,rank):
    by_output=matrix.reshape(2,4,8)
    scale=np.linalg.norm(by_output,axis=(0,2))
    assert np.all(scale>0)
    row_scale=np.tile(1/scale,2)
    return svd(matrix*row_scale[:,None],rank)/row_scale[:,None]


def edit_errors(original,cut,pairs):
    target=original.reshape(2,4,8)
    predicted=cut.reshape(2,4,8)
    delta=target[:,:,pairs[:,1]]-target[:,:,pairs[:,0]]
    residual=predicted[:,:,pairs[:,1]]-predicted[:,:,pairs[:,0]]-delta
    global_error=float(np.sum(residual**2)/np.sum(delta**2))
    per_output=np.sum(residual**2,axis=(0,2))/np.sum(delta**2,axis=(0,2))
    return dict(global_relative_mse=global_error,
                relative_mse_by_output=per_output.tolist(),
                max_output_relative_mse=float(per_output.max()))


def task_scores(table,bias,tokens,target,attrs):
    prediction=logits([table],tokens,bias)>0
    correct=(prediction==target).all(axis=1)
    heldout=attrs['heldout']
    return dict(full_joint_accuracy=float(correct.mean()),
                rare_heldout_joint_accuracy=float(correct[heldout].mean()),
                weak_D_accuracy=float(np.mean(prediction[:,3]==target[:,3])),
                weak_D_rare_heldout_accuracy=float(np.mean(
                    prediction[heldout,3]==target[heldout,3])))


def run():
    torch.set_num_threads(1)
    records=[];model_checks=[];cuts={}
    for seed in range(10):
        model=FourCountAttention().double()
        model.load_state_dict(torch.load(HERE/f'four_count_weak_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        table=sum(head_tables(model))
        bias=model.read.bias.detach().numpy()
        test_t,target_t,attrs=examples(30000,50000+seed,False)
        tokens=test_t.numpy();target=target_t.numpy().astype(bool)
        native_logits=model(test_t).detach().numpy()
        gate=float(np.max(np.abs(native_logits-logits([table],tokens,bias))))
        assert gate<1e-10
        assert task_scores(table,bias,tokens,target,attrs)['full_joint_accuracy']==1
        empirical=weights(examples(100000,60000+seed,True)[0].numpy())
        uniform=np.ones((8,8))/64
        for calibration in CALIBRATIONS:
            scale=np.array([1.,1.,1.,calibration])
            calibrated=table*scale[None,None,:]
            calibrated_bias=bias*scale
            matrix=calibrated.transpose(0,2,1).reshape(8,8)
            baseline=task_scores(calibrated,calibrated_bias,tokens,target,attrs)
            assert baseline['full_joint_accuracy']==1
            weak_margin=np.abs(native_logits[:,3]*calibration)
            signed_weak_margin=(target[:,3]*2-1)*native_logits[:,3]*calibration
            model_checks.append(dict(seed=seed,calibration=calibration,
                exact_table_forward_max_abs=gate,
                median_calibrated_D_logit_margin=float(np.median(weak_margin)),
                expected_D_accuracy_under_independent_gaussian_logit_noise={
                    str(sigma):float(np.mean(ndtr(signed_weak_margin/sigma)))
                    for sigma in (.5,1.)},
                native_task=baseline))
            for rank in RANKS:
                choices={'uniform_weight_only':(svd(matrix,rank),None),
                         'output_balanced_weight_only':(balanced_svd(matrix,rank),None)}
                for name,measure in [('empirical_usage',empirical),
                                     ('half_uniform_empirical',.5*(uniform+empirical))]:
                    choices[name]=weighted_fit(matrix,measure,rank)
                for name,(cut,fit) in choices.items():
                    cuts[f's{seed}_c{calibration:g}_r{rank}_{name}']=cut
                    cut_table=cut.reshape(2,4,8).transpose(0,2,1)
                    result=task_scores(cut_table,calibrated_bias,tokens,target,attrs)
                    per_output=np.sum((cut-matrix).reshape(2,4,8)**2,axis=(0,2))/np.sum(
                        matrix.reshape(2,4,8)**2,axis=(0,2))
                    records.append(dict(seed=seed,calibration=calibration,
                        rank=rank,measure=name,task=result,
                        broad_relative_table_mse=float(np.sum((cut-matrix)**2)/
                                                      np.sum(matrix**2)),
                        relative_table_mse_by_output=per_output.tolist(),
                        max_output_relative_table_mse=float(per_output.max()),
                        all_pair_edits=edit_errors(matrix,cut,ALL_PAIRS),
                        D_family_withheld_edits=edit_errors(matrix,cut,NO_D_PAIRS),
                        fit=fit))
        print('seed',seed,'complete',flush=True)
    result=dict(description='Ten trained four-rule bilinear attention models, D output post-training scaled by .1, rank-three/four exact-function reductions under uniform, usage, mixed, and output-balanced weight-only measures; compare global and output-relative edit gates with D token family withheld from one panel.',
        calibrations=CALIBRATIONS,ranks=RANKS,
        all_pair_count=len(ALL_PAIRS),D_family_withheld_pair_count=len(NO_D_PAIRS),
        model_training_uses_task_labels=True,
        reduction_and_rank_gates_use_task_labels=False,
        calibration_is_post_training_positive_output_scale=True,
        records=records,model_checks=model_checks)
    (HERE/'four_count_weak_rule_reduction.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(HERE/'four_count_weak_rule_reduction_cuts.npz',**cuts)
    for calibration in CALIBRATIONS:
        for rank in RANKS:
            for name in ('uniform_weight_only','empirical_usage',
                         'half_uniform_empirical','output_balanced_weight_only'):
                part=[r for r in records if r['calibration']==calibration and
                      r['rank']==rank and r['measure']==name]
                print(calibration,rank,name,'D',round(np.mean([
                    r['task']['weak_D_accuracy'] for r in part]),3),
                    'all',round(np.mean([r['task']['full_joint_accuracy'] for r in part]),3),
                    'global edit',round(np.mean([r['all_pair_edits']['global_relative_mse']
                                                for r in part]),4),
                    'max output edit',round(np.mean([
                        r['all_pair_edits']['max_output_relative_mse'] for r in part]),4))


if __name__=='__main__':run()
