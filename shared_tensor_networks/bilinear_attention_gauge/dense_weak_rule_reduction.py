"""Weight-only, usage, and output-balanced cuts on dense-coded weak rules."""

import json
from itertools import combinations

import numpy as np
import torch

from analyze_toy_language import rms
from dual_joint_function_canonical import HERE
from three_count_metric import weighted_fit
from train_dense_four_count_weak import (DenseFourCountAttention, FEATURES,
                                         NTOKENS, examples)


CALIBRATIONS=(1.,.1)
RANKS=(3,4)
MEASURES=('uniform_weight_only','empirical_usage',
          'half_uniform_empirical','output_balanced_weight_only')
ALL_PAIRS=np.asarray(tuple(combinations(range(NTOKENS),2)))
SAME_D_PAIRS=np.asarray(tuple((i,j) for i,j in combinations(range(NTOKENS),2)
                              if ((i>>3)&1)==((j>>3)&1)))
DIRECT_D_EDGES=np.asarray(tuple((i,i+8) for i in range(8)))


def exact_table(model):
    embedding=model.embed.weight.detach().numpy()
    query,key=embedding[16:18],embedding[:NTOKENS]
    total=np.zeros((2,NTOKENS,4))
    for head in range(4):
        sl=slice(head*4,(head+1)*4)
        def projected(name,inputs):
            return rms(inputs@getattr(model,name).weight.detach().numpy()[sl].T)
        first=projected('q1',query)@projected('k1',key).T/4
        second=projected('q2',query)@projected('k2',key).T/4
        value=key@model.v.weight.detach().numpy()[head]
        read=model.read.weight.detach().numpy()[:,head]
        total+=(first*second)[:,:,None]*value[None,:,None]*read[None,None,:]
    return total


def table_logits(table,bias,tokens):
    query=tokens[:,-1]-16
    return bias+table[query[:,None],tokens[:,:-1]].sum(axis=1)


def svd(matrix,rank):
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    return (u[:,:rank]*s[:rank])@v[:rank]


def balanced_svd(matrix,rank):
    by_output=matrix.reshape(2,4,NTOKENS)
    scale=np.linalg.norm(by_output,axis=(0,2))
    assert np.all(scale>0)
    row_scale=np.tile(1/scale,2)
    return svd(matrix*row_scale[:,None],rank)/row_scale[:,None]


def empirical_weights(tokens):
    queries=tokens[:,-1]-16
    keys=tokens[:,:-1]
    counts=np.zeros((2,NTOKENS))
    np.add.at(counts,(np.broadcast_to(queries[:,None],keys.shape),keys),1)
    assert np.all(counts>0)
    repeated=np.repeat(counts,4,axis=0)
    return repeated/repeated.sum()


def edit_errors(original,cut,pairs):
    target=original.reshape(2,4,NTOKENS)
    predicted=cut.reshape(2,4,NTOKENS)
    delta=target[:,:,pairs[:,1]]-target[:,:,pairs[:,0]]
    residual=predicted[:,:,pairs[:,1]]-predicted[:,:,pairs[:,0]]-delta
    per_output=np.sum(residual**2,axis=(0,2))/np.sum(delta**2,axis=(0,2))
    return dict(global_relative_mse=float(np.sum(residual**2)/np.sum(delta**2)),
                relative_mse_by_output=per_output.tolist(),
                max_output_relative_mse=float(per_output.max()))


def behavior(logits,reference,target,attrs):
    prediction=logits>0
    native=reference>0
    heldout=attrs['heldout']
    return dict(native_full_joint_accuracy=float(np.mean(np.all(native==target,axis=1))),
                native_rare_joint_accuracy=float(np.mean(np.all(
                    native[heldout]==target[heldout],axis=1))),
                full_joint_accuracy=float(np.mean(np.all(prediction==target,axis=1))),
                rare_joint_accuracy=float(np.mean(np.all(
                    prediction[heldout]==target[heldout],axis=1))),
                full_prediction_agreement=float(np.mean(np.all(prediction==native,axis=1))),
                rare_prediction_agreement=float(np.mean(np.all(
                    prediction[heldout]==native[heldout],axis=1))),
                D_prediction_agreement=float(np.mean(prediction[:,3]==native[:,3])),
                D_ground_truth_accuracy=float(np.mean(prediction[:,3]==target[:,3])))


def run():
    torch.set_num_threads(1)
    rows=[];checks=[];cuts={}
    for seed in range(10):
        model=DenseFourCountAttention().double()
        model.load_state_dict(torch.load(HERE/f'dense_four_count_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        table=exact_table(model)
        bias=model.read.bias.detach().numpy()
        test_t,target_t,attrs=examples(30000,50000+seed,False)
        tokens=test_t.numpy();target=target_t.numpy().astype(bool)
        direct=model(test_t).detach().numpy()
        exact=table_logits(table,bias,tokens)
        gate=float(np.max(np.abs(direct-exact)))
        assert gate<2e-10,gate
        fit_t,_,_=examples(100000,60000+seed,True)
        empirical=empirical_weights(fit_t.numpy())
        uniform=np.ones((8,NTOKENS))/(8*NTOKENS)
        for calibration in CALIBRATIONS:
            scale=np.array([1.,1.,1.,calibration])
            calibrated=table*scale[None,None,:]
            calibrated_bias=bias*scale
            reference=exact*scale
            original=calibrated.transpose(0,2,1).reshape(8,NTOKENS)
            checks.append(dict(seed=seed,calibration=calibration,
                exact_table_forward_max_abs=gate,
                native=behavior(reference,reference,target,attrs),
                median_D_margin=float(np.median(np.abs(reference[:,3])))))
            for rank in RANKS:
                choices={'uniform_weight_only':(svd(original,rank),None),
                         'output_balanced_weight_only':(balanced_svd(original,rank),None)}
                for name,measure in [('empirical_usage',empirical),
                                     ('half_uniform_empirical',.5*(uniform+empirical))]:
                    choices[name]=weighted_fit(original,measure,rank)
                for name,(cut,fit) in choices.items():
                    cuts[f's{seed}_c{calibration:g}_r{rank}_{name}']=cut
                    cut_table=cut.reshape(2,4,NTOKENS).transpose(0,2,1)
                    logits=table_logits(cut_table,calibrated_bias,tokens)
                    per_output=np.sum((cut-original).reshape(2,4,NTOKENS)**2,axis=(0,2))/np.sum(
                        original.reshape(2,4,NTOKENS)**2,axis=(0,2))
                    rows.append(dict(seed=seed,calibration=calibration,
                        rank=rank,measure=name,behavior=behavior(logits,reference,target,attrs),
                        broad_relative_table_mse=float(np.sum((cut-original)**2)/
                                                      np.sum(original**2)),
                        relative_table_mse_by_output=per_output.tolist(),
                        max_output_relative_table_mse=float(per_output.max()),
                        all_pair_edits=edit_errors(original,cut,ALL_PAIRS),
                        no_raw_D_bit_change_edits=edit_errors(original,cut,SAME_D_PAIRS),
                        direct_D_flip_edits=edit_errors(original,cut,DIRECT_D_EDGES),
                        fit=fit))
        print('seed',seed,'complete',flush=True)
    result=dict(description='Ten trained dense-coded four-rule attention models with a post-training weak D output; exact weight-derived 8x16 token function reduced with four label-free metrics, then audited on fresh examples and edit families.',
        feature_mix=FEATURES.tolist(),calibrations=CALIBRATIONS,ranks=RANKS,
        measures=MEASURES,all_pairs=len(ALL_PAIRS),same_D_raw_bit_pairs=len(SAME_D_PAIRS),
        direct_D_flip_edges=len(DIRECT_D_EDGES),
        model_training_uses_task_labels=True,reduction_fit_uses_task_labels=False,
        dense_code_is_nonorthogonal=True,rows=rows,model_checks=checks)
    (HERE/'dense_weak_rule_reduction.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(HERE/'dense_weak_rule_reduction_cuts.npz',**cuts)
    for calibration in CALIBRATIONS:
        for rank in RANKS:
            for measure in MEASURES:
                part=[x for x in rows if x['calibration']==calibration and
                      x['rank']==rank and x['measure']==measure]
                print(calibration,rank,measure,'D agreement',round(np.mean([
                    x['behavior']['D_prediction_agreement'] for x in part]),4),
                    'joint',round(np.mean([x['behavior']['full_prediction_agreement']
                                           for x in part]),4),
                    'global edit',round(np.mean([x['all_pair_edits']['global_relative_mse']
                                                for x in part]),4),
                    'D edge error',round(np.mean([x['direct_D_flip_edits']['global_relative_mse']
                                                 for x in part]),4))


if __name__=='__main__':run()
