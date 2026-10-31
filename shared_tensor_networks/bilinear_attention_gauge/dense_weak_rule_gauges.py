"""Fixed-function gauge priors under correlated dense token coding."""

import json
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize

from dual_joint_function_canonical import HERE
from dense_weak_rule_reduction import (ALL_PAIRS, SAME_D_PAIRS,
                                       DIRECT_D_EDGES, table_logits)
from train_dense_four_count_weak import (DenseFourCountAttention, FEATURES,
                                         NTOKENS, examples)


MEASURES=('uniform_weight_only','empirical_usage',
          'half_uniform_empirical','output_balanced_weight_only')
PRIORS=('svd','key_l1','all_pair_edits','same_raw_D_bit_edits')
PLANES=tuple(combinations(range(4),2))
TRUTH=FEATURES.T.copy()
TRUTH/=np.linalg.norm(TRUTH,axis=1,keepdims=True)


def rotation(angles):
    result=np.eye(4)
    for angle,(i,j) in zip(angles,PLANES):
        c,s=np.cos(angle),np.sin(angle)
        factor=np.eye(4)
        factor[i,i]=factor[j,j]=c
        factor[i,j]=-s
        factor[j,i]=s
        result=factor@result
    return result


def fit_rotation(dictionary,prior,seed):
    pairs=ALL_PAIRS if prior=='all_pair_edits' else SAME_D_PAIRS
    def objective(angles):
        d=rotation(angles)@dictionary
        if prior=='key_l1':return np.abs(d).sum()
        return np.mean(np.abs(d[:,pairs[:,1]]-d[:,pairs[:,0]]).sum(axis=0))
    rng=np.random.default_rng(seed)
    initial=[np.zeros(6)]+[rng.uniform(-np.pi,np.pi,6) for _ in range(7)]
    fits=[minimize(objective,start,method='Powell',
          options=dict(maxiter=150,xtol=1e-5,ftol=1e-6)) for start in initial]
    selected=min(fits,key=lambda x:x.fun)
    return rotation(selected.x),float(selected.fun),[
        float(x.fun) for x in fits]


def alignment(dictionary):
    normalized=dictionary/np.linalg.norm(dictionary,axis=1,keepdims=True)
    correlation=np.abs(normalized@TRUTH.T)
    row,column=linear_sum_assignment(-correlation)
    scores=np.zeros(4);mapping={}
    for axis,rule in zip(row,column):
        scores[rule]=correlation[axis,rule]
        mapping[int(rule)]=int(axis)
    return scores,mapping


def component_checks(left,dictionary,mapping,tokens,target,bias):
    cut=left@dictionary
    baseline=table_logits(cut.reshape(2,4,NTOKENS).transpose(0,2,1),bias,tokens)>0
    axis=mapping[3]
    component=np.outer(left[:,axis],dictionary[axis])
    removed=cut-component
    prediction=table_logits(removed.reshape(2,4,NTOKENS).transpose(0,2,1),
                            bias,tokens)>0
    complete=cut.reshape(2,4,NTOKENS)
    selected=component.reshape(2,4,NTOKENS)
    pairs=DIRECT_D_EDGES
    full_delta=complete[:,:,pairs[:,1]]-complete[:,:,pairs[:,0]]
    component_delta=selected[:,:,pairs[:,1]]-selected[:,:,pairs[:,0]]
    return dict(
        baseline_joint_accuracy=float(np.mean(np.all(baseline==target,axis=1))),
        baseline_D_accuracy=float(np.mean(baseline[:,3]==target[:,3])),
        D_accuracy_after_removal=float(np.mean(prediction[:,3]==target[:,3])),
        D_agreement_with_baseline_after_removal=float(np.mean(prediction[:,3]==baseline[:,3])),
        other_three_joint_accuracy_after_removal=float(np.mean(np.all(
            prediction[:,:3]==target[:,:3],axis=1))),
        other_three_joint_agreement_with_baseline_after_removal=float(np.mean(np.all(
            prediction[:,:3]==baseline[:,:3],axis=1))),
        D_flip_output_response_relative_error=float(np.sum((
            component_delta[:,3]-full_delta[:,3])**2)/np.sum(full_delta[:,3]**2)),
        D_flip_offtarget_energy_fraction=float(np.sum(component_delta[:,:3]**2)/
            max(np.sum(component_delta**2),1e-30)))


def run():
    torch.set_num_threads(1)
    saved=np.load(HERE/'dense_weak_rule_reduction_cuts.npz')
    rows=[];factor_store={}
    for seed in range(10):
        test_t,target_t,attrs=examples(10000,70000+seed,False)
        tokens=test_t.numpy();target=target_t.numpy().astype(bool)
        model=DenseFourCountAttention().double()
        model.load_state_dict(torch.load(HERE/f'dense_four_count_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        bias=model.read.bias.detach().numpy()*np.array([1.,1.,1.,.1])
        for measure in MEASURES:
            cut=saved[f's{seed}_c0.1_r4_{measure}']
            u,s,v=np.linalg.svd(cut,full_matrices=False)
            left=u[:,:4]*np.sqrt(s[:4])
            dictionary=np.sqrt(s[:4,None])*v[:4]
            assert np.max(np.abs(left@dictionary-cut))<1e-9
            bases={'svd':(np.eye(4),None,None)}
            for i,prior in enumerate(PRIORS[1:]):
                bases[prior]=fit_rotation(dictionary,prior,
                    20261031+seed*1000+MEASURES.index(measure)*100+i)
            rng=np.random.default_rng(20261101+seed*100+MEASURES.index(measure))
            random=[]
            for _ in range(32):
                raw=rng.standard_normal((4,4))
                q,r=np.linalg.qr(raw)
                q=q*np.sign(np.diag(r))[None,:]
                d=q@dictionary;l=left@q.T
                cos,mapping=alignment(d)
                random.append(dict(mean_alignment=float(cos.mean()),
                    D_alignment=float(cos[3]),
                    checks=component_checks(l,d,mapping,tokens,target,bias)))
            for prior,(q,objective,starts) in bases.items():
                d=q@dictionary;l=left@q.T
                factor_store[f's{seed}_{measure}_{prior}_dictionary']=d
                factor_store[f's{seed}_{measure}_{prior}_left']=l
                exact=float(np.max(np.abs(l@d-cut)))
                assert exact<1e-9
                cos,mapping=alignment(d)
                rows.append(dict(seed=seed,measure=measure,prior=prior,
                    fit_objective=objective,fit_start_objectives=starts,
                    matched_variable_cosines=cos.tolist(),
                    mean_variable_alignment=float(cos.mean()),
                    D_alignment=float(cos[3]),
                    component=component_checks(l,d,mapping,tokens,target,bias),
                    exact_function_max_abs=exact,
                    random_control=dict(
                        mean_variable_alignment=float(np.mean([
                            x['mean_alignment'] for x in random])),
                        D_alignment=float(np.mean([x['D_alignment'] for x in random])),
                        D_accuracy_after_removal=float(np.mean([
                            x['checks']['D_accuracy_after_removal'] for x in random])),
                        other_three_joint_accuracy_after_removal=float(np.mean([
                            x['checks']['other_three_joint_accuracy_after_removal']
                            for x in random])),
                        D_flip_output_response_relative_error=float(np.mean([
                            x['checks']['D_flip_output_response_relative_error']
                            for x in random])))))
        print('seed',seed,'complete',flush=True)
    result=dict(description='Exact orthogonal gauges of rank-four dense-coded weak-rule functions under four reduction metrics; same-raw-D-bit edit prior withholds all changes of the raw D bit, while full weight-derived cuts still include D.',
        gauge_priors=PRIORS,measures=MEASURES,
        direct_D_flip_edges=DIRECT_D_EDGES.tolist(),
        gauge_fit_uses_task_labels=False,physical_head_intervention=False,
        rows=rows)
    (HERE/'dense_weak_rule_gauges.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(HERE/'dense_weak_rule_gauge_factors.npz',**factor_store)
    for prior in PRIORS:
        part=[x for x in rows if x['prior']==prior]
        print(prior,'align',round(np.mean([x['mean_variable_alignment'] for x in part]),3),
              'D',round(np.mean([x['D_alignment'] for x in part]),3),
              'D-removal',round(np.mean([x['component']['D_accuracy_after_removal']
                                         for x in part]),3),
              'other3',round(np.mean([
                  x['component']['other_three_joint_accuracy_after_removal']
                  for x in part]),3),
              'heldout D edit error',round(np.mean([
                  x['component']['D_flip_output_response_relative_error']
                  for x in part]),3))


if __name__=='__main__':run()
