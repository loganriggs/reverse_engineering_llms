"""Exact-gauge legibility priors after faithful rank-four toy reductions."""

import json
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize

from analyze_three_count_language import logits
from dual_joint_function_canonical import HERE
from train_four_count_weak_rule import FourCountAttention, examples


MEASURES=('uniform_weight_only','empirical_usage',
          'half_uniform_empirical','output_balanced_weight_only')
ALL_PAIRS=np.asarray(tuple(combinations(range(8),2)))
NO_D_PAIRS=np.asarray(tuple(combinations(range(6),2)))
PLANES=tuple(combinations(range(4),2))
TRUTH=np.zeros((4,8))
TRUTH[np.arange(4),2*np.arange(4)]=-1
TRUTH[np.arange(4),2*np.arange(4)+1]=1
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
    pairs=ALL_PAIRS if prior=='all_pair_edits' else NO_D_PAIRS
    def objective(angles):
        d=rotation(angles)@dictionary
        if prior=='key_l1':return np.abs(d).sum()
        return np.mean(np.abs(d[:,pairs[:,1]]-d[:,pairs[:,0]]).sum(axis=0))
    rng=np.random.default_rng(seed)
    initial=[np.zeros(6)]+[rng.uniform(-np.pi,np.pi,6) for _ in range(7)]
    fitted=[minimize(objective,start,method='Powell',
             options=dict(maxiter=150,xtol=1e-5,ftol=1e-6)) for start in initial]
    selected=min(fitted,key=lambda x:x.fun)
    return rotation(selected.x),float(selected.fun),[
        float(x.fun) for x in fitted]


def alignment(dictionary):
    normalized=dictionary/np.linalg.norm(dictionary,axis=1,keepdims=True)
    correlation=np.abs(normalized@TRUTH.T)
    row,column=linear_sum_assignment(-correlation)
    by_rule=np.zeros(4)
    mapping={}
    for axis,rule in zip(row,column):
        by_rule[rule]=correlation[axis,rule]
        mapping[int(rule)]=int(axis)
    return by_rule,mapping


def removal(left,dictionary,mapping,tokens,target,bias):
    # These are deletions of functional rank-one components, not physical heads.
    cut=left@dictionary
    original=logits([cut.reshape(2,4,8).transpose(0,2,1)],tokens,bias)>0
    assert np.mean(np.all(original==target,axis=1))==1
    axis=mapping[3]
    removed=cut-np.outer(left[:,axis],dictionary[axis])
    prediction=logits([removed.reshape(2,4,8).transpose(0,2,1)],tokens,bias)>0
    return dict(D_accuracy=float(np.mean(prediction[:,3]==target[:,3])),
                other_three_joint_accuracy=float(np.mean(np.all(
                    prediction[:,:3]==target[:,:3],axis=1))),
                all_four_joint_accuracy=float(np.mean(np.all(prediction==target,axis=1))))


def run():
    torch.set_num_threads(1)
    saved=np.load(HERE/'four_count_weak_rule_reduction_cuts.npz')
    rows=[]
    for seed in range(10):
        test_t,target_t,attrs=examples(10000,70000+seed,False)
        tokens=test_t.numpy();target=target_t.numpy().astype(bool)
        # This is a fresh, fully competent panel; it is not used to fit axes.
        model=FourCountAttention().double()
        model.load_state_dict(torch.load(HERE/f'four_count_weak_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        bias=model.read.bias.detach().numpy()*np.array([1,1,1,.1])
        for measure in MEASURES:
            cut=saved[f's{seed}_c0.1_r4_{measure}']
            u,s,v=np.linalg.svd(cut,full_matrices=False)
            left=u[:,:4]*np.sqrt(s[:4])
            dictionary=np.sqrt(s[:4,None])*v[:4]
            assert np.max(np.abs(left@dictionary-cut))<1e-9
            bases={'svd':(np.eye(4),None,None)}
            for i,prior in enumerate(('key_l1','all_pair_edits','exclude_D_edits')):
                bases[prior]=fit_rotation(dictionary,prior,
                    20261031+seed*1000+MEASURES.index(measure)*100+i)
            rng=np.random.default_rng(20261101+seed*100+MEASURES.index(measure))
            random_rows=[]
            for _ in range(32):
                raw=rng.standard_normal((4,4))
                q,r=np.linalg.qr(raw)
                q=q*np.sign(np.diag(r))[None,:]
                d=q@dictionary
                l=left@q.T
                cos,mapping=alignment(d)
                random_rows.append(dict(mean_alignment=float(cos.mean()),
                    D_alignment=float(cos[3]),
                    D_removal=removal(l,d,mapping,tokens,target,bias)))
            for name,(q,objective,starts) in bases.items():
                d=q@dictionary
                l=left@q.T
                assert np.max(np.abs(l@d-cut))<1e-9
                cos,mapping=alignment(d)
                rows.append(dict(seed=seed,measure=measure,prior=name,
                    fit_objective=objective,fit_start_objectives=starts,
                    matched_variable_cosines=cos.tolist(),
                    mean_variable_alignment=float(cos.mean()),
                    D_alignment=float(cos[3]),
                    D_removal=removal(l,d,mapping,tokens,target,bias),
                    exact_function_max_abs=float(np.max(np.abs(l@d-cut))),
                    random_control=dict(mean_alignment=float(np.mean([
                        x['mean_alignment'] for x in random_rows])),
                        D_alignment=float(np.mean([x['D_alignment'] for x in random_rows])),
                        D_after_removal_accuracy=float(np.mean([
                        x['D_removal']['D_accuracy'] for x in random_rows])),
                        other_three_after_removal_accuracy=float(np.mean([
                        x['D_removal']['other_three_joint_accuracy']
                        for x in random_rows])))))
        print('seed',seed,'complete',flush=True)
    result=dict(description='Exact orthogonal gauge priors at fixed faithful rank-four functions from four reduction measures, ten trained weak-D models; D-token edits excluded from one fit prior; known variable labels used only for post-fit evaluation and component removal.',
        function='calibrated D-output scale .1',
        gauge_priors=('svd','key_l1','all_pair_edits','exclude_D_edits'),
        D_family_exclusion_key_ids=(6,7),
        output_labels_used_for_gauge_fit=False,
        physical_head_intervention=False,
        rows=rows)
    (HERE/'four_count_weak_rule_gauges.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in result['gauge_priors']:
        part=[r for r in rows if r['prior']==name]
        print(name,'mean align',round(np.mean([r['mean_variable_alignment']
                    for r in part]),4),'D align',round(np.mean([r['D_alignment']
                    for r in part]),4),'D removal',round(np.mean([
                    r['D_removal']['D_accuracy'] for r in part]),3),
                    'other three',round(np.mean([
                    r['D_removal']['other_three_joint_accuracy'] for r in part]),3))


if __name__=='__main__':run()
