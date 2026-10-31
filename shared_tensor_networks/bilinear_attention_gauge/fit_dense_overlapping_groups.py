"""Learn an exact nonorthogonal shared dictionary with static group links.

Fits a four-atom dictionary B and eight query/output group coefficients C
to a fixed rank-four function W=C B. The invertible parameterization keeps
function fidelity identical while sweeping a row-link sparsity prior against
an intervention-response sparsity prior. This is a static overlapping-group
model, not a sample-dependent gated autoencoder.
"""

import argparse
import json

import numpy as np
import torch
from scipy.optimize import minimize

from analyze_dense_weak_rule_graph import graph_scores
from dense_weak_rule_reduction import ALL_PAIRS
from dense_weak_new_task_transfer import candidates, raw_targets, select_and_score
from dual_joint_function_canonical import HERE


DEFAULT_RATIOS=(0.,5.,10.,20.,40.,80.)


def factors(matrix,transform,right):
    raw=transform@right
    norms=np.linalg.norm(raw,axis=1)
    dictionary=raw/norms[:,None]
    left=(matrix@right.T)@np.linalg.inv(transform)@np.diag(norms)
    return left,dictionary


def objective_and_gradient(z,right_t,baseline_t,pair_left,pair_right,ratio):
    try:
        raw=torch.tensor(z.reshape(4,4),dtype=torch.float64,requires_grad=True)
        normalized=raw/torch.linalg.vector_norm(raw,dim=1,keepdim=True)
        b=normalized@right_t
        c=torch.linalg.solve(normalized.T,baseline_t.T).T
        link=(torch.sqrt(c*c+1e-8)/torch.linalg.vector_norm(
            c,dim=1,keepdim=True)).sum()
        delta=b[:,pair_right]-b[:,pair_left]
        edit=torch.sqrt(delta*delta+1e-8).sum(dim=0).mean()
        singular=torch.linalg.svdvals(normalized)
        condition=singular[0]/singular[-1]
        value=link+ratio*edit+10*torch.clamp(condition-4,min=0)**2
        grad=torch.autograd.grad(value,raw)[0]
        return float(value.detach()),grad.detach().numpy().ravel()
    except (RuntimeError,np.linalg.LinAlgError):
        return 1e12,np.zeros_like(z)


def fit(matrix,ratio,seed,nstarts=5,warm_dictionary=None):
    u,s,right=np.linalg.svd(matrix,full_matrices=False)
    right=right[:4]
    baseline=matrix@right.T
    right_t=torch.tensor(right,dtype=torch.float64)
    baseline_t=torch.tensor(baseline,dtype=torch.float64)
    pair_left=torch.tensor(ALL_PAIRS[:,0],dtype=torch.long)
    pair_right=torch.tensor(ALL_PAIRS[:,1],dtype=torch.long)
    def objective(z):
        return objective_and_gradient(z,right_t,baseline_t,
                                      pair_left,pair_right,ratio)
    rng=np.random.default_rng(seed)
    starts=[np.eye(4),matrix[:4]@right.T]
    if warm_dictionary is not None:
        starts.append(warm_dictionary@right.T)
    for _ in range(nstarts-len(starts)):
        raw=rng.normal(size=(4,4))
        q,r=np.linalg.qr(raw)
        starts.append(q*np.sign(np.diag(r))[None,:])
    results=[]
    for start in starts:
        result=minimize(objective,start.ravel(),jac=True,method='L-BFGS-B',
                        options=dict(maxiter=600,ftol=1e-11,gtol=1e-8))
        results.append(result)
    chosen=min(results,key=lambda result:result.fun)
    t=chosen.x.reshape(4,4)
    left,dictionary=factors(matrix,t,right)
    error=float(np.max(np.abs(left@dictionary-matrix)))
    assert error<1e-8,error
    c=left/np.linalg.norm(left,axis=1,keepdims=True)
    link=float(np.abs(c).sum())
    edit=float(np.mean(np.abs(dictionary[:,ALL_PAIRS[:,1]]-
                              dictionary[:,ALL_PAIRS[:,0]]).sum(axis=0)))
    return left,dictionary,dict(objective=float(chosen.fun),link_l1=link,
        edit_l1=edit,condition=float(np.linalg.cond(dictionary)),
        max_abs_reconstruction_error=error,starts=[float(x.fun) for x in results],
        optimizer_success=bool(chosen.success),optimizer_message=str(chosen.message))


def check():
    from scipy.optimize import check_grad
    rng=np.random.default_rng(20261112)
    matrix=rng.normal(size=(8,4))@rng.normal(size=(4,16))
    _,_,vh=np.linalg.svd(matrix,full_matrices=False)
    right=vh[:4]
    right_t=torch.tensor(right,dtype=torch.float64)
    baseline_t=torch.tensor(matrix@right.T,dtype=torch.float64)
    pair_left=torch.tensor(ALL_PAIRS[:,0],dtype=torch.long)
    pair_right=torch.tensor(ALL_PAIRS[:,1],dtype=torch.long)
    near_dependent=np.eye(4)
    near_dependent[3]=[1.,0.,0.,.2]
    for transform in (np.eye(4)+.1*rng.normal(size=(4,4)),
                      near_dependent):
        z=transform.ravel()
        fun=lambda v:objective_and_gradient(v,right_t,baseline_t,
                                           pair_left,pair_right,80.)[0]
        jac=lambda v:objective_and_gradient(v,right_t,baseline_t,
                                           pair_left,pair_right,80.)[1]
        error=check_grad(fun,jac,z,epsilon=1e-8)
        assert error/max(1.,np.linalg.norm(jac(z)))<1e-6,error
        left,dictionary=factors(matrix,transform,right)
        assert np.max(np.abs(left@dictionary-matrix))<1e-10
    print('Group objective gradient and exact-factorization checks passed.')


def transfer(dictionary,seed):
    rng=np.random.default_rng(seed)
    test=rng.integers(0,16,size=(10000,7))
    truth=raw_targets(test)
    test_candidates=candidates(dictionary,test)
    agreements=[np.mean(c[:,:,None]==y[:,None,:],axis=0)
                for c,y in zip(test_candidates,truth)]
    rows=[]
    for episode in range(10):
        train=rng.integers(0,16,size=(64,7))
        train_truth=raw_targets(train)
        train_candidates=candidates(dictionary,train)
        rows.append([select_and_score(c,a,y) for c,a,y in zip(
            train_candidates,agreements,train_truth)])
    return dict(primitive_accuracy=float(np.mean([
        item['test_accuracy'] for episode in rows for item in episode[0]])),
        xor_accuracy=float(np.mean([
        item['test_accuracy'] for episode in rows for item in episode[1]])))


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--first-seed',type=int,default=0)
    parser.add_argument('--seeds',type=int,default=10)
    parser.add_argument('--starts',type=int,default=5)
    parser.add_argument('--ratios',default=','.join(map(str,DEFAULT_RATIOS)))
    parser.add_argument('--warm-edit-gauge',action='store_true')
    parser.add_argument('--output',default='dense_overlapping_group_sweep.json')
    args=parser.parse_args()
    if args.check:
        check()
        return
    ratios=[float(v) for v in args.ratios.split(',')]
    torch.set_num_threads(1)
    cuts=np.load(HERE/'dense_weak_rule_reduction_cuts.npz')
    gauge_factors=(np.load(HERE/'dense_weak_rule_gauge_factors.npz')
                   if args.warm_edit_gauge else None)
    rows=[];factor_store={}
    for seed in range(args.first_seed,args.seeds):
        for measure in ('uniform_weight_only','empirical_usage'):
            matrix=cuts[f's{seed}_c0.1_r4_{measure}']
            for i,ratio in enumerate(ratios):
                warm=(gauge_factors[f's{seed}_{measure}_all_pair_edits_dictionary']
                      if gauge_factors is not None else None)
                left,dictionary,fit_info=fit(matrix,ratio,
                    20261111+seed*10000+int(measure=='empirical_usage')*1000+i,
                    args.starts,warm)
                graph=graph_scores(left,dictionary)
                scores=transfer(dictionary,20261120+seed)
                support=np.abs(left)>=.1*np.max(np.abs(left),axis=1,keepdims=True)
                rows.append(dict(seed=seed,measure=measure,ratio=ratio,
                    fit=fit_info,active_group_links=int(np.sum(support)),
                    raw_atom_alignment=graph['mean_raw_atom_alignment'],
                    graph_cell_accuracy=graph['graph']['0.1']['cell_accuracy'],
                    transfer=scores))
                prefix=f's{seed}_{measure}_r{ratio:g}'
                factor_store[prefix+'_dictionary']=dictionary
                factor_store[prefix+'_left']=left
        print('seed',seed,'complete',flush=True)
    summary={}
    for measure in ('uniform_weight_only','empirical_usage'):
        summary[measure]={}
        for ratio in ratios:
            part=[r for r in rows if r['measure']==measure and r['ratio']==ratio]
            summary[measure][str(ratio)]={key:float(np.mean([
                r[key] for r in part])) for key in ('active_group_links',
                  'raw_atom_alignment','graph_cell_accuracy')}
            for key in ('primitive_accuracy','xor_accuracy'):
                summary[measure][str(ratio)][key]=float(np.mean([
                    r['transfer'][key] for r in part]))
    result=dict(description='Exact nonorthogonal four-atom static overlapping-group factorization. Row-normalized group-link L1 + ratio times atom edit-response L1; 10%-relative group support. No task labels in fit; raw labels used only for evaluation.',
        first_seed=args.first_seed,n_seeds=args.seeds-args.first_seed,
        nstarts=args.starts,warm_edit_gauge=args.warm_edit_gauge,
        ratios=ratios,rows=rows,summary=summary)
    (HERE/args.output).write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(HERE/args.output.replace('.json','_factors.npz'),**factor_store)
    for measure,levels in summary.items():
        for ratio,stats in levels.items():
            print(measure,ratio,stats)


if __name__=='__main__':run()
