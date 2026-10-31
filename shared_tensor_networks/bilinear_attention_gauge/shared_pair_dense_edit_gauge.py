"""Fit edit-sparse exact gauges on trained dense-token shared-product models."""

import json
from itertools import combinations

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from dual_joint_function_canonical import HERE
from shared_pair_gauge import monomial_strengths,profile
from shared_pair_graph_prune import monomial_features
from three_count_task_fit import count_features
from train_shared_pair_count import SharedPairCountModel,targets
from train_shared_pair_dense import TOKEN_SIGNS,examples_dense

SEEDS=range(10)
REDUCTIONS=('uniform_rank3','empirical_rank3')
PAIRS={
    'all_pairs':tuple(combinations(range(8),2)),
    'hamming1':tuple((i,j) for i,j in combinations(range(8),2)
                     if (i^j).bit_count()==1),
}
TRUTH=TOKEN_SIGNS.T/np.linalg.norm(TOKEN_SIGNS.T,axis=1,keepdims=True)


def fit_edit(base,pairs,seed):
    edges=np.asarray(pairs,dtype=int)
    changes=base[:,edges[:,1]]-base[:,edges[:,0]]
    return fit_changes(base,changes,seed)


def fit_changes(base,changes,seed):
    assert changes.ndim==2 and changes.shape[0]==3 and changes.shape[1]>0
    rng=np.random.default_rng(seed)
    starts=np.vstack([np.zeros(3),rng.uniform(-np.pi,np.pi,(128,3))])
    def score(params):
        rotation=Rotation.from_euler('xyz',params).as_matrix()
        return float(np.mean(np.sum(np.abs(rotation@changes),axis=0)))
    values=np.asarray([score(start) for start in starts])
    best=(float('inf'),None)
    for i in np.argsort(values)[:8]:
        result=minimize(score,starts[i],method='Powell',
                        options={'maxiter':250,'xtol':1e-7,'ftol':1e-9})
        if result.fun<best[0]:best=(float(result.fun),result.x)
    return Rotation.from_euler('xyz',best[1]).as_matrix()@base,best[0]


def task_score(matrix,q,bias,pb,dictionary,tokens,truth,attrs):
    counts=count_features(tokens).astype(np.float64)
    query=(tokens[:,-1]-8).astype(int)
    bit_logits=bias+np.einsum('nk,nok->no',counts,matrix.reshape(2,3,8)[query])
    parity=np.einsum('ni,pij,nj->np',bit_logits,q,bit_logits)+pb
    t,labels=monomial_strengths(matrix,q,dictionary)
    energy=np.sum(t*t,axis=(0,1))
    keep=np.argsort(energy)[-3:]
    omit=np.array([i for i in range(6) if i not in keep])
    z=counts@dictionary.T
    features=monomial_features(z)
    removed=np.einsum('npk,nk->np',t[:,query,:].transpose(1,0,2)[:,:,omit],
                      features[:,omit])
    pruned=np.c_[bit_logits,parity-removed]
    masks={'all':np.ones(len(tokens),dtype=bool),'rare_heldout':attrs['heldout']}
    return dict(kept_nodes=[labels[i] for i in keep],
        removed_coefficient_energy_fraction=float(energy[omit].sum()/energy.sum()),
        scores={name:dict(n=int(mask.sum()),
            five_output_accuracy=float(np.mean(np.all((pruned[mask]>0)==truth[mask],axis=1))))
            for name,mask in masks.items()})


def run():
    torch.set_num_threads(1)
    analysis=json.loads((HERE/'shared_pair_dense_analysis.json').read_text())
    existing=json.loads((HERE/'shared_pair_dense_gauge.json').read_text())
    rows=[]
    for source in analysis['rows']:
        seed=source['seed'];q=np.asarray(source['quadratic_effective'])
        model=SharedPairCountModel().double()
        model.load_state_dict(torch.load(HERE/f'shared_pair_dense_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        bias=model.backbone.read.bias.detach().numpy()
        pb=model.parity_bias.detach().numpy()
        tok,bits,attrs=examples_dense(30000,530000+seed,False)
        tok=tok.numpy();truth=targets(bits).numpy().astype(bool)
        for reduction in REDUCTIONS:
            matrix=np.asarray(source['matrices'][reduction])
            _,s,v=np.linalg.svd(matrix,full_matrices=False)
            assert s[3]<1e-7
            base=v[:3]
            saved=next(r for r in existing['rows'] if r['seed']==seed and
                       r['reduction']==reduction)['gauges']
            choices={name:np.asarray(saved[name]['dictionary'])
                     for name in ('svd','key_l1','joint_nullspace','cell_one_only')}
            fit_values={}
            for kind,pairs in PAIRS.items():
                choices[kind],fit_values[kind]=fit_edit(base,pairs,
                    20261104+seed*100+(0 if kind=='all_pairs' else 1))
            outcomes={}
            for name,d in choices.items():
                assert np.max(np.abs(d.T@d-base.T@base))<1e-8
                assert (np.linalg.norm(matrix@d.T@d-matrix)/
                        np.linalg.norm(matrix))<1e-8
                result=profile(matrix,q,d,TRUTH)
                result['dictionary']=d.tolist()
                result['task_after_three_node_prune']=task_score(
                    matrix,q,bias,pb,d,tok,truth,attrs)
                result['all_pair_edit_l1']=float(np.mean([
                    np.sum(np.abs(d[:,j]-d[:,i])) for i,j in PAIRS['all_pairs']]))
                result['hamming1_edit_l1']=float(np.mean([
                    np.sum(np.abs(d[:,j]-d[:,i])) for i,j in PAIRS['hamming1']]))
                outcomes[name]=result
            rows.append(dict(seed=seed,reduction=reduction,
                             fit_scores=fit_values,gauges=outcomes))
        print('seed',seed,flush=True)
    result=dict(description='Exact rank-three learned-function gauges optimized for sparse token-swap responses; no task labels in fitting; all-pair swaps unlabeled, Hamming-one swaps use known input code',
        all_pair_count=len(PAIRS['all_pairs']),hamming1_pair_count=len(PAIRS['hamming1']),
        seeds=list(SEEDS),rows=rows)
    (HERE/'shared_pair_dense_edit_gauge.json').write_text(json.dumps(result,indent=2)+'\n')
    for reduction in REDUCTIONS:
        for name in ('svd','key_l1','joint_nullspace','cell_one_only','all_pairs','hamming1'):
            subset=[r['gauges'][name] for r in rows if r['reduction']==reduction]
            print(reduction,name,
                'alignment',round(np.mean([x['matched_variable_cosine'] for x in subset]),3),
                'rare',round(np.mean([x['task_after_three_node_prune']['scores']['rare_heldout']['five_output_accuracy'] for x in subset]),3),
                'cell',round(np.mean([x['mean_cell_top_one_monomial_energy_fraction'] for x in subset]),3),flush=True)


if __name__=='__main__':run()
