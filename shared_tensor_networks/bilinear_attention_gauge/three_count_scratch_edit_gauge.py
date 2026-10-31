"""Cross-task transfer of unlabeled all-pair edit gauges to learned parity."""

import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from shared_pair_dense_edit_gauge import PAIRS,fit_changes,fit_edit
from shared_pair_graph_prune import monomial_features
from three_count_metric import pair_weights,weighted_fit
from three_count_nonlinear_gauge import TRUTH
from three_count_task_fit import count_features
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples

SEEDS=range(10)
LABELS=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))


def monomial_strengths(matrix,q,d):
    c=(matrix@d.T).reshape(2,3,3)
    h=np.einsum('qic,ij,qjd->qcd',c,q,c)
    return np.stack([h[:,i,j]*(np.sqrt(2) if i!=j else 1)
                     for i,j in LABELS],axis=-1)


def alignment(d):
    cosine=np.abs(d@TRUTH.T)
    row,col=linear_sum_assignment(-cosine)
    return float(np.mean(cosine[row,col]))


def task_score(matrix,q,bias,pb,d,tokens,target,attrs):
    counts=count_features(tokens).astype(np.float64)
    query=(tokens[:,-1]-8).astype(int)
    bit_logits=bias+np.einsum('nk,nok->no',counts,matrix.reshape(2,3,8)[query])
    parity=np.einsum('ni,ij,nj->n',bit_logits,q,bit_logits)+pb
    t=monomial_strengths(matrix,q,d)
    energy=np.sum(t*t,axis=0)
    keep=np.argsort(energy)[-2:]
    omit=np.array([i for i in range(6) if i not in keep])
    z=counts@d.T
    features=monomial_features(z)
    signal=bit_logits-bias
    direct_quadratic=np.einsum('ni,ij,nj->n',signal,q,signal)
    represented=np.einsum('nk,nk->n',t[query],features)
    assert np.max(np.abs(represented[:256]-direct_quadratic[:256]))<1e-8
    removed=np.einsum('nk,nk->n',t[query][:,omit],features[:,omit])
    pruned=np.c_[bit_logits,parity-removed]
    truth=np.c_[target,np.logical_xor(target[:,0],target[:,1])]
    unpruned=np.c_[bit_logits,parity]
    groups={'all':np.ones(len(tokens),dtype=bool),'rare_heldout':attrs['heldout']}
    return dict(kept_nodes=[LABELS[i] for i in keep],
        top_two_monomial_energy_fraction=float(energy[keep].sum()/energy.sum()),
        removed_quadratic_energy_fraction=float(energy[omit].sum()/energy.sum()),
        unpruned_accuracy=float(np.mean(np.all((unpruned>0)==truth,axis=1))),
        unpruned_scores={name:dict(n=int(mask.sum()),
            all_four_accuracy=float(np.mean(np.all((unpruned[mask]>0)==truth[mask],axis=1))))
            for name,mask in groups.items()},
        scores={name:dict(n=int(mask.sum()),
            all_four_accuracy=float(np.mean(np.all((pruned[mask]>0)==truth[mask],axis=1))),
            parity_accuracy=float(np.mean((pruned[mask,3]>0)==truth[mask,3])))
            for name,mask in groups.items()})


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in SEEDS:
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        matrix=flatten(sum(head_tables(model.backbone)))
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        uniform=(u[:,:3]*s[:3])@v[:3]
        train,_,_=examples(20000,110000+seed,True)
        empirical,_=weighted_fit(matrix,pair_weights(train.numpy()),3)
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        bias=model.backbone.read.bias.detach().numpy()
        pb=float(model.parity_bias.detach())
        tokens,bits,attrs=examples(30000,130000+seed,False)
        tokens=tokens.numpy();target=bits.numpy().astype(bool)
        for reduction,cut in (('uniform_rank3',uniform),('empirical_rank3',empirical)):
            _,cs,cv=np.linalg.svd(cut,full_matrices=False)
            assert cs[3]<1e-8
            base=cv[:3]
            choices={'svd':base}
            choices['key_l1'],_=fit_changes(base,base,20261108+100*seed)
            choices['all_pairs'],_=fit_edit(base,PAIRS['all_pairs'],20261109+100*seed)
            results={}
            for name,d in choices.items():
                assert np.linalg.norm(cut@d.T@d-cut)/np.linalg.norm(cut)<1e-8
                task=task_score(cut,q,bias,pb,d,tokens,target,attrs)
                results[name]=dict(dictionary=d.tolist(),
                    matched_variable_cosine=alignment(d),
                    key_l1=float(np.abs(d).sum()),
                    all_pair_edit_l1=float(np.mean([np.sum(np.abs(d[:,j]-d[:,i]))
                        for i,j in PAIRS['all_pairs']])),
                    task_after_two_node_prune=task)
            rows.append(dict(seed=seed,reduction=reduction,
                singular_values=s.tolist(),gauges=results))
        print('seed',seed,flush=True)
    result=dict(description='Ten from-scratch trained single-parity language models; uniform weight-only and unlabeled training-frequency rank-three functions, frozen during gauge fitting; select two quadratic nodes by coefficient energy without task labels',
        seeds=list(SEEDS),model_training_uses_task_labels=True,
        gauge_fit_uses_task_labels=False,node_selection_uses_task_labels=False,
        input_swap_prior='all 28 unordered key token pairs, exact uniform measure',
        rows=rows)
    (HERE/'three_count_scratch_edit_gauge.json').write_text(json.dumps(result,indent=2)+'\n')
    for reduction in ('uniform_rank3','empirical_rank3'):
        for name in ('svd','key_l1','all_pairs'):
            x=[r['gauges'][name] for r in rows if r['reduction']==reduction]
            print(reduction,name,'cosine',np.mean([v['matched_variable_cosine'] for v in x]),
                'top2',np.mean([v['task_after_two_node_prune']['top_two_monomial_energy_fraction'] for v in x]),
                'all',np.mean([v['task_after_two_node_prune']['scores']['all']['all_four_accuracy'] for v in x]),
                'rare',np.mean([v['task_after_two_node_prune']['scores']['rare_heldout']['all_four_accuracy'] for v in x]),flush=True)


if __name__=='__main__':run()
