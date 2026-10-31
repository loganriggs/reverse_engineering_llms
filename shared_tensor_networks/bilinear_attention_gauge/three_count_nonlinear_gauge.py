"""Exact-gauge key versus context-edit sparsity in the nonlinear count toy."""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from dual_joint_function_canonical import HERE
from three_count_nonlinear_metric import count_features,examples,parity


SOURCE='three_count_nonlinear_metric_s0-1-2-3-4-5-6-7-8-9.json'
REDUCTIONS=('uniform_table','empirical_table','context_nonlinear_weight_0.1')
FAMILIES=('A','B','C')
TRUTH=np.zeros((3,8))
for family in range(3):
    TRUTH[family,2*family:2*family+2]=[-1/np.sqrt(2),1/np.sqrt(2)]


def rotation(angle):
    c=np.cos(angle);s=np.sin(angle)
    return np.array([[c,-s],[s,c]])


def edit_panel(n,seed,holdout,rate=.75):
    tokens,_,attrs=examples(n,seed,holdout,rate)
    tokens=tokens.numpy()
    counts=count_features(tokens).astype(np.float64)
    rng=np.random.default_rng(seed+123456)
    edited=counts.copy()
    for i,row in enumerate(tokens[:,:-1]):
        keys=row[row<8]
        old=int(keys[rng.integers(len(keys))])
        new=int(rng.integers(7))
        if new>=old:new+=1
        edited[i,old]-=1;edited[i,new]+=1
    return dict(counts=counts,edited=edited,
                query=(tokens[:,-1]-8).astype(np.int64),attrs=attrs)


def exact_components(matrix,bias,panel,dictionary):
    coefficients=(matrix@dictionary.T).reshape(2,3,2)
    def one(counts):
        coordinate=counts@dictionary.T
        components=np.einsum('nc,noc->nco',coordinate,
                              coefficients[panel['query']])
        logits=bias+components.sum(axis=1)
        return logits,components
    before,e_before=one(panel['counts'])
    after,e_after=one(panel['edited'])
    total=parity(after)-parity(before)
    effects=np.stack([total-(parity(after-e_after[:,c])-parity(
        before-e_before[:,c])) for c in range(2)],axis=1)
    cross_before=-(e_before[:,0,0]*e_before[:,1,1]+
                   e_before[:,1,0]*e_before[:,0,1])
    cross_after=-(e_after[:,0,0]*e_after[:,1,1]+
                  e_after[:,1,0]*e_after[:,0,1])
    cross_delta=cross_after-cross_before
    # The residual of summing single-axis deletions is precisely the
    # omitted/duplicated A×B computation node of the bilinear readout.
    cross_gate=float(np.max(np.abs(total-effects.sum(axis=1)+cross_delta)))
    assert cross_gate<1e-8,cross_gate
    reference=(bias+np.einsum('nk,nok->no',panel['counts'],
                        matrix.reshape(2,3,8)[panel['query']]))
    gate=max(float(np.max(np.abs(before-reference))),cross_gate)
    assert gate<1e-9,gate
    return total,effects,gate


def scores(total,effects):
    energy=effects*effects
    effective=(np.abs(effects).sum(axis=1)**2)/np.maximum(energy.sum(axis=1),1e-20)
    weight=total*total
    return dict(energy_weighted_effective_axes=float(np.sum(weight*effective)/
                                                     np.maximum(weight.sum(),1e-20)),
                mean_absolute_component_effect=float(np.mean(np.abs(effects).sum(axis=1))),
                interaction_residual_relative_mse=float(np.sum((total-effects.sum(axis=1))**2)/
                                                        np.maximum(weight.sum(),1e-20)),
                total_edit_response_energy=float(np.sum(weight)))


def family_profile(matrix,bias,panel,dictionary):
    energies=np.zeros((2,3))
    for family in range(3):
        old=2*family;new=old+1
        mask=panel['counts'][:,old]>0
        variant={**panel,'counts':panel['counts'][mask],
                 'edited':panel['counts'][mask].copy(),
                 'query':panel['query'][mask]}
        variant['edited'][:,old]-=1
        variant['edited'][:,new]+=1
        _,effects,_=exact_components(matrix,bias,variant,dictionary)
        energies[:,family]=np.mean(effects*effects,axis=0)
    purity=float(np.sum(np.max(energies,axis=1))/np.maximum(energies.sum(),1e-20))
    return dict(energy_by_axis_family=energies.tolist(),
                weighted_family_purity=purity)


def key_alignment(dictionary):
    similarity=np.abs(dictionary@TRUTH.T)
    axis,family=linear_sum_assignment(-similarity)
    return dict(mean_distinct_family_cosine=float(np.mean(similarity[axis,family])),
                matched_families=[FAMILIES[i] for i in family],
                matched_cosines=similarity[axis,family].tolist(),
                all_cosines=similarity.tolist())


def run(grid=512,ntrain=4096,ntest=4096):
    source=json.loads((HERE/SOURCE).read_text())
    angles=np.arange(grid)*np.pi/(2*grid)
    rows=[]
    for row in source['rows']:
        seed=row['seed']
        # The query/output bias is fixed in every reduced function.
        import torch
        from train_three_count_language import ThreeCountAttention
        model=ThreeCountAttention().double()
        model.load_state_dict(torch.load(HERE/f'three_count_seed{seed}.pt',
                             map_location='cpu',weights_only=True))
        bias=model.read.bias.detach().numpy()
        train=edit_panel(ntrain,61000+seed,True)
        test=edit_panel(ntest,62000+seed,False)
        shifted=edit_panel(ntest,63000+seed,False,rate=1.2)
        for reduction in REDUCTIONS:
            matrix=np.asarray(row['matrices'][reduction],dtype=np.float64)
            u,s,v=np.linalg.svd(matrix,full_matrices=False)
            assert s[2]<1e-7
            dictionary=v[:2]
            initial_projector=dictionary.T@dictionary
            train_scores=[];key_l1=[]
            for angle in angles:
                d=rotation(angle)@dictionary
                total,effects,_=exact_components(matrix,bias,train,d)
                train_scores.append(scores(total,effects)['energy_weighted_effective_axes'])
                key_l1.append(float(np.abs(d).sum()))
            selected={'svd':0,
                      'key_l1':int(np.argmin(key_l1)),
                      'context_edit_sparsity':int(np.argmin(train_scores))}
            rng=np.random.default_rng(71000+seed)
            random_index=rng.integers(grid,size=64)
            gauges={};max_projector=0.
            for name,index in selected.items():
                d=rotation(angles[index])@dictionary
                projector_error=float(np.max(np.abs(d.T@d-initial_projector)))
                assert projector_error<1e-10
                max_projector=max(max_projector,projector_error)
                panels={}
                for panel_name,panel in (('train',train),('test',test),
                                         ('high_count_shift',shifted)):
                    total,effects,gate=exact_components(matrix,bias,panel,d)
                    panels[panel_name]=dict(**scores(total,effects),
                                             exact_function_gate=gate)
                gauges[name]=dict(angle=float(angles[index]),key_l1=float(np.abs(d).sum()),
                                  panels=panels,
                                  postfit_key_alignment=key_alignment(d),
                                  test_family_effects=family_profile(matrix,bias,test,d))
            random_effective=[];random_purity=[]
            random_alignment=[]
            for index in random_index:
                d=rotation(angles[index])@dictionary
                projector_error=float(np.max(np.abs(d.T@d-initial_projector)))
                assert projector_error<1e-10
                max_projector=max(max_projector,projector_error)
                total,effects,_=exact_components(matrix,bias,test,d)
                random_effective.append(scores(total,effects)[
                    'energy_weighted_effective_axes'])
                random_purity.append(family_profile(matrix,bias,test,d)[
                    'weighted_family_purity'])
                random_alignment.append(key_alignment(d)[
                    'mean_distinct_family_cosine'])
            test_total_energies=[gauges[name]['panels']['test'][
                'total_edit_response_energy'] for name in selected]
            test_total_relative_disagreement=float((max(test_total_energies)-
                min(test_total_energies))/max(test_total_energies))
            assert test_total_relative_disagreement<1e-10
            rows.append(dict(seed=seed,reduction=reduction,rank=2,
                fixed_function_projector_error=max_projector,
                test_total_relative_disagreement=test_total_relative_disagreement,
                train_grid_min_effective_axes=float(np.min(train_scores)),
                gauges=gauges,
                random_control=dict(n=len(random_index),
                    test_effective_axes_mean=float(np.mean(random_effective)),
                    test_effective_axes_p05=float(np.quantile(random_effective,.05)),
                    test_family_purity_mean=float(np.mean(random_purity)),
                    test_key_alignment_mean=float(np.mean(random_alignment)))))
        print(seed,flush=True)
    result=dict(source=SOURCE,rank=2,angle_grid=grid,
        n_fit_unlabeled_edit_contexts=ntrain,n_test_edit_contexts=ntest,
        fit_holdout='rare whole combination excluded; no task labels or A/B/C edit family labels used',
        objective='minimize energy-weighted effective number of axis-deletion effects on random single-key edit response of analytic bilinear output',
        scope='exact orthogonal rotations of each fixed rank-two key subspace; task fidelity is invariant; downstream readout analytic',
        rows=rows)
    name='three_count_nonlinear_gauge.json'
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--grid',type=int,default=512)
    p.add_argument('--ntrain',type=int,default=4096)
    p.add_argument('--ntest',type=int,default=4096)
    a=p.parse_args()
    run(a.grid,a.ntrain,a.ntest)
