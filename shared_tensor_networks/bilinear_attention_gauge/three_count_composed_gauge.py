"""Exact-gauge priors for trained nonlinear count-attention readouts."""

import argparse
import json

import numpy as np
import torch

from dual_joint_function_canonical import HERE
from three_count_nonlinear_gauge import edit_panel,key_alignment,rotation,scores
from train_three_count_composed import ComposedCountModel


SOURCE='three_count_composed_analysis.json'
REDUCTIONS=('uniform_rank2','empirical_rank2')


def quadratic(y,q):
    return np.einsum('ni,ij,nj->n',y,q,y)


def quadratic_cross_fraction(matrix,q,dictionary):
    coefficients=(matrix@dictionary.T).reshape(2,3,2)
    h=np.einsum('qic,ij,qjd->qcd',coefficients,q,coefficients)
    return float(2*np.sum(h[:,0,1]**2)/np.maximum(np.sum(h*h),1e-30))


def components(matrix,bias,q,panel,dictionary):
    coefficients=(matrix@dictionary.T).reshape(2,3,2)
    def one(counts):
        coordinates=counts@dictionary.T
        axis=np.einsum('nc,noc->nco',coordinates,coefficients[panel['query']])
        return bias+axis.sum(axis=1),axis
    before,a=one(panel['counts']);after,b=one(panel['edited'])
    total=quadratic(after,q)-quadratic(before,q)
    effect=np.stack([total-(quadratic(after-b[:,c],q)-
                            quadratic(before-a[:,c],q)) for c in range(2)],axis=1)
    cross_before=2*np.einsum('ni,ij,nj->n',a[:,0],q,a[:,1])
    cross_after=2*np.einsum('ni,ij,nj->n',b[:,0],q,b[:,1])
    cross_gate=float(np.max(np.abs(total-effect.sum(axis=1)+
                                    cross_after-cross_before)))
    assert cross_gate<1e-8,cross_gate
    table=matrix.reshape(2,3,8)
    direct=bias+np.einsum('nk,nok->no',panel['counts'],table[panel['query']])
    function_gate=float(np.max(np.abs(before-direct)))
    assert function_gate<1e-9,function_gate
    return total,effect,max(cross_gate,function_gate)


def run(grid=512,ntrain=2048,ntest=2048,scratch=False):
    torch.set_num_threads(1)
    source=('three_count_composed_scratch_analysis.json' if scratch else SOURCE)
    analysis=json.loads((HERE/source).read_text())
    angles=np.arange(grid)*np.pi/(2*grid)
    rows=[]
    for row in analysis['rows']:
        seed=row['seed'];stage='scratch' if scratch else row['stage']
        model=ComposedCountModel().double()
        path=(HERE/f'three_count_composed_scratch_seed{seed}.pt' if scratch
              else HERE/f'three_count_composed_frozen_seed{seed}.pt'
              if stage=='frozen' else HERE/f'three_count_composed_seed{seed}.pt')
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        bias=model.backbone.read.bias.detach().numpy()
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        offset=(200000 if scratch else 0 if stage=='frozen' else 100000)
        train=edit_panel(ntrain,71000+offset+seed,True)
        test=edit_panel(ntest,72000+offset+seed,False)
        shift=edit_panel(ntest,73000+offset+seed,False,rate=1.2)
        for reduction in REDUCTIONS:
            matrix=np.asarray(row['rank2_matrices'][reduction])
            u,s,v=np.linalg.svd(matrix,full_matrices=False)
            assert s[2]<1e-7
            dictionary=v[:2];projector=dictionary.T@dictionary
            fit=[];key=[];cross=[]
            for angle in angles:
                d=rotation(angle)@dictionary
                total,effects,_=components(matrix,bias,q,train,d)
                fit.append(scores(total,effects)['energy_weighted_effective_axes'])
                key.append(float(np.abs(d).sum()))
                cross.append(quadratic_cross_fraction(matrix,q,d))
            selected={'svd':0,'key_l1':int(np.argmin(key)),
                      'context_edit':int(np.argmin(fit)),
                      'quadratic_diagonal':int(np.argmin(cross))}
            choices={};max_projector=0.
            for name,index in selected.items():
                d=rotation(angles[index])@dictionary
                err=float(np.max(np.abs(d.T@d-projector)))
                assert err<1e-10
                max_projector=max(max_projector,err)
                panels={}
                for label,panel in (('train',train),('test',test),('shift',shift)):
                    total,effects,gate=components(matrix,bias,q,panel,d)
                    panels[label]=dict(**scores(total,effects),exact_gate=gate)
                choices[name]=dict(angle=float(angles[index]),key_l1=float(np.abs(d).sum()),
                    alignment=key_alignment(d),
                    quadratic_cross_fraction=quadratic_cross_fraction(matrix,q,d),
                    panels=panels)
            energies=[c['panels']['test']['total_edit_response_energy']
                      for c in choices.values()]
            relative=(max(energies)-min(energies))/max(energies)
            assert relative<1e-10
            rng=np.random.default_rng(81000+offset+seed)
            random_effective=[];random_alignment=[]
            for index in rng.integers(grid,size=64):
                d=rotation(angles[index])@dictionary
                total,effects,_=components(matrix,bias,q,test,d)
                random_effective.append(scores(total,effects)[
                    'energy_weighted_effective_axes'])
                random_alignment.append(key_alignment(d)[
                    'mean_distinct_family_cosine'])
            rows.append(dict(seed=seed,stage=stage,reduction=reduction,
                max_projector_error=max_projector,
                test_response_energy_relative_disagreement=relative,
                random_control=dict(n=64,
                    mean_test_effective_axes=float(np.mean(random_effective)),
                    mean_key_alignment=float(np.mean(random_alignment))),
                gauges=choices))
        print(seed,stage,flush=True)
    result=dict(source=source,angle_grid=grid,n_fit_edit_contexts=ntrain,
        n_test_edit_contexts=ntest,rank=2,
        task_labels_used_in_gauge_fit=False,
        stages=['scratch'] if scratch else ['frozen','joint'],
        reductions=list(REDUCTIONS),rows=rows)
    output=('three_count_composed_scratch_gauge.json' if scratch
            else 'three_count_composed_gauge.json')
    (HERE/output).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scratch',action='store_true')
    args=parser.parse_args()
    run(scratch=args.scratch)
