"""Finite held-out token edits: full model versus reduced, gauge-attributed function."""

import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_edit_family_holdout import HELDOUT
from three_count_metric import pair_weights,weighted_fit
from three_count_nonlinear_gauge import TRUTH
from three_count_task_fit import count_features
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


def panel(seed,rate,per_direction=128):
    tokens,_,attrs=examples(30000,170000+100*seed+int(rate*100),False,rate=rate)
    tokens=tokens.numpy()
    rng=np.random.default_rng(180000+100*seed+int(rate*100))
    before=[];after=[];metadata=[]
    strata={'common':~attrs['rare'],
            'rare_seen':attrs['rare']&~attrs['heldout'],
            'rare_heldout':attrs['heldout']}
    for stratum,mask in strata.items():
        for family,(i,j) in enumerate(HELDOUT):
            for source,destination in ((i,j),(j,i)):
                valid=np.flatnonzero(mask&np.any(tokens[:,:-1]==source,axis=1))
                requested=64 if stratum=='rare_heldout' else per_direction
                assert len(valid)>=requested,(seed,rate,stratum,source,len(valid))
                indices=rng.choice(valid,requested,replace=False)
                base=tokens[indices].copy()
                edited=base.copy()
                position=np.argmax(base[:,:-1]==source,axis=1)
                edited[np.arange(len(edited)),position]=destination
                before.append(base);after.append(edited)
                metadata.extend([dict(stratum=stratum,family=family,
                                      source=source,destination=destination)]*requested)
    return np.concatenate(before),np.concatenate(after),metadata


def outputs(matrix,q,bias,pb,tokens):
    counts=count_features(tokens).astype(float)
    query=(tokens[:,-1]-8).astype(int)
    bit=bias+np.einsum('nk,nok->no',counts,matrix.reshape(2,3,8)[query])
    parity=np.einsum('ni,ij,nj->n',bit,q,bit)+pb
    return np.c_[bit,parity]


def edit_components(matrix,q,bias,d,before,after):
    count_before=count_features(before).astype(float)
    delta_counts=count_features(after).astype(float)-count_before
    query=(before[:,-1]-8).astype(int)
    assert np.array_equal(query,after[:,-1]-8)
    left=matrix.reshape(2,3,8)@d.T
    z_before=count_before@d.T
    delta_z=delta_counts@d.T
    bit_before=bias+np.einsum('nc,noc->no',z_before,left[query])
    bit_parts=left[query]*delta_z[:,None,:]
    delta_bit=bit_parts.sum(axis=2)
    parity_parts=(2*np.einsum('no,op,npc->nc',bit_before,q,bit_parts)+
                  np.einsum('noc,op,np->nc',bit_parts,q,delta_bit))
    parts=np.concatenate((bit_parts,parity_parts[:,None,:]),axis=1)
    total=outputs(matrix,q,bias,0.,after)-outputs(matrix,q,bias,0.,before)
    assert np.max(np.abs(parts.sum(axis=2)-total))<1e-8
    return parts


def model_outputs(model,tokens):
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(tokens[i:i+2048])).numpy()
                               for i in range(0,len(tokens),2048)])


def relative_mse(predicted,reference):
    return float(np.mean((predicted-reference)**2)/
                 max(np.mean(reference**2),1e-20))


def run():
    torch.set_num_threads(1)
    gauge_source=json.loads((HERE/'three_count_edit_family_holdout.json').read_text())
    rows=[]
    for seed in range(10):
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        matrix=flatten(sum(head_tables(model.backbone)))
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        uniform=(u[:,:3]*s[:3])@v[:3]
        fit,_,_=examples(20000,110000+seed,True)
        empirical,_=weighted_fit(matrix,pair_weights(fit.numpy()),3)
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        bias=model.backbone.read.bias.detach().numpy()
        pb=float(model.parity_bias.detach())
        for rate in (.75,1.25):
            before,after,metadata=panel(seed,rate)
            exact_before=model_outputs(model,before)
            exact_after=model_outputs(model,after)
            table_gate=max(np.max(np.abs(exact_before-outputs(matrix,q,bias,pb,before))),
                           np.max(np.abs(exact_after-outputs(matrix,q,bias,pb,after))))
            assert table_gate<1e-9,table_gate
            reference=exact_after-exact_before
            for reduction,cut in (('uniform_rank3',uniform),
                                  ('empirical_rank3',empirical)):
                surrogate=outputs(cut,q,bias,pb,after)-outputs(cut,q,bias,pb,before)
                gauge_row=next(x for x in gauge_source['rows'] if
                               x['seed']==seed and x['reduction']==reduction)
                gauges=gauge_row['gauges']
                for name in ('svd','key_l1','all_pairs','fit_without_bit_flips'):
                    d=np.asarray(gauges[name]['dictionary'])
                    parts=edit_components(cut,q,bias,d,before,after)
                    assert np.max(np.abs(parts.sum(axis=2)-surrogate))<1e-8
                    match=linear_sum_assignment(-np.abs(d@TRUTH.T))
                    matched={int(variable):int(component)
                             for component,variable in zip(*match)}
                    for stratum in ('common','rare_seen','rare_heldout'):
                        for family in range(4):
                            mask=np.array([item['stratum']==stratum and
                                           item['family']==family for item in metadata])
                            full=reference[mask];approx=surrogate[mask]
                            row=dict(seed=seed,rate=rate,reduction=reduction,
                                gauge=name,stratum=stratum,family=family,
                                n=int(mask.sum()),
                                full_delta_energy=float(np.mean(full**2)),
                                surrogate_vs_full_relative_mse=relative_mse(approx,full),
                                per_output_relative_mse=[relative_mse(approx[:,i],full[:,i])
                                                         for i in range(4)],
                                max_exact_table_error=float(table_gate))
                            if family<3:
                                component=matched[family]
                                chosen=parts[mask,:,component]
                                row.update(matched_component=component,
                                    matched_component_vs_full_relative_mse=
                                        relative_mse(chosen,full),
                                    matched_shapley_energy_fraction=float(
                                        np.sum(chosen**2)/max(np.sum(parts[mask]**2),1e-20)))
                            rows.append(row)
        print('seed',seed,flush=True)
    result=dict(description='Finite unseen-context token substitutions on ten trained three-variable bilinear-attention models, with exact full-model outputs and rank-three reduced-function Shapley component attributions',
        gauge_fit_held_out_edges=HELDOUT,
        context_panels={'ordinary_rate':.75,'shifted_rate':1.25},
        gauge_matching_uses_known_variable_labels_only_after_fit=True,
        rows=rows)
    (HERE/'three_count_finite_edit_transfer.json').write_text(json.dumps(result,indent=2)+'\n')
    for reduction in ('uniform_rank3','empirical_rank3'):
        for rate in (.75,1.25):
            for name in ('svd','key_l1','all_pairs','fit_without_bit_flips'):
                subset=[r for r in rows if r['reduction']==reduction and
                        r['rate']==rate and r['gauge']==name and r['family']<3]
                print(reduction,rate,name,
                    'full_edit_mse',round(np.mean([r['surrogate_vs_full_relative_mse'] for r in subset]),4),
                    'matched_component_mse',round(np.mean([r['matched_component_vs_full_relative_mse'] for r in subset]),4),
                    'matched_energy',round(np.mean([r['matched_shapley_energy_fraction'] for r in subset]),4),flush=True)


if __name__=='__main__':run()
