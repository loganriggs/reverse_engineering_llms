"""All-pair versus role-preserving edit gauges on necessary two-head retrieval."""

import json
from itertools import combinations

import numpy as np
import torch

from dual_joint_function_canonical import (HERE,causal_metrics,flatten,
    matched_cosines,rotations,unflatten)
from dual_joint_metric import weighted_fit
from analyze_dual_output_language import factors,score
from analyze_toy_language import pair_measure,pair_table
from train_dual_output_language import DualOutputAttention,batch,exhaustive

PAIRS={'all_pairs':tuple(combinations(range(6),2)),
       'within_role':((0,1),(2,3),(4,5))}


def run():
    torch.set_num_threads(1)
    all_rot,angles=rotations()
    seq_t,target_t,attrs=exhaustive()
    seq=seq_t.numpy();target=target_t.numpy().astype(float)
    rows=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f'dual_output_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        heads=[factors(model,h) for h in range(2)]
        original=flatten(sum(pair_table(h,np.eye(h['rank'])) for h in heads))
        fit_seq,_=batch(100000,torch.Generator().manual_seed(100000+seed))
        pair=pair_measure(fit_seq.numpy())
        assert np.all(pair>0)
        empirical=np.repeat(pair,2,axis=0)
        empirical/=empirical.sum()
        uniform=np.ones_like(empirical)/empirical.size
        measures={'uniform':uniform,'empirical':empirical,
                  'half_uniform_empirical':.5*(uniform+empirical)}
        bias=model.read.bias.detach().numpy()
        for measure,weight in measures.items():
            approx,fit=weighted_fit(original,weight,2)
            u,s,v=np.linalg.svd(approx,full_matrices=False)
            left=u[:,:2]*np.sqrt(s[:2])
            dictionary=np.sqrt(s[:2,None])*v[:2]
            matrix=left@dictionary
            assert np.max(np.abs(matrix-approx))<1e-10
            candidates=np.einsum('tij,jk->tik',all_rot,dictionary)
            objective={'key_l1':np.abs(candidates).sum(axis=(1,2))}
            candidate_left=np.einsum('ij,tkj->tik',left,all_rot)
            objective['row_l1']=np.abs(candidate_left).sum(axis=(1,2))
            objective['balanced_factor_l1']=(
                objective['key_l1']/np.linalg.norm(dictionary)+
                objective['row_l1']/np.linalg.norm(left))
            objective['component_response_l1']=np.einsum(
                'trc,tck->t',np.abs(candidate_left),np.abs(candidates))
            for name,pairs in PAIRS.items():
                edges=np.asarray(pairs)
                changes=candidates[:,:,edges[:,1]]-candidates[:,:,edges[:,0]]
                objective[name]=np.mean(np.sum(np.abs(changes),axis=1),axis=1)
            choices={'svd':0,**{name:int(np.argmin(value))
                                for name,value in objective.items()}}
            results={}
            for name,index in choices.items():
                d=candidates[index]
                l=left@all_rot[index].T
                assert np.max(np.abs(l@d-matrix))<1e-10
                predictions=score([unflatten(l@d)],seq,target,bias,attrs)[0]
                causal=causal_metrics(l,d,seq,target,attrs,bias)
                results[name]=dict(angle=float(angles[index]),
                    dictionary=d.tolist(),row_factors=l.tolist(),
                    matched_A_B_cosine=matched_cosines(d)[0],
                    key_l1=float(np.abs(d).sum()),
                    all_pair_edit_l1=float(np.mean([
                        np.sum(np.abs(d[:,j]-d[:,i])) for i,j in PAIRS['all_pairs']])),
                    within_role_edit_l1=float(np.mean([
                        np.sum(np.abs(d[:,j]-d[:,i])) for i,j in PAIRS['within_role']])),
                    mean_intended_bit_effect_fraction=float(np.mean([
                        z['intended_bit_energy_fraction_including_filler']
                        for z in causal.values()])),
                    exhaustive_joint_accuracy=predictions['joint_accuracy'],
                    rare_joint_accuracy=predictions['rare_heldout']['joint_accuracy'])
            rows.append(dict(seed=seed,measure=measure,fit=fit,
                relative_uniform_table_mse=float(np.sum((original-matrix)**2)/
                                                np.sum(original**2)),gauges=results))
        print('seed',seed,flush=True)
    result=dict(description='Uniform and frequency-weighted rank-two combined-token-function gauges for ten necessary-two-head retrieval models; edit measures use no task labels; within-role condition uses known input roles',
        seeds=list(range(10)),all_pair_count=len(PAIRS['all_pairs']),
        within_role_pair_count=len(PAIRS['within_role']),
        n_frequency_fit_sequences=100000,
        model_training_uses_task_labels=True,gauge_fit_uses_task_labels=False,
        rows=rows)
    (HERE/'dual_joint_edit_gauge.json').write_text(json.dumps(result,indent=2)+'\n')
    for measure in ('uniform','empirical','half_uniform_empirical'):
        for name in ('svd','key_l1','row_l1','balanced_factor_l1',
                     'component_response_l1','all_pairs','within_role'):
            part=[r['gauges'][name] for r in rows if r['measure']==measure]
            print(measure,name,'cosine',round(np.mean([r['matched_A_B_cosine'] for r in part]),3),
                'min',round(min(r['matched_A_B_cosine'] for r in part),3),
                'causal',round(np.mean([r['mean_intended_bit_effect_fraction'] for r in part]),3),
                'full',round(np.mean([r['exhaustive_joint_accuracy'] for r in part]),3),
                'rare',round(np.mean([r['rare_joint_accuracy'] for r in part]),3),flush=True)


if __name__=='__main__':run()
