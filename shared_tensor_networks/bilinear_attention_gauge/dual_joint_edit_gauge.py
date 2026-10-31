"""Reduction metric and legibility gauges on trained two-head retrieval."""

import argparse
import json
from itertools import combinations

import numpy as np
import torch

from dual_joint_function_canonical import (HERE,causal_metrics,flatten,
    matched_cosines,rotations,unflatten)
from dual_joint_metric import weighted_fit
from analyze_dual_output_language import factors,logits,score
from analyze_toy_language import pair_measure,pair_table
from train_dual_output_language import DualOutputAttention,batch,exhaustive
from train_correlated_dual_output_language import batch as correlated_batch

PAIRS={'all_pairs':tuple(combinations(range(6),2)),
       'within_role':((0,1),(2,3),(4,5))}


def run(family='original'):
    torch.set_num_threads(1)
    assert family in ('original','correlated')
    training_batch=batch if family=='original' else correlated_batch
    all_rot,angles=rotations()
    seq_t,target_t,attrs=exhaustive()
    seq=seq_t.numpy();target=target_t.numpy().astype(float)
    key_counts=np.eye(6)[seq[:,:3]].sum(axis=1)
    query=seq[:,-1]-6
    same=np.array([a['A']==a['B'] for a in attrs])
    correlated=np.where(same,.9,.1).astype(float)
    correlated/=correlated.sum()
    rows=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model_path=(HERE/f'dual_output_seed{seed}.pt' if family=='original'
                    else HERE/'correlated_dual_models'/f'seed{seed}.pt')
        model.load_state_dict(torch.load(model_path,
                                         map_location='cpu',weights_only=True))
        model.eval()
        heads=[factors(model,h) for h in range(2)]
        original=flatten(sum(pair_table(h,np.eye(h['rank'])) for h in heads))
        fit_seq,_=training_batch(100000,torch.Generator().manual_seed(100000+seed))
        fit_array=fit_seq.numpy()
        pair=pair_measure(fit_array)
        assert np.all(pair>0)
        empirical=np.repeat(pair,2,axis=0)
        empirical/=empirical.sum()
        uniform=np.ones_like(empirical)/empirical.size
        measures={'uniform':uniform,'empirical':empirical,
                  'half_uniform_empirical':.5*(uniform+empirical)}
        ordered=np.sort(fit_array[:,:3],axis=1)
        training_codes=8*ordered[:,0]+4*(ordered[:,1]-2)+2*(ordered[:,2]-4)+(fit_array[:,-1]-6)
        training_counts=np.bincount(training_codes,minlength=16)
        assert training_counts[5]==0 and training_counts[7]==0
        exhaustive_codes=np.array([8*a['A']+4*a['B']+2*a['F']+int(a['swap'])
                                   for a in attrs])
        training_weights=training_counts[exhaustive_codes]/(6*len(fit_array))
        assert abs(training_weights.sum()-1)<1e-12
        bias=model.read.bias.detach().numpy()
        original_table=unflatten(original)
        original_logits=logits([original_table],seq,bias)
        direct_logits=model(seq_t).detach().numpy()
        direct_gate=float(np.max(np.abs(direct_logits-original_logits)))
        assert direct_gate<1e-10,direct_gate
        original_scores=score([original_table],seq,target,bias,attrs)[0]
        for measure,weight in measures.items():
            approx,fit=weighted_fit(original,weight,2)
            u,s,v=np.linalg.svd(approx,full_matrices=False)
            left=u[:,:2]*np.sqrt(s[:2])
            dictionary=np.sqrt(s[:2,None])*v[:2]
            matrix=left@dictionary
            assert np.max(np.abs(matrix-approx))<1e-10
            reduced_logits=logits([unflatten(matrix)],seq,bias)
            heldout=np.array([a['heldout'] for a in attrs])
            full_logit_mse=float(np.mean((reduced_logits-original_logits)**2))
            rare_logit_mse=float(np.mean((reduced_logits[heldout]-original_logits[heldout])**2))
            candidates=np.einsum('tij,jk->tik',all_rot,dictionary)
            objective={'key_l1':np.abs(candidates).sum(axis=(1,2))}
            candidate_left=np.einsum('ij,tkj->tik',left,all_rot)
            objective['row_l1']=np.abs(candidate_left).sum(axis=(1,2))
            objective['balanced_factor_l1']=(
                objective['key_l1']/np.linalg.norm(dictionary)+
                objective['row_l1']/np.linalg.norm(left))
            objective['component_response_l1']=np.einsum(
                'trc,tck->t',np.abs(candidate_left),np.abs(candidates))
            key_sums=np.einsum('tck,nk->tnc',candidates,key_counts)
            left_query=candidate_left.reshape(len(all_rot),2,2,2)[:,query]
            sequence_response=np.abs(left_query*key_sums[:,:,None,:]).sum(axis=(2,3))
            objective['sequence_response_uniform_l1']=sequence_response.mean(axis=1)
            objective['sequence_response_correlated90_l1']=sequence_response@correlated
            objective['sequence_response_training_l1']=sequence_response@training_weights
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
                    objective_scores={key:float(value[index])
                                      for key,value in objective.items()},
                    matched_A_B_cosine=matched_cosines(d)[0],
                    key_l1=float(np.abs(d).sum()),
                    all_pair_edit_l1=float(np.mean([
                        np.sum(np.abs(d[:,j]-d[:,i])) for i,j in PAIRS['all_pairs']])),
                    within_role_edit_l1=float(np.mean([
                        np.sum(np.abs(d[:,j]-d[:,i])) for i,j in PAIRS['within_role']])),
                    mean_intended_bit_effect_fraction=float(np.mean([
                        z['intended_bit_energy_fraction_including_filler']
                        for z in causal.values()])),
                    component_removal={key:value['after_component_removal']
                                       for key,value in causal.items()},
                    exhaustive_joint_accuracy=predictions['joint_accuracy'],
                    rare_joint_accuracy=predictions['rare_heldout']['joint_accuracy'],
                    exhaustive_bce=predictions['bce'],
                    rare_bce=predictions['rare_heldout']['bce'])
            rows.append(dict(seed=seed,measure=measure,fit=fit,
                training_stratum_counts=training_counts.tolist(),
                direct_table_max_abs=direct_gate,original_scores=original_scores,
                full_logit_mse=full_logit_mse,rare_heldout_logit_mse=rare_logit_mse,
                relative_uniform_table_mse=float(np.sum((original-matrix)**2)/
                                                np.sum(original**2)),gauges=results))
        print('seed',seed,flush=True)
    result=dict(description='Uniform and frequency-weighted rank-two combined-token-function gauges for ten necessary-two-head retrieval models; edit measures use no task labels; within-role condition uses known input roles',
        family=family,seeds=list(range(10)),all_pair_count=len(PAIRS['all_pairs']),
        within_role_pair_count=len(PAIRS['within_role']),
        n_frequency_fit_sequences=100000,
        correlated_sequence_gauge_same_bit_probability=.9,
        correlated_sequence_gauge_is_synthetic_stress_test=True,
        model_training_uses_task_labels=True,gauge_fit_uses_task_labels=False,
        rows=rows)
    output=(HERE/'dual_joint_edit_gauge.json' if family=='original'
            else HERE/'correlated_dual_models'/'joint_edit_gauge.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    for measure in ('uniform','empirical','half_uniform_empirical'):
        for name in ('svd','key_l1','row_l1','balanced_factor_l1',
                     'component_response_l1','sequence_response_uniform_l1',
                     'sequence_response_correlated90_l1',
                     'sequence_response_training_l1','all_pairs','within_role'):
            part=[r['gauges'][name] for r in rows if r['measure']==measure]
            print(measure,name,'cosine',round(np.mean([r['matched_A_B_cosine'] for r in part]),3),
                'min',round(min(r['matched_A_B_cosine'] for r in part),3),
                'causal',round(np.mean([r['mean_intended_bit_effect_fraction'] for r in part]),3),
                'full',round(np.mean([r['exhaustive_joint_accuracy'] for r in part]),3),
                'rare',round(np.mean([r['rare_joint_accuracy'] for r in part]),3),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=('original','correlated'),default='original')
    run(parser.parse_args().family)
