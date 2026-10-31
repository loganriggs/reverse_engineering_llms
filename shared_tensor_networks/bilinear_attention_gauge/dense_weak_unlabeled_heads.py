"""Can internal head responses reveal a rule whose output is withheld from fit?"""

import json

import numpy as np
import torch
from scipy.optimize import minimize

from analyze_toy_language import rms
from analyze_dense_weak_rule_graph import RAW, graph_scores
from dense_weak_rule_gauges import rotation
from dense_weak_rule_reduction import (DIRECT_D_EDGES, ALL_PAIRS,
                                       empirical_weights, svd, table_logits)
from dual_joint_function_canonical import HERE
from three_count_metric import weighted_fit
from train_dense_four_count_weak import (DenseFourCountAttention, NTOKENS,
                                         examples)


def head_table(model):
    embedding=model.embed.weight.detach().numpy()
    queries,keys=embedding[16:18],embedding[:NTOKENS]
    heads=np.zeros((2,4,NTOKENS))
    for head in range(4):
        sl=slice(head*4,(head+1)*4)
        def projected(name,inputs):
            return rms(inputs@getattr(model,name).weight.detach().numpy()[sl].T)
        s1=projected('q1',queries)@projected('k1',keys).T/4
        s2=projected('q2',queries)@projected('k2',keys).T/4
        value=keys@model.v.weight.detach().numpy()[head]
        heads[:,head,:]=s1*s2*value
    return heads


def output_rows(head_rows,read):
    return np.einsum('oh,qhk->qok',read,head_rows.reshape(2,4,NTOKENS))\
        .reshape(8,NTOKENS)


def fit_edit_rotation(dictionary,pairs,seed,nstarts=32):
    def objective(angles):
        atoms=rotation(angles)@dictionary
        return float(np.mean(np.abs(atoms[:,pairs[:,1]]-
                                    atoms[:,pairs[:,0]]).sum(axis=0)))
    rng=np.random.default_rng(seed)
    starts=[np.zeros(6)]+[rng.uniform(-np.pi,np.pi,6) for _ in range(nstarts-1)]
    fits=[minimize(objective,start,method='Powell',options=dict(
        maxiter=150,xtol=1e-5,ftol=1e-6)) for start in starts]
    chosen=min(fits,key=lambda fit:fit.fun)
    return rotation(chosen.x),float(chosen.fun)


def direct_D_component_error(output_left,dictionary,matching):
    axis=matching[3]
    component=np.outer(output_left[:,axis],dictionary[axis])
    full=output_left@dictionary
    pair=DIRECT_D_EDGES
    delta=full.reshape(2,4,16)[:,:,pair[:,1]]-full.reshape(2,4,16)[:,:,pair[:,0]]
    part=component.reshape(2,4,16)[:,:,pair[:,1]]-component.reshape(2,4,16)[:,:,pair[:,0]]
    return float(np.sum((delta[:,3]-part[:,3])**2)/np.sum(delta[:,3]**2))


def match_raw(dictionary):
    from scipy.optimize import linear_sum_assignment
    normalized=dictionary/np.linalg.norm(dictionary,axis=1,keepdims=True)
    row,column=linear_sum_assignment(-np.abs(normalized@RAW.T))
    return {int(bit):int(axis) for axis,bit in zip(row,column)}


def run():
    torch.set_num_threads(1)
    excluded={tuple(pair) for pair in DIRECT_D_EDGES.tolist()}
    no_direct_D=np.array([pair for pair in ALL_PAIRS if tuple(pair) not in excluded])
    assert len(no_direct_D)==112
    rows=[];factors={}
    for seed in range(10):
        model=DenseFourCountAttention().double()
        model.load_state_dict(torch.load(HERE/f'dense_four_count_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        heads=head_table(model)
        matrix=heads.reshape(8,NTOKENS)
        read=model.read.weight.detach().numpy()
        bias=model.read.bias.detach().numpy()
        test,target,attrs=examples(30000,81000+seed,False)
        direct=model(test).detach().numpy()
        extracted=table_logits(output_rows(matrix,read).reshape(2,4,16)
                               .transpose(0,2,1),bias,test.numpy())
        gate=float(np.max(np.abs(direct-extracted)))
        assert gate<2e-10,gate
        fit_tokens,_,_=examples(100000,82000+seed,True)
        usage=empirical_weights(fit_tokens.numpy())
        measures={'weight_only':(svd(matrix,4),None),
                  'empirical_usage':weighted_fit(matrix,usage,4)}
        for measure,(cut,fit_info) in measures.items():
            cut_logits=table_logits(output_rows(cut,read).reshape(2,4,16)
                                    .transpose(0,2,1),bias,test.numpy())
            prediction=cut_logits>0
            native=direct>0
            labels=target.numpy().astype(bool)
            behavior=dict(
                native_joint_accuracy=float(np.mean(np.all(native==labels,axis=1))),
                joint_prediction_agreement=float(np.mean(np.all(prediction==native,axis=1))),
                D_prediction_agreement=float(np.mean(prediction[:,3]==native[:,3])),
                rare_joint_prediction_agreement=float(np.mean(np.all(
                    prediction[attrs['heldout']]==native[attrs['heldout']],axis=1))))
            u,s,v=np.linalg.svd(cut,full_matrices=False)
            left=u[:,:4]*np.sqrt(s[:4]);dictionary=np.sqrt(s[:4,None])*v[:4]
            assert np.max(np.abs(left@dictionary-cut))<1e-9
            bases={'svd':(np.eye(4),None)}
            for i,(name,pairs) in enumerate((
                    ('all_pair_edit_l1',ALL_PAIRS),
                    ('no_direct_D_flip_edit_l1',no_direct_D))):
                bases[name]=fit_edit_rotation(dictionary,pairs,
                        20261101+1000*seed+100*int(measure=='empirical_usage')+i)
            rng=np.random.default_rng(20261109+1000*seed+100*int(
                measure=='empirical_usage'))
            for draw in range(32):
                raw=rng.standard_normal((4,4))
                q,r=np.linalg.qr(raw)
                q=q*np.sign(np.diag(r))[None,:]
                bases[f'random_rotation_{draw}']=(q,None)
            for prior,(q,objective) in bases.items():
                atoms=q@dictionary
                inner_left=left@q.T
                # Map internal head coordinates into downstream output rows
                # only for evaluation. The fitting objective never uses read.
                out_left=np.einsum('oh,qhr->qor',read,inner_left.reshape(2,4,4))\
                           .reshape(8,4)
                score=graph_scores(out_left,atoms)
                match=match_raw(atoms)
                residual=float(np.sum((cut-matrix)**2)/np.sum(matrix**2))
                rows.append(dict(seed=seed,measure=measure,prior=prior,
                    exact_forward_max_abs=gate,
                    rank4_head_relative_mse=residual,
                    behavior=behavior,
                    gauge_objective=objective,fit=fit_info,
                    raw_atom_alignment=score['mean_raw_atom_alignment'],
                    graph_cell_accuracy_at_0_1=score['graph']['0.1']['cell_accuracy'],
                    exact_graph_row_fraction_at_0_1=score['graph']['0.1']['exact_row_fraction'],
                    D_flip_D_output_component_relative_error=direct_D_component_error(
                        out_left,atoms,match),
                    D_output_shared_atom_ratios=score['D_output_shared_atom_ratios']))
                prefix=f's{seed}_{measure}_{prior}'
                if not prior.startswith('random_rotation'):
                    factors[prefix+'_dictionary']=atoms
                    factors[prefix+'_head_left']=inner_left
                    factors[prefix+'_output_left']=out_left
        print('seed',seed,'complete',flush=True)
    summary={}
    for measure in measures:
        summary[measure]={}
        for prior in ('svd','all_pair_edit_l1','no_direct_D_flip_edit_l1',
                      'random_rotation'):
            part=[row for row in rows if row['measure']==measure and
                  (row['prior'].startswith('random_rotation') if prior=='random_rotation'
                   else row['prior']==prior)]
            summary[measure][prior]={key:float(np.mean([r[key] for r in part]))
                for key in ('rank4_head_relative_mse','raw_atom_alignment',
                            'graph_cell_accuracy_at_0_1',
                            'exact_graph_row_fraction_at_0_1',
                            'D_flip_D_output_component_relative_error')}
            summary[measure][prior]['behavior']={key:float(np.mean([
                r['behavior'][key] for r in part])) for key in part[0]['behavior']}
    result=dict(description='Fit four-dimensional dictionaries to internal QK1*QK2*V head response tables, without any output readout weights or task labels. D output and direct D-flip edits are used only in post-fit audit; no-direct-D prior excludes those eight edges from rotation fitting but sees other D-varying pairs.',
        fit_samples=100000,test_samples=30000,rows=rows,summary=summary)
    (HERE/'dense_weak_unlabeled_heads.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(HERE/'dense_weak_unlabeled_head_factors.npz',**factors)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
