"""Exact rank-four gauges and answer-free groups inside a faithful joint cut."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_bracket_group_competent_transfer import margin
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_exact_gauge import fit as fit_weight_l1
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_later_head_unlabeled_gauge import fit as fit_edit_prior,pca
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


SEED=20261122
RANDOM_CONTROLS=16
METHODS=('svd','raw_pca','normalized_pca','edit_l1','varimax','weight_l1')


def head_state(model,probe,ids):
    final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:],stop_at=13)
    start=8*probe.head_dim
    return probe.last_input[0,-1,start:start+probe.head_dim].clone()


def effective_axes(x):
    x=np.abs(x)
    return (x.sum(axis=-1)**2/np.maximum((x*x).sum(axis=-1),1e-30))


def run():
    torch.set_num_threads(4)
    rng=np.random.default_rng(SEED)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    train=[x for x in meta['panels']['train'] if x['type']=='bracket']
    validation=meta['panels']['validation']
    assert len(train)==80 and len(validation)==800
    types=np.array([x['type'] for x in validation])
    assert int(np.sum(types=='bracket'))==80
    bases_file=np.load(ART/'head_joint_moment_fit_bases.npz')
    basis=bases_file['bracket_closure__lambda_0.1'].copy()
    assert np.linalg.norm(basis.T@basis-np.eye(4))<1e-10
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    weights=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                       weights_only=True,mmap=True)
    model.load_state_dict(weights);del weights
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[13].attn);probe.head=8
    with torch.no_grad():
        train_raw=np.stack([(head_state(model,probe,x['edited_ids'])-
                             head_state(model,probe,x['base_ids'])).numpy()
                            for x in train]).astype(np.float64)
        val_raw=[]
        for i,x in enumerate(validation):
            val_raw.append((head_state(model,probe,x['edited_ids'])-
                            head_state(model,probe,x['base_ids'])).numpy())
            if (i+1)%200==0:print('validation edits',i+1,'/',len(validation),flush=True)
        val_raw=np.stack(val_raw).astype(np.float64)
    train_coords=train_raw@basis
    val_coords=val_raw@basis
    rotations={'svd':np.eye(4),
               'raw_pca':pca(train_coords),
               'normalized_pca':pca(train_coords/np.linalg.norm(
                   train_coords,axis=1,keepdims=True).clip(1e-12))}
    for kind,name in (('activation_l1','edit_l1'),('varimax','varimax')):
        rotations[name]=fit_edit_prior(train_coords,kind,rng)[0][1]
    w=model.transformer.h[13].attn.c_proj.weight.detach().numpy()[:,8*128:9*128]
    w_projected=torch.tensor(w.astype(np.float64)@basis,dtype=torch.float64)
    rotations['weight_l1']=fit_weight_l1('weight',w_projected,None,None,rng)[0]['rotation']
    for i in range(RANDOM_CONTROLS):
        q,_=np.linalg.qr(rng.standard_normal((4,4)))
        rotations[f'random_{i:02d}']=q
    assert all(np.linalg.norm(r.T@r-np.eye(4))<1e-8 for r in rotations.values())
    selections={}
    for name,r in rotations.items():
        train_energy=np.mean((train_coords@r)**2,axis=0)
        selections[name]=np.argsort(train_energy)[::-1][:2].tolist()
    val_bracket=val_coords[types=='bracket']
    independent_axis=pca(val_bracket)[:,0]
    pca_axis=rotations['raw_pca'][:,selections['raw_pca'][0]]
    train_val_axis_cosine=float(abs(independent_axis@pca_axis))
    selectivity={}
    for name,r in rotations.items():
        axis=selections[name][0]
        score=np.abs(val_coords@r[:,axis])
        top=np.argsort(score)[-80:]
        held=val_bracket@r
        selectivity[name]=dict(selected_axes=selections[name],
            validation_top_decile_bracket_precision=float(np.mean(types[top]=='bracket')),
            validation_bracket_effective_axes_mean=float(np.mean(effective_axes(held))),
            validation_bracket_rank1_energy_fraction=float(np.sum(held[:,axis]**2)/
                np.sum(held**2)))
    for name,score in [('full_head_norm',np.linalg.norm(val_raw,axis=1)),
                       ('joint_rank4_norm',np.linalg.norm(val_coords,axis=1))]:
        top=np.argsort(score)[-80:]
        selectivity[name]=dict(validation_top_decile_bracket_precision=float(
            np.mean(types[top]=='bracket')))
    fresh=json.loads((ART/'head_joint_bracket_confirmation.json').read_text())
    a=tokenizer.encode(')',add_special_tokens=False).ids[0]
    b=tokenizer.encode(']',add_special_tokens=False).ids[0]
    rows=[]
    with torch.no_grad():
        for case in fresh['rows']:
            before,after=case['base_ids'],case['edited_ids']
            delta=(head_state(model,probe,after)-head_state(model,probe,before))[None,:]
            probe.basis=torch.tensor(basis.copy(),dtype=torch.float32)
            baseline=margin(model,before,a,b)
            probe.injection=(delta@probe.basis)@probe.basis.T
            rank4=margin(model,before,a,b)-baseline
            probe.injection=None
            assert abs(rank4-case['effects']['lambda_0.1'])<1e-4
            effects={}
            for name,r in rotations.items():
                directions=selections[name]
                effects[name]={}
                for rank in (1,2):
                    group=probe.basis@torch.tensor(r[:,directions[:rank]].copy(),dtype=torch.float32)
                    probe.injection=(delta@group)@group.T
                    effects[name][f'rank{rank}']=margin(model,before,a,b)-baseline
                    probe.injection=None
            probe.basis=None
            rows.append(dict(family=case['family'],base_text=case['base_text'],
                edited_text=case['edited_text'],native_head_path=case['native_head_path'],
                joint_rank4_path=rank4,effects=effects))
            if len(rows)%8==0:print('fresh path',len(rows),'/',len(fresh['rows']),flush=True)
    norm=float(np.mean([abs(x['native_head_path']) for x in rows]))
    cut_norm=float(np.mean([abs(x['joint_rank4_path']) for x in rows]))
    path_summary={}
    for name in rotations:
        path_summary[name]={}
        for rank in (1,2):
            value=f'rank{rank}'
            path_summary[name][value]=dict(
                relative_mae_vs_native=float(np.mean([
                    abs(x['effects'][name][value]-x['native_head_path'])
                    for x in rows])/norm),
                relative_mae_vs_joint_rank4=float(np.mean([
                    abs(x['effects'][name][value]-x['joint_rank4_path'])
                    for x in rows])/cut_norm),
                sign_match_joint_rank4=int(sum(np.sign(x['effects'][name][value])==
                    np.sign(x['joint_rank4_path']) for x in rows)))
    result=dict(description='Exact gauges of the faithful joint bracket rank-four cut, fitted from 80 answer-free bracket edits or output weights; 800 independent diverse edit-type responses for selectivity and 24 fresh competent bracket prompts for finite paths',
        seed=SEED,random_controls=RANDOM_CONTROLS,methods=METHODS,
        train_bracket_edits=len(train),validation_edit_count=len(validation),
        validation_bracket_edits=int(np.sum(types=='bracket')),
        fit_uses_output_answer_labels=False,
        train_validation_raw_pca_axis_cosine=train_val_axis_cosine,
        exact_gauge_projector_max_fro_error=max(float(np.linalg.norm(
            basis@r@r.T@basis.T-basis@basis.T)) for r in rotations.values()),
        selectivity=selectivity,path_summary=path_summary,rows=rows)
    (ART/'joint_bracket_legibility.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ART/'joint_bracket_legibility_rotations.npz',**rotations)
    np.savez_compressed(ART/'joint_bracket_legibility_edit_responses.npz',
                        train_raw=train_raw,validation_raw=val_raw,
                        validation_types=types)
    print('PCA axis train/validation cosine',train_val_axis_cosine,flush=True)
    for name in METHODS:
        print(name,selectivity[name],path_summary[name],flush=True)


if __name__=='__main__':run()
