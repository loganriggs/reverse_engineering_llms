"""Reduction metric x answer-free group rank on competent unseen brackets."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_bracket_group_competent_transfer import margin,state,swap
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import HeadCut
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


MEASURES=('weight','corpus','task_delta','mixed','random')


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    train=[x for x in meta['panels']['train'] if x['type']=='bracket']
    assert len(train)==80
    candidates=json.loads((ART/'competent_bracket_identities.json').read_text())
    tests=[x for x in candidates['rows'] if x['panel']=='validation']
    assert len(tests)==21 and all(x['both_native_correct'] for x in tests)
    fit_swaps={swap(x['base_ids'],x['edited_ids']) for x in train}
    assert all(swap(x['base_ids'],x['edited_ids']) not in fit_swaps
               for x in tests)
    with np.load(ART/'later_head_rank_pilot_bases.npz') as saved:
        bases={name:torch.tensor(saved[f'bracket_closure__{name}'][:,:4].copy(),
                                  dtype=torch.float32) for name in MEASURES}
        assert all(np.linalg.norm(saved[f'bracket_closure__{name}'][:,:4].T@
                                  saved[f'bracket_closure__{name}'][:,:4]-np.eye(4))
                   <1e-12 for name in MEASURES)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    weights=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                       weights_only=True,mmap=True)
    model.load_state_dict(weights);del weights
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[13].attn);probe.head=8
    a=tokenizer.encode(')',add_special_tokens=False).ids[0]
    b=tokenizer.encode(']',add_special_tokens=False).ids[0]
    fit_vectors=[]
    with torch.no_grad():
        for case in train:
            fit_vectors.append((state(model,probe,case['edited_ids'])-
                                state(model,probe,case['base_ids'])).numpy())
    fit_vectors=np.stack(fit_vectors).astype(np.float64)
    rotations={};spectra={}
    for name,basis in bases.items():
        coordinates=fit_vectors@basis.numpy().astype(np.float64)
        values,vecs=np.linalg.eigh(coordinates.T@coordinates/len(coordinates))
        rotations[name]=torch.tensor(vecs[:,::-1].copy(),dtype=torch.float32)
        spectra[name]=values[::-1].tolist()
    rows=[]
    with torch.no_grad():
        for index,case in enumerate(tests):
            before=case['base_ids'];after=case['edited_ids']
            delta=(state(model,probe,after)-state(model,probe,before))[None,:]
            probe.basis=torch.eye(probe.head_dim)
            native_baseline=margin(model,before,a,b)
            probe.injection=delta
            full_head=margin(model,before,a,b)-native_baseline
            probe.injection=None
            assert abs(full_head-case['native_head_path'])<1e-4
            effects={};task_margins={}
            for name,basis in bases.items():
                probe.basis=basis
                baseline=margin(model,before,a,b)
                edited_margin=margin(model,after,a,b)
                task_margins[name]=dict(base=baseline,edited=edited_margin)
                effect={}
                for rank in (1,2,4):
                    subspace=basis@rotations[name][:,:rank] if rank<4 else basis
                    probe.injection=(delta@subspace)@subspace.T
                    effect[f'rank{rank}']=margin(model,before,a,b)-baseline
                    probe.injection=None
                effects[name]=effect
                if name=='weight':
                    assert abs(effect['rank4']-case['rank4_path'])<1e-4
            probe.basis=None
            rows.append(dict(family=case['family'],
                base_text=case['base_text'],edited_text=case['edited_text'],
                token_swap=case['token_swap'],native_head_path=full_head,
                effects=effects,rank4_task_margins=task_margins))
            if (index+1)%7==0:print('validation',index+1,'/',len(tests),flush=True)
    summary=[]
    for family in list(dict.fromkeys(x['family'] for x in rows))+['all']:
        group=rows if family=='all' else [x for x in rows if x['family']==family]
        norm=np.mean([abs(x['native_head_path']) for x in group])
        for name in MEASURES:
            part=[x['effects'][name] for x in group]
            summary.append(dict(family=family,measure=name,n=len(group),
                mean_abs_native_head_path=float(norm),
                rank4_both_forced_choice_correct=sum(
                    x['rank4_task_margins'][name]['base']>0 and
                    x['rank4_task_margins'][name]['edited']<0
                    for x in group),
                relative_mae_vs_native={f'rank{rank}':float(np.mean([
                    abs(e[f'rank{rank}']-x['native_head_path'])
                    for e,x in zip(part,group)])/norm) for rank in (1,2,4)},
                relative_mae_vs_own_rank4={f'rank{rank}':float(np.mean([
                    abs(e[f'rank{rank}']-e['rank4']) for e in part])/
                    max(np.mean([abs(e['rank4']) for e in part]),1e-20))
                    for rank in (1,2)}))
    result=dict(description='Cross five prefit rank-four later-head reduction measures with the same answer-free 80-edit covariance rank-one/rank-two group fit; validate finite full-model head paths on 21 competent prompts with six unseen token-ID substitutions',
        measures=MEASURES,rank4_bases_from='later_head_rank_pilot_bases.npz',
        fit_edit_count=len(train),test_count=len(tests),
        all_test_token_swaps_absent_from_group_fit=True,
        task_labels_used_for_group_fit=False,fit_spectra=spectra,
        rows=rows,summary=summary)
    (ART/'bracket_reduction_group_factorial.json').write_text(
        json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ART/'bracket_reduction_group_factorial_rotations.npz',
                        **{k:v.numpy() for k,v in rotations.items()})
    for row in summary:
        if row['family']=='all':print(row,flush=True)


if __name__=='__main__':run()
