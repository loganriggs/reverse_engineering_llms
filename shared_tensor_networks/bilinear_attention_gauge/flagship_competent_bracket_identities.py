"""Frozen bracket-axis transfer to competent, unseen opening-token identities."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import HeadCut,capture_head_states,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


FAMILIES={
    'plain':(('', '(','['),('Evaluate','(', '['),('Compute\n','(', '['),('Review','(', '[')),
    'colon':(('Evaluate',':(', ':['),('Compute',':(', ':['),('Review',':(', ':[')),
    'double':(('', '((', '[['),('Evaluate','((', '[['),('Review','((', '[[')),
    'space_double':(('Try ','((', '[['),('Compute ','((', '[['),('Review ','((', '[[')),
    'equals':(('x','=(','=['),('y','=(','=['),('z','=(','=[')),
    'quote':(('He wrote ','"(', '"['),('She wrote ','"(', '"['),('They wrote ','"(', '"[')),
}
DISCOVERY=((2,'*',3),(4,'+',5),(7,'-',1))
VALIDATION=((5,'*',4),(8,'+',2),(9,'-',3))


def token_pair(a,b):
    assert len(a)==len(b)
    changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    assert len(changed)==1,(a,b)
    i=changed[0]
    return (a[i],b[i])


def cases(tokenizer):
    out=[]
    for family,templates in FAMILIES.items():
        for panel,prefixes,expressions in (
            ('discovery',templates[:2],DISCOVERY),
            ('validation',templates[2:],VALIDATION)):
            for prefix,left,right in prefixes:
                for a,op,b in expressions:
                    base=f'{prefix}{left}{a} {op} {b}'
                    edited=f'{prefix}{right}{a} {op} {b}'
                    base_ids=tokenizer.encode(base,add_special_tokens=False).ids
                    edited_ids=tokenizer.encode(edited,add_special_tokens=False).ids
                    swap=token_pair(base_ids,edited_ids)
                    out.append(dict(family=family,panel=panel,base_text=base,
                        edited_text=edited,base_ids=base_ids,edited_ids=edited_ids,
                        token_swap=swap))
    return out


def margin(model,ids,answer_a,answer_b):
    logits=final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:])[0]
    return float(logits[answer_a]-logits[answer_b])


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    seen={tuple(sorted(token_pair(x['base_ids'],x['edited_ids'])))
          for panel in meta['panels'].values() for x in panel}
    prompts=cases(tokenizer)
    identities={family:{tuple(c['token_swap']) for c in prompts if c['family']==family}
                for family in FAMILIES}
    assert all(len(pairs)==1 for pairs in identities.values()),identities
    assert len({next(iter(pairs)) for pairs in identities.values()})==len(FAMILIES)
    assert all(tuple(sorted(c['token_swap'])) not in seen for c in prompts)
    answer_a=tokenizer.encode(')',add_special_tokens=False).ids
    answer_b=tokenizer.encode(']',add_special_tokens=False).ids
    assert len(answer_a)==len(answer_b)==1
    a,b=answer_a[0],answer_b[0]
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    p=HeadCut(model.transformer.h[13].attn);p.head=8
    basis_file=np.load(ART/'later_head_rank_pilot_bases.npz')
    basis=torch.tensor(basis_file['bracket_closure__weight'][:,:4].copy(),dtype=torch.float32)
    rotation_file=np.load(ART/'later_head_unlabeled_edit_gauge_rotations.npz')
    selected=json.loads((ART/'later_head_unlabeled_edit_gauge.json').read_text())['records']
    record=next(x for x in selected if x['target']=='bracket_closure' and
                x['panel']=='train' and x['condition']=='all' and
                x['method']=='raw_pca' and x['selection_rule']=='global_energy')
    rotation=rotation_file['bracket_closure__train__all__raw_pca']
    axis=basis@torch.tensor(rotation[:,record['selected_axis']].copy(),dtype=torch.float32)
    rows=[]
    with torch.no_grad():
        for index,case in enumerate(prompts):
            before=case['base_ids'];after=case['edited_ids']
            logits_a=final_logits(model,torch.tensor(before,dtype=torch.long)[None,:])[0]
            logits_b=final_logits(model,torch.tensor(after,dtype=torch.long)[None,:])[0]
            native_a=float(logits_a[a]-logits_a[b])
            native_b=float(logits_b[a]-logits_b[b])
            result=dict(**case,native_base_margin=native_a,
                native_edited_margin=native_b,
                both_native_correct=bool(native_a>0 and native_b<0),
                base_target_rank=int((logits_a>logits_a[a]).sum())+1,
                edited_target_rank=int((logits_b>logits_b[b]).sum())+1,
                base_target_probability=float(torch.softmax(logits_a,dim=0)[a]),
                edited_target_probability=float(torch.softmax(logits_b,dim=0)[b]))
            if case['panel']=='validation':
                z_a=capture_head_states(model,{13:p},before)[13][8]
                z_b=capture_head_states(model,{13:p},after)[13][8]
                delta=(z_b-z_a)[None,:]
                p.basis=torch.eye(p.head_dim)
                full_base=margin(model,before,a,b)
                p.injection=delta
                native_path=margin(model,before,a,b)-full_base
                p.injection=None
                p.basis=basis
                cut_base=margin(model,before,a,b)
                p.injection=(delta@basis)@basis.T
                rank4_path=margin(model,before,a,b)-cut_base
                p.injection=None
                p.injection=(delta@axis)*axis[None,:]
                axis_path=margin(model,before,a,b)-cut_base
                p.injection=None;p.basis=None
                result.update(native_head_path=native_path,
                    rank4_path=rank4_path,frozen_axis_path=axis_path)
            rows.append(result)
            if (index+1)%12==0:print('done',index+1,'/',len(prompts),flush=True)
    summary=[]
    for family in FAMILIES:
        discovery=[r for r in rows if r['family']==family and r['panel']=='discovery']
        validation=[r for r in rows if r['family']==family and r['panel']=='validation']
        native=np.array([r['native_head_path'] for r in validation])
        cut=np.array([r['rank4_path'] for r in validation])
        axis_effect=np.array([r['frozen_axis_path'] for r in validation])
        summary.append(dict(family=family,token_swap=discovery[0]['token_swap'],
            n_discovery=len(discovery),n_validation=len(validation),
            discovery_both_correct=sum(r['both_native_correct'] for r in discovery),
            validation_both_correct=sum(r['both_native_correct'] for r in validation),
            validation_both_top1=sum(r['base_target_rank']==1 and
                                     r['edited_target_rank']==1 for r in validation),
            validation_median_target_rank=float(np.median([
                rank for r in validation for rank in
                (r['base_target_rank'],r['edited_target_rank'])])),
            predeclared_competent_selection=sum(r['both_native_correct'] for r in discovery)>=5,
            mean_abs_native_head_path=float(np.mean(np.abs(native))),
            rank4_vs_native_relative_mae=float(np.mean(np.abs(cut-native))/
                                               max(np.mean(np.abs(native)),1e-20)),
            frozen_axis_vs_rank4_relative_mae=float(np.mean(np.abs(axis_effect-cut))/
                                                    max(np.mean(np.abs(cut)),1e-20))))
    validation=[r for r in rows if r['panel']=='validation']
    stats=np.array([[sum(abs(r['frozen_axis_path']-r['rank4_path'])
                         for r in validation if r['family']==family),
                     sum(abs(r['rank4_path']) for r in validation if r['family']==family),
                     sum(abs(r['rank4_path']-r['native_head_path'])
                         for r in validation if r['family']==family),
                     sum(abs(r['native_head_path']) for r in validation if r['family']==family)]
                    for family in FAMILIES])
    rng=np.random.default_rng(20261114)
    chosen=rng.integers(0,len(FAMILIES),(10000,len(FAMILIES)))
    sampled=stats[chosen].sum(axis=1)
    overall=dict(n_validation=len(validation),
        both_forced_choice_correct=sum(r['both_native_correct'] for r in validation),
        both_targets_top1=sum(r['base_target_rank']==1 and
                              r['edited_target_rank']==1 for r in validation),
        max_target_rank=max(max(r['base_target_rank'],r['edited_target_rank'])
                            for r in validation),
        rank4_vs_native_relative_mae=float(stats[:,2].sum()/stats[:,3].sum()),
        frozen_axis_vs_rank4_relative_mae=float(stats[:,0].sum()/stats[:,1].sum()),
        rank4_identity_bootstrap95=np.quantile(sampled[:,2]/sampled[:,3],
                                               [.025,.975]).tolist(),
        axis_identity_bootstrap95=np.quantile(sampled[:,0]/sampled[:,1],
                                             [.025,.975]).tolist())
    result=dict(description='Screen new bracket token-ID swaps on discovery prompts; evaluate frozen rank-four head and answer-free axis on new prefixes and arithmetic expressions; all identity families retained',
        answer_token_ids=[a,b],all_swaps_absent_from_fit_panels=True,
        competence_selection_rule='at least 5 of 6 discovery prompt pairs correct',
        overall=overall,summary=summary,rows=rows)
    (ART/'competent_bracket_identities.json').write_text(json.dumps(result,indent=2)+'\n')
    for s in summary:print(s,flush=True)


if __name__=='__main__':run()
