"""Does a two-axis bracket group cover new edit identities at fixed rank four?"""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_format_shift import proposed_pairs as format_pairs
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

TARGET='bracket_closure'
LAYER,HEAD=13,8


def margin(model,ids,a,b):
    out=final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:])[0]
    return float(out[a]-out[b])


def state(model,p,ids):
    final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:],stop_at=LAYER)
    return p.last_input[0,-1,HEAD*p.head_dim:(HEAD+1)*p.head_dim].clone()


def moment(arr):
    return arr.T@arr/len(arr)


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    unseen=json.loads((ART/'unseen_token_edit_transfer.json').read_text())['rows']
    brace=[x for x in unseen if x['family']==TARGET]
    fit_brace=[x for x in brace if not x['base_text'].startswith('If ')]
    test_brace=[x for x in brace if x['base_text'].startswith('If ')]
    assert len(fit_brace)==12 and len(test_brace)==6
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    weights=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(weights);del weights
    model.requires_grad_(False)
    p=HeadCut(model.transformer.h[LAYER].attn);p.head=HEAD
    basis_np=np.load(ART/'later_head_rank_pilot_bases.npz')[f'{TARGET}__weight'][:,:4].copy()
    basis=torch.tensor(basis_np,dtype=torch.float32)
    edits=np.load(ART/'later_head_unlabeled_edits.npz')[f'{TARGET}__train']
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())['panels']['train']
    original=np.asarray([v for v,r in zip(edits,meta) if r['type']=='bracket'],dtype=np.float64)
    assert len(original)==80
    brace_coords={'brace_to_paren':[],'brace_to_square':[]}
    with torch.no_grad():
        for case in fit_brace:
            before,after=case['base_ids'],case['edited_ids']
            d=state(model,p,after)-state(model,p,before)
            group=('brace_to_paren' if '(' in case['edited_text'].split(' ')[1]
                   else 'brace_to_square')
            brace_coords[group].append((d@basis).numpy())
    groups={'paren_to_square':original,
            **{name:np.stack(values).astype(np.float64)
               for name,values in brace_coords.items()}}
    assert all(len(groups[name])==6 for name in brace_coords)
    moments={name:moment(x) for name,x in groups.items()}
    balanced=sum(m/max(np.trace(m),1e-30) for m in moments.values())/len(moments)
    raw=sum(m*len(groups[name]) for name,m in moments.items())/sum(map(len,groups.values()))
    def eigenbasis(m):
        vals,vecs=np.linalg.eigh(m)
        return vals[::-1],vecs[:,::-1].copy()
    values,rotation=eigenbasis(balanced)
    raw_values,raw_rotation=eigenbasis(raw)
    prior=np.load(ART/'later_head_unlabeled_edit_gauge_rotations.npz')[
        f'{TARGET}__train__all__raw_pca']
    chosen=next(x['selected_axis'] for x in json.loads((
        ART/'later_head_unlabeled_edit_gauge.json').read_text())['records']
        if x['target']==TARGET and x['panel']=='train' and x['condition']=='all'
        and x['method']=='raw_pca' and x['selection_rule']=='global_energy')
    projections={'old_axis':prior[:,chosen:chosen+1],
                 **{f'balanced_rank{r}':rotation[:,:r] for r in (1,2,3)},
                 **{f'raw_rank{r}':raw_rotation[:,:r] for r in (1,2,3)}}
    tests=[]
    for case in test_brace:
        tests.append(dict(panel='new_brace_context',base_ids=case['base_ids'],
            edited_ids=case['edited_ids'],answer=case['answer'],
            group='brace_to_paren' if '(' in case['edited_text'].split(' ')[1]
                  else 'brace_to_square'))
    for family,a_text,b_text,a_token,b_token in format_pairs():
        if family!=TARGET:continue
        a=tokenizer.encode(a_text,add_special_tokens=False).ids
        b=tokenizer.encode(b_text,add_special_tokens=False).ids
        answer=[tokenizer.encode(s,add_special_tokens=False).ids[0]
                for s in (a_token,b_token)]
        tests.append(dict(panel='old_pair_new_format',base_ids=a,edited_ids=b,
                          answer=answer,group='paren_to_square'))
    rows=[]
    with torch.no_grad():
        for case in tests:
            before,after=case['base_ids'],case['edited_ids'];a,b=case['answer']
            delta=(state(model,p,after)-state(model,p,before))[None,:]
            p.basis=basis
            base=margin(model,before,a,b)
            p.injection=(delta@basis)@basis.T
            full=margin(model,before,a,b)-base
            p.injection=None
            effects={}
            for name,r in projections.items():
                w=basis@torch.tensor(r.copy(),dtype=torch.float32)
                p.injection=(delta@w)@w.T
                effects[name]=margin(model,before,a,b)-base
                p.injection=None
            p.basis=None
            rows.append(dict(panel=case['panel'],group=case['group'],
                base_ids=before,edited_ids=after,rank4_path=full,effects=effects))
    summary={}
    for panel in ('new_brace_context','old_pair_new_format'):
        for group in sorted({x['group'] for x in rows if x['panel']==panel}):
            part=[x for x in rows if x['panel']==panel and x['group']==group]
            full=np.asarray([x['rank4_path'] for x in part]);s={}
            for name in projections:
                effect=np.asarray([x['effects'][name] for x in part])
                s[name]=float(np.mean(np.abs(effect-full))/
                              max(np.mean(np.abs(full)),1e-20))
            summary[f'{panel}__{group}']=dict(n=len(part),
                mean_abs_rank4_path=float(np.mean(np.abs(full))),relative_mae=s)
    (ART/'bracket_edit_span.json').write_text(json.dumps(dict(
        description='Bracket edit-identity group span at fixed weight rank4; fit uses original 80 paren/square edits plus 12 brace edits from Try/Compute, tests heldout If braces and old-pair new formats',
        fit_group_counts={k:len(v) for k,v in groups.items()},
        balanced_eigenvalues=values.tolist(),raw_eigenvalues=raw_values.tolist(),
        summary=summary,rows=rows),indent=2)+'\n')
    print('balanced eigenvalues',values,'raw',raw_values,flush=True)
    for key,val in summary.items():print(key,val,flush=True)


if __name__=='__main__':run()
