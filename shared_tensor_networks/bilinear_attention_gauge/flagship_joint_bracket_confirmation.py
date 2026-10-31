"""Fresh, unfiltered bracket expressions for frozen joint rank-four cuts."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_bracket_group_competent_transfer import margin,swap
from flagship_competent_bracket_identities import FAMILIES
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


EXPRESSIONS=((3,'*',8),(6,'+',9),(12,'-',4),(11,'-',2))
METHODS=('weight','lambda_0','lambda_0.1','lambda_0.3')


def head_state(model,probe,ids):
    final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:],stop_at=13)
    start=8*probe.head_dim
    return probe.last_input[0,-1,start:start+probe.head_dim].clone()


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    previous=json.loads((ART/'competent_bracket_identities.json').read_text())
    existing={(x['base_text'],x['edited_text']) for x in previous['rows']}
    edits=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    fit_swaps={swap(x['base_ids'],x['edited_ids']) for x in edits['panels']['train']
               if x['type']=='bracket'}
    cases=[]
    for family,templates in FAMILIES.items():
        prefix,left,right=templates[-1]
        for a,operator,b in EXPRESSIONS:
            base=f'{prefix}{left}{a} {operator} {b}'
            edited=f'{prefix}{right}{a} {operator} {b}'
            assert (base,edited) not in existing
            before=tokenizer.encode(base,add_special_tokens=False).ids
            after=tokenizer.encode(edited,add_special_tokens=False).ids
            token_swap=swap(before,after)
            assert token_swap not in fit_swaps
            cases.append(dict(family=family,base_text=base,edited_text=edited,
                              base_ids=before,edited_ids=after,token_swap=token_swap))
    assert len(cases)==24 and len({x['token_swap'] for x in cases})==6
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[13].attn);probe.head=8
    old=np.load(ART/'output_aware_head_fit_bases.npz')
    joint=np.load(ART/'head_joint_moment_fit_bases.npz')
    bases={'weight':old['bracket_closure__weight'],
           **{name:joint[f'bracket_closure__{name}'] for name in METHODS if name!='weight'}}
    a=tokenizer.encode(')',add_special_tokens=False).ids[0]
    b=tokenizer.encode(']',add_special_tokens=False).ids[0]
    rows=[]
    with torch.no_grad():
        for case in cases:
            before,after=case['base_ids'],case['edited_ids']
            native_base_logits=final_logits(model,torch.tensor(before)[None,:])[0]
            native_edit_logits=final_logits(model,torch.tensor(after)[None,:])[0]
            native_base=float(native_base_logits[a]-native_base_logits[b])
            native_edited=float(native_edit_logits[a]-native_edit_logits[b])
            correct=native_base>0 and native_edited<0
            delta=(head_state(model,probe,after)-head_state(model,probe,before))[None,:]
            probe.basis=torch.eye(probe.head_dim)
            probe.injection=delta
            native_path=margin(model,before,a,b)-native_base
            probe.injection=None
            effects={};task_margins={}
            for name,basis in bases.items():
                basis=torch.tensor(basis.copy(),dtype=torch.float32)
                probe.basis=basis
                baseline=margin(model,before,a,b)
                edited_margin=margin(model,after,a,b)
                probe.injection=(delta@basis)@basis.T
                effects[name]=margin(model,before,a,b)-baseline
                probe.injection=None
                task_margins[name]=dict(base=baseline,edited=edited_margin)
            probe.basis=None
            rows.append(dict(**case,native_base_margin=native_base,
                native_edited_margin=native_edited,both_native_correct=correct,
                base_target_rank=int((native_base_logits>native_base_logits[a]).sum())+1,
                edited_target_rank=int((native_edit_logits>native_edit_logits[b]).sum())+1,
                native_head_path=native_path,effects=effects,
                task_margins=task_margins))
            if len(rows)%8==0:print('confirmation',len(rows),'/',len(cases),flush=True)
    summary=[]
    for panel in ('all','competent'):
        part=rows if panel=='all' else [x for x in rows if x['both_native_correct']]
        norm=np.mean([abs(x['native_head_path']) for x in part])
        for name in METHODS:
            summary.append(dict(panel=panel,method=name,count=len(part),
                native_correct=sum(x['both_native_correct'] for x in part),
                cut_correct=sum(x['task_margins'][name]['base']>0 and
                                x['task_margins'][name]['edited']<0 for x in part),
                mean_abs_native_head_path=float(norm),
                relative_mae_vs_native=float(np.mean([
                    abs(x['effects'][name]-x['native_head_path']) for x in part])/norm)))
    result=dict(description='Post-grid independent-context confirmation of fixed rank-four bracket cuts at six token-ID identities with four entirely new arithmetic expressions each, unfiltered',
        expressions=EXPRESSIONS,methods=METHODS,
        all_prompts_absent_from_prior_identity_panel=True,
        all_token_swaps_absent_from_edit_fit=True,rows=rows,summary=summary)
    (ART/'head_joint_bracket_confirmation.json').write_text(json.dumps(result,indent=2)+'\n')
    for item in summary:print(item,flush=True)


if __name__=='__main__':run()
