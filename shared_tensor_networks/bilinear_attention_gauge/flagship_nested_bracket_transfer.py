"""Frozen bracket cuts and groups on nested inner/outer closer tasks."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_bracket_group_competent_transfer import margin,swap
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


PREFIXES=('','Compute ','Review ')
EXPRESSIONS=('3 * 8','6 + 9','12 - 4','11 - 2')


def head_state(model,probe,ids):
    final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:],stop_at=13)
    start=8*probe.head_dim
    return probe.last_input[0,-1,start:start+probe.head_dim].clone()


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    edit_meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    fit_swaps={swap(x['base_ids'],x['edited_ids']) for x in edit_meta['panels']['train']
               if x['type']=='bracket'}
    cases=[]
    for kind in ('outer','inner'):
        for prefix in PREFIXES:
            for expression in EXPRESSIONS:
                if kind=='outer':
                    base=f'{prefix}([{expression}]'
                    edited=f'{prefix}[[{expression}]'
                else:
                    base=f'{prefix}([{expression}'
                    edited=f'{prefix}(({expression}'
                before=tokenizer.encode(base,add_special_tokens=False).ids
                after=tokenizer.encode(edited,add_special_tokens=False).ids
                token_swap=swap(before,after)
                assert token_swap not in fit_swaps
                cases.append(dict(kind=kind,prefix=prefix,expression=expression,
                    base_text=base,edited_text=edited,base_ids=before,
                    edited_ids=after,token_swap=token_swap))
    assert len(cases)==24
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[13].attn);probe.head=8
    weight=np.load(ART/'output_aware_head_fit_bases.npz')['bracket_closure__weight']
    joint=np.load(ART/'head_joint_moment_fit_bases.npz')['bracket_closure__lambda_0.1']
    joint_gauge=json.loads((ART/'joint_bracket_legibility.json').read_text())
    joint_r=np.load(ART/'joint_bracket_legibility_rotations.npz')['raw_pca']
    joint_axes=joint_gauge['selectivity']['raw_pca']['selected_axes']
    weight_r=np.load(ART/'bracket_reduction_group_factorial_rotations.npz')['weight']
    bases={'weight':torch.tensor(weight.copy(),dtype=torch.float32),
           'joint':torch.tensor(joint.copy(),dtype=torch.float32)}
    rotations={'weight':torch.tensor(weight_r.copy(),dtype=torch.float32),
               'joint':torch.tensor(joint_r.copy(),dtype=torch.float32)}
    axes={'weight':[0,1],'joint':joint_axes}
    a=tokenizer.encode(')',add_special_tokens=False).ids[0]
    b=tokenizer.encode(']',add_special_tokens=False).ids[0]
    rows=[]
    with torch.no_grad():
        for case in cases:
            before,after=case['base_ids'],case['edited_ids']
            logits_base=final_logits(model,torch.tensor(before)[None,:])[0]
            logits_edited=final_logits(model,torch.tensor(after)[None,:])[0]
            base_margin=float(logits_base[a]-logits_base[b])
            edited_margin=float(logits_edited[a]-logits_edited[b])
            native_correct=(base_margin>0 and edited_margin<0 if case['kind']=='outer'
                            else base_margin<0 and edited_margin>0)
            delta=(head_state(model,probe,after)-head_state(model,probe,before))[None,:]
            probe.basis=torch.eye(probe.head_dim)
            probe.injection=delta
            native_path=margin(model,before,a,b)-base_margin
            probe.injection=None
            effects={};cut_margins={}
            for name,basis in bases.items():
                probe.basis=basis
                baseline=margin(model,before,a,b)
                cut_edited=margin(model,after,a,b)
                probe.injection=(delta@basis)@basis.T
                rank4=margin(model,before,a,b)-baseline
                probe.injection=None
                effects[name]={'rank4':rank4}
                for rank in (1,2):
                    directions=basis@rotations[name][:,axes[name][:rank]]
                    probe.injection=(delta@directions)@directions.T
                    effects[name][f'rank{rank}']=margin(model,before,a,b)-baseline
                    probe.injection=None
                cut_margins[name]=dict(base=baseline,edited=cut_edited)
            probe.basis=None
            rows.append(dict(**case,native_base_margin=base_margin,
                native_edited_margin=edited_margin,both_native_correct=native_correct,
                base_target_rank=(int((logits_base>logits_base[a]).sum())+1 if
                                  case['kind']=='outer' else int((logits_base>logits_base[b]).sum())+1),
                edited_target_rank=(int((logits_edited>logits_edited[b]).sum())+1 if
                                    case['kind']=='outer' else int((logits_edited>logits_edited[a]).sum())+1),
                native_head_path=native_path,effects=effects,cut_margins=cut_margins))
            if len(rows)%8==0:print('nested',len(rows),'/',len(cases),flush=True)
    summary=[]
    for kind in ('inner','outer','all'):
        for selection in ('all','competent'):
            part=[x for x in rows if (kind=='all' or x['kind']==kind) and
                  (selection=='all' or x['both_native_correct'])]
            norm=np.mean([abs(x['native_head_path']) for x in part])
            for name in bases:
                effects=[x['effects'][name] for x in part]
                cut_norm=np.mean([abs(e['rank4']) for e in effects])
                summary.append(dict(kind=kind,selection=selection,measure=name,
                    count=len(part),native_correct=sum(x['both_native_correct'] for x in part),
                    cut_correct=sum((x['cut_margins'][name]['base']>0 and
                                     x['cut_margins'][name]['edited']<0 if x['kind']=='outer'
                                     else x['cut_margins'][name]['base']<0 and
                                     x['cut_margins'][name]['edited']>0) for x in part),
                    mean_abs_native_head_path=float(norm),
                    rank4_relative_mae_vs_native=float(np.mean([
                        abs(e['rank4']-x['native_head_path']) for e,x in zip(effects,part)])/norm),
                    group_relative_mae_vs_rank4={f'rank{rank}':float(np.mean([
                        abs(e[f'rank{rank}']-e['rank4']) for e in effects])/cut_norm)
                        for rank in (1,2)},
                    group_relative_mae_vs_native={f'rank{rank}':float(np.mean([
                        abs(e[f'rank{rank}']-x['native_head_path'])
                        for e,x in zip(effects,part)])/norm) for rank in (1,2)}))
    result=dict(description='Exploratory but unfiltered nested-bracket transfer: frozen weight and joint rank-four cuts plus separately answer-free edit-PCA groups on 12 inner and 12 outer closer pairs; no nested examples used to fit cuts or groups',
        prefixes=PREFIXES,expressions=EXPRESSIONS,
        all_token_swaps_absent_from_original_bracket_edit_fit=True,
        no_nested_examples_in_fit=True,rows=rows,summary=summary)
    (ART/'nested_bracket_transfer.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in summary:
        if row['selection']=='all':print(row,flush=True)


if __name__=='__main__':run()
