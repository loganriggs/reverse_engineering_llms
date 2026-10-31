"""Reverse natural head-state transplants into edited prompts."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_head_holdout import pairs as first_holdout
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_format_shift import proposed_pairs as format_shift
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, capture_head_states, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


def margin(model,ids,a,b):
    value=final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:])[0]
    return float(value[a]-value[b])


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    bases=np.load(ART/'later_head_rank_pilot_bases.npz')
    rotations=np.load(ART/'later_head_exact_gauge_rotations.npz')
    earlier=json.loads((ART/'later_head_exact_gauge_finite.json').read_text())['summary']
    rows=[]
    panels={'lexical_holdout':first_holdout(),'format_shift':format_shift()}
    with torch.no_grad():
        for panel,pairs in panels.items():
            for family,base_prompt,edited_prompt,answer_a,answer_b in pairs:
                if family not in TARGETS:continue
                before=tokenizer.encode(base_prompt,add_special_tokens=False).ids
                after=tokenizer.encode(edited_prompt,add_special_tokens=False).ids
                a=tokenizer.encode(answer_a,add_special_tokens=False).ids
                b=tokenizer.encode(answer_b,add_special_tokens=False).ids
                assert len(before)==len(after) and sum(u!=v for u,v in zip(before,after))==1
                assert len(a)==len(b)==1
                layer,head=TARGETS[family];p=probes[layer];p.head=head
                base=margin(model,before,a[0],b[0])
                z_base=capture_head_states(model,probes,before)[layer][head]
                edited=margin(model,after,a[0],b[0])
                z_edited=capture_head_states(model,probes,after)[layer][head]
                reverse_delta=(z_base-z_edited)[None,:]
                p.basis=torch.eye(p.head_dim)
                p.injection=reverse_delta
                native_reverse=margin(model,after,a[0],b[0])
                p.injection=None
                basis=torch.tensor(bases[f'{family}__weight'][:,:4].copy(),dtype=torch.float32)
                p.basis=basis
                cut_edit=margin(model,after,a[0],b[0])
                p.injection=(reverse_delta@basis)@basis.T
                cut_reverse=margin(model,after,a[0],b[0])
                p.injection=None
                axis=earlier[f'{family}__effect__all']['selected_axis']
                r=rotations[f'{family}__effect__all']
                direction=basis@torch.tensor(r[:,axis].copy(),dtype=torch.float32)
                p.injection=(reverse_delta@direction)*direction[None,:]
                axis_reverse=margin(model,after,a[0],b[0])
                p.injection=None;p.basis=None
                rows.append(dict(panel=panel,family=family,base_prompt=base_prompt,
                    edited_prompt=edited_prompt,base_margin=base,edited_margin=edited,
                    native_reverse_margin=native_reverse,cut_edited_margin=cut_edit,
                    cut_reverse_margin=cut_reverse,axis_reverse_margin=axis_reverse,
                    native_reverse_effect=native_reverse-edited,
                    cut_reverse_effect=cut_reverse-cut_edit,
                    axis_reverse_effect=axis_reverse-cut_edit,
                    both_native_correct=base>0 and edited<0,
                    native_reverse_flips_to_base=native_reverse>0,
                    axis_reverse_flips_to_base=axis_reverse>0))
                print(panel,family,len(rows),'reverse',round(native_reverse-edited,3),
                      'axis',round(axis_reverse-cut_edit,3),flush=True)
    summary={}
    for panel in panels:
        for family in TARGETS:
            part=[x for x in rows if x['panel']==panel and x['family']==family]
            native=np.array([x['native_reverse_effect'] for x in part])
            cut=np.array([x['cut_reverse_effect'] for x in part])
            axis=np.array([x['axis_reverse_effect'] for x in part])
            summary[f'{panel}__{family}']=dict(n=len(part),
                both_native_correct=sum(x['both_native_correct'] for x in part),
                native_reverse_to_base_flips=sum(x['native_reverse_flips_to_base'] for x in part),
                axis_reverse_to_base_flips=sum(x['axis_reverse_flips_to_base'] for x in part),
                mean_abs_native_reverse=float(np.mean(np.abs(native))),
                cut_vs_native_relative_mae=float(np.mean(np.abs(cut-native))/np.mean(np.abs(native))),
                axis_vs_cut_relative_mae=float(np.mean(np.abs(axis-cut))/np.mean(np.abs(cut))),
                axis_vs_native_relative_mae=float(np.mean(np.abs(axis-native))/np.mean(np.abs(native))),
                axis_sign_match_native=int(np.sum(np.sign(axis)==np.sign(native))))
    result=dict(description='Reverse base-minus-edited head-output transplant into edited prompt; frozen weight rank-4 cuts and effect axes',
                caveat='Reverse patch tests path control in an edited context, not formal necessity under all possible alternatives.',
                summary=summary,rows=rows)
    (ART/'later_head_reverse_patch.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary,flush=True)


if __name__=='__main__':run()
