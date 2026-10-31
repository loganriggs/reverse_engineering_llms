"""Frozen rank-4 gauges on new bracket and agreement prompt formats."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut, capture_head_states, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

VARIANTS=('svd','edit__all','effect__all')


def proposed_pairs():
    out=[]
    for operator in ('*','/','and',','):
        for a,b in ((2,3),(4,5)):
            out.append(('bracket_closure',f'({a} {operator} {b}',
                        f'[{a} {operator} {b}',')',']'))
    for prefix in ('If ','Compute '):
        for a,b in ((2,3),(4,5)):
            out.append(('bracket_closure',f'{prefix}({a} + {b}',
                        f'{prefix}[{a} + {b}',')',']'))
    for preposition in ('beside','behind','above','below','alongside'):
        for a,b,d in (('bird','birds','trees'),('doctor','doctors','patients')):
            out.append(('subject_verb_agreement',f'The {a} {preposition} the {d}',
                        f'The {b} {preposition} the {d}',' is',' are'))
    for prefix in ('Yesterday, ','Today, '):
        for a,b,d in (('bird','birds','trees'),('doctor','doctors','patients')):
            out.append(('subject_verb_agreement',f'{prefix}the {a} near the {d}',
                        f'{prefix}the {b} near the {d}',' is',' are'))
    return out


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
    with torch.no_grad():
        for family,base_prompt,edited_prompt,answer_a,answer_b in proposed_pairs():
            before=tokenizer.encode(base_prompt,add_special_tokens=False).ids
            after=tokenizer.encode(edited_prompt,add_special_tokens=False).ids
            a=tokenizer.encode(answer_a,add_special_tokens=False).ids
            b=tokenizer.encode(answer_b,add_special_tokens=False).ids
            assert len(before)==len(after) and sum(u!=v for u,v in zip(before,after))==1
            assert len(a)==len(b)==1
            layer,head=TARGETS[family];p=probes[layer];p.head=head
            native_base=margin(model,before,a[0],b[0])
            z_base=capture_head_states(model,probes,before)[layer][head]
            native_edit=margin(model,after,a[0],b[0])
            z_edit=capture_head_states(model,probes,after)[layer][head]
            delta=(z_edit-z_base)[None,:]
            p.basis=torch.eye(p.head_dim)
            p.injection=delta
            native_path=margin(model,before,a[0],b[0])-native_base
            p.injection=None
            basis=torch.tensor(bases[f'{family}__weight'][:,:4].copy(),dtype=torch.float32)
            p.basis=basis
            cut_base=margin(model,before,a[0],b[0])
            p.injection=(delta@basis)@basis.T
            rank4_path=margin(model,before,a[0],b[0])-cut_base
            p.injection=None
            effects={}
            for variant in VARIANTS:
                r=np.eye(4) if variant=='svd' else rotations[f'{family}__{variant}']
                axis=earlier[f'{family}__{variant}']['selected_axis']
                direction=basis@torch.tensor(r[:,axis].copy(),dtype=torch.float32)
                p.injection=(delta@direction)*direction[None,:]
                effects[variant]=margin(model,before,a[0],b[0])-cut_base
                p.injection=None
            p.basis=None
            rows.append(dict(family=family,base_prompt=base_prompt,edited_prompt=edited_prompt,
                native_base_margin=native_base,native_edited_margin=native_edit,
                both_native_correct=native_base>0 and native_edit<0,
                native_path_effect=native_path,rank4_path_effect=rank4_path,
                selected_axis_effects=effects))
            print(family,len(rows),'correct',rows[-1]['both_native_correct'],
                  'path',round(native_path,3),'effect-axis',round(effects['effect__all'],3),flush=True)
    summary={}
    for family in TARGETS:
        part=[x for x in rows if x['family']==family]
        native=np.array([x['native_path_effect'] for x in part])
        cut=np.array([x['rank4_path_effect'] for x in part])
        summary[family]=dict(n=len(part),both_native_correct=sum(x['both_native_correct'] for x in part),
            rank4_native_path_relative_mae=float(np.mean(np.abs(cut-native))/np.mean(np.abs(native))),
            variants={})
        for variant in VARIANTS:
            one=np.array([x['selected_axis_effects'][variant] for x in part])
            summary[family]['variants'][variant]=dict(
                single_axis_vs_rank4_relative_mae=float(np.mean(np.abs(one-cut))/np.mean(np.abs(cut))),
                single_axis_vs_native_relative_mae=float(np.mean(np.abs(one-native))/np.mean(np.abs(native))),
                sign_match_native=int(np.sum(np.sign(one)==np.sign(native))))
    result=dict(description='Unfiltered format-shift token pairs with frozen weight rank-4 cuts, rotations, and axis selections',
                summary=summary,rows=rows)
    (ART/'later_head_exact_gauge_format_shift.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary,flush=True)


if __name__=='__main__':run()
