"""Predeclared new lexical and numeric pairs for selected later-layer heads."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_head_scan import ProjectionInputProbe
from flagship_explicit_task_path_patch import score
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT, GPTConfig
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

TARGETS={'country_capital':(13,0),'bracket_closure':(13,8),
         'subject_verb_agreement':(11,3),'antonym':(8,7)}


def pairs():
    out=[]
    for a,aa,b,bb in [('Norway',' Oslo','Sweden',' Stockholm'),
                      ('Austria',' Vienna','Ireland',' Dublin'),
                      ('Belgium',' Brussels','Poland',' Warsaw'),
                      ('Denmark',' Copenhagen','Finland',' Helsinki'),
                      ('Turkey',' Ankara','Pakistan',' Islamabad')]:
        out.append(('country_capital',f'The capital of {a} is',f'The capital of {b} is',aa,bb))
    for a,b,d in [('bird','birds','trees'),('apple','apples','bowls'),
                  ('table','tables','rooms'),('parent','parents','children'),
                  ('doctor','doctors','patients'),('river','rivers','valleys')]:
        out.append(('subject_verb_agreement',f'The {a} near the {d}',
                    f'The {b} near the {d}',' is',' are'))
    for a,b in [(1,9),(2,8),(3,7),(4,6),(5,9),(8,1)]:
        out.append(('bracket_closure',f'({a} - {b}',f'[{a} - {b}',')',']'))
    for a,aa,b,bb in [('happy',' sad','sad',' happy'),
                      ('wet',' dry','dry',' wet'),
                      ('old',' young','young',' old'),
                      ('high',' low','low',' high')]:
        out.append(('antonym',f'The opposite of {a} is',
                    f'The opposite of {b} is',aa,bb))
    return out


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:ProjectionInputProbe(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    rows=[]
    with torch.no_grad():
        for family,base_prompt,edited_prompt,answer_a,answer_b in pairs():
            before=tokenizer.encode(base_prompt,add_special_tokens=False).ids
            after=tokenizer.encode(edited_prompt,add_special_tokens=False).ids
            choices=[tokenizer.encode(x,add_special_tokens=False).ids for x in (answer_a,answer_b)]
            assert len(before)==len(after) and sum(a!=b for a,b in zip(before,after))==1
            assert len(choices[0])==len(choices[1])==1
            layer,selected_head=TARGETS[family];probe=probes[layer]
            base=score(model,before,choices[0][0],choices[1][0])
            base_input=probe.last_input.clone()
            edited=score(model,after,choices[0][0],choices[1][0])
            delta=probe.last_input-base_input
            effects=[]
            for head in range(probe.n_head):
                a=head*probe.head_dim
                probe.injection=(head,delta[:,:,a:a+probe.head_dim])
                effects.append(score(model,before,choices[0][0],choices[1][0])-base)
                probe.injection=None
            a=selected_head*probe.head_dim
            selected_delta=delta[:,:,a:a+probe.head_dim]
            last_only=torch.zeros_like(selected_delta)
            last_only[:,-1]=selected_delta[:,-1]
            probe.injection=(selected_head,last_only)
            last_effect=score(model,before,choices[0][0],choices[1][0])-base
            probe.injection=None
            earlier_only=selected_delta-last_only
            probe.injection=(selected_head,earlier_only)
            earlier_effect=score(model,before,choices[0][0],choices[1][0])-base
            probe.injection=None
            rows.append(dict(family=family,base_prompt=base_prompt,
                             edited_prompt=edited_prompt,native_edit_effect=edited-base,
                             selected_layer=layer,selected_head=selected_head,
                             selected_head_effect=effects[selected_head],all_head_effects=effects,
                             selected_last_position_effect=last_effect,
                             selected_earlier_positions_effect=earlier_effect,
                             both_choices_correct=base>0 and edited<0))
            print(family,len(rows),round(edited-base,3),round(effects[selected_head],3),flush=True)
    summary={}
    for family in TARGETS:
        part=[r for r in rows if r['family']==family]
        x=np.array([r['all_head_effects'] for r in part])
        native=np.array([r['native_edit_effect'] for r in part])
        layer,head=TARGETS[family]
        summary[family]=dict(n=len(part),both_choices_correct=sum(r['both_choices_correct'] for r in part),
            selected_layer=layer,selected_head=head,
            mean_abs_native=float(np.mean(np.abs(native))),
            selected_mean_abs_effect=float(np.mean(np.abs(x[:,head]))),
            selected_sign_alignment=float(np.mean(np.sign(x[:,head])==np.sign(native))),
            selected_last_mean_abs_effect=float(np.mean(np.abs([r['selected_last_position_effect'] for r in part]))),
            selected_earlier_mean_abs_effect=float(np.mean(np.abs([r['selected_earlier_positions_effect'] for r in part]))),
            selected_head_fraction_of_sum_abs_head_effect=float(np.mean(np.abs(x[:,head]))/np.sum(np.mean(np.abs(x),axis=0))),
            mean_abs_by_head=np.mean(np.abs(x),axis=0).tolist())
    result=dict(description='Frozen layer/head selection from original 23 pairs; new lexical/numeric pairs, no outcome filtering',
                summary=summary,rows=rows)
    path=ART/'all_head_explicit_task_head_holdout.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',path,flush=True)


if __name__=='__main__':run()
