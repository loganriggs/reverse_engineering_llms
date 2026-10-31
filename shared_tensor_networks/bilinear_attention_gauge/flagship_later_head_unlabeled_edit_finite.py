"""Finite held-out task paths for answer-free edit-derived head gauges."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_head_holdout import pairs as lexical_pairs
from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_format_shift import proposed_pairs as format_pairs
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut,capture_head_states,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

VARIANTS=(('train','all','raw_pca'),('validation','all','raw_pca'),
          ('train','all','normalized_pca'),('train','all','edit_l1'),
          ('validation','all','edit_l1'),
          ('train','exclude_target_type','raw_pca'),
          ('train','exclude_target_type','edit_l1'))


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
    rotations=np.load(ART/'later_head_unlabeled_edit_gauge_rotations.npz')
    records=json.loads((ART/'later_head_unlabeled_edit_gauge.json').read_text())['records']
    supervised=np.load(ART/'later_head_exact_gauge_rotations.npz')
    supervised_axes=json.loads((ART/'later_head_exact_gauge_finite.json').read_text())['summary']
    choices={}
    for target in TARGETS:
        choices[target]={}
        for panel,condition,method in VARIANTS:
            key=f'{panel}__{condition}__{method}'
            row=next(x for x in records if x['target']==target and x['panel']==panel and
                     x['condition']==condition and x['method']==method and
                     x['selection_rule']=='global_energy')
            choices[target][key]=(rotations[f'{target}__{key}'],row['selected_axis'])
        choices[target]['supervised_effect']=(supervised[f'{target}__effect__all'],
            supervised_axes[f'{target}__effect__all']['selected_axis'])
    rows=[]
    with torch.no_grad():
        for panel,pairs in [('lexical_holdout',lexical_pairs()),('format_shift',format_pairs())]:
            for family,base_prompt,edited_prompt,answer_a,answer_b in pairs:
                if family not in TARGETS:continue
                before=tokenizer.encode(base_prompt,add_special_tokens=False).ids
                after=tokenizer.encode(edited_prompt,add_special_tokens=False).ids
                a=tokenizer.encode(answer_a,add_special_tokens=False).ids
                b=tokenizer.encode(answer_b,add_special_tokens=False).ids
                assert len(a)==len(b)==1
                layer,head=TARGETS[family];p=probes[layer];p.head=head
                z_base=capture_head_states(model,probes,before)[layer][head]
                z_edited=capture_head_states(model,probes,after)[layer][head]
                delta=(z_edited-z_base)[None,:]
                basis=torch.tensor(bases[f'{family}__weight'][:,:4].copy(),dtype=torch.float32)
                p.basis=basis
                base=margin(model,before,a[0],b[0])
                p.injection=(delta@basis)@basis.T
                full=margin(model,before,a[0],b[0])-base
                p.injection=None
                effects={}
                for name,(r,axis) in choices[family].items():
                    assert np.max(abs(r.T@r-np.eye(4)))<1e-6
                    u=basis@torch.tensor(r[:,axis].copy(),dtype=torch.float32)
                    p.injection=(delta@u)*u[None,:]
                    effects[name]=margin(model,before,a[0],b[0])-base
                    p.injection=None
                p.basis=None
                rows.append(dict(panel=panel,family=family,base_prompt=base_prompt,
                    edited_prompt=edited_prompt,full_rank4_path_effect=full,
                    single_axis_effects=effects))
                print(panel,family,len(rows),flush=True)
    summary={}
    for panel in ('lexical_holdout','format_shift'):
        for target in TARGETS:
            part=[x for x in rows if x['panel']==panel and x['family']==target]
            full=np.array([x['full_rank4_path_effect'] for x in part])
            for name in choices[target]:
                effect=np.array([x['single_axis_effects'][name] for x in part])
                summary[f'{panel}__{target}__{name}']=dict(n=len(part),
                    relative_mae=float(np.mean(np.abs(effect-full))/np.mean(np.abs(full))),
                    sign_matches=int(np.sum(np.sign(effect)==np.sign(full))))
    result=dict(description='Finite single-axis patches from answer-free balanced edit gauges; separate train and validation fits; frozen task panels',
                variants=['__'.join(x) for x in VARIANTS]+['supervised_effect'],
                summary=summary,rows=rows)
    (ART/'later_head_unlabeled_edit_finite.json').write_text(json.dumps(result,indent=2)+'\n')
    for target in TARGETS:
        print(target,[(name,round(summary[f'format_shift__{target}__{name}']['relative_mae'],3))
                      for name in choices[target]],flush=True)


if __name__=='__main__':run()
