"""Finite task-path check for frozen unlabeled and supervised head axes."""

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
from flagship_later_head_unlabeled_gauge import sparse_axis
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


def margin(model,ids,a,b):
    value=final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:])[0]
    return float(value[a]-value[b])


def candidates(target):
    small=np.load(ART/'later_head_unlabeled_gauge_rotations.npz')
    small_rows=json.loads((ART/'later_head_unlabeled_gauge.json').read_text())['records']
    large=np.load(ART/'later_head_unlabeled_large_eval_rotations.npz')
    large_rows=json.loads((ART/'later_head_unlabeled_large_eval.json').read_text())['records']
    multi=np.load(ART/'later_head_multicorpus_unlabeled_rotations.npz')
    multi_rows=json.loads((ART/'later_head_multicorpus_unlabeled.json').read_text())['records']
    sup=np.load(ART/'later_head_exact_gauge_rotations.npz')
    sup_summary=json.loads((ART/'later_head_exact_gauge_finite.json').read_text())['summary']
    z=np.load(ART/'later_head_unlabeled_large.npz')[f'{target}__fineweb_train']
    out={'svd_unlabeled_axis':(np.eye(4),sparse_axis(z,np.eye(4))[0]),
         'supervised_effect_axis':(sup[f'{target}__effect__all'],
            sup_summary[f'{target}__effect__all']['selected_axis'])}
    if target=='bracket_closure':
        small_method='activation_l1_centered';small_length=5
        large_method='activation_l1_centered';multi_method='activation_l1_centered'
    else:
        small_method='pca_centered';small_length=32
        large_method='pca_centered';multi_method='varimax_centered'
    x=next(r for r in small_rows if r['target']==target and r['length']==small_length and r['method']==small_method)
    out['small_natural_fit']=(small[f'{target}__l{small_length}__{small_method}'],x['selected_axis'])
    x=next(r for r in large_rows if r['target']==target and r['panel']=='full_8192' and r['method']==large_method)
    out['large_natural_fit']=(large[f'{target}__full_8192__{large_method}'],x['selected_axis'])
    for split in (0,1):
        x=next(r for r in multi_rows if r['target']==target and r['split']==split and
               r['method']==multi_method and r['rule']=='pooled_rare')
        out[f'multicorpus_split{split}']=(multi[f'{target}__split{split}__{multi_method}'],x['selected_axis'])
    return out


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
    selection={t:candidates(t) for t in TARGETS}
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
                z_edit=capture_head_states(model,probes,after)[layer][head]
                delta=(z_edit-z_base)[None,:]
                basis=torch.tensor(bases[f'{family}__weight'][:,:4].copy(),dtype=torch.float32)
                p.basis=basis
                baseline=margin(model,before,a[0],b[0])
                p.injection=(delta@basis)@basis.T
                full=margin(model,before,a[0],b[0])-baseline
                p.injection=None
                effects={}
                for name,(r,axis) in selection[family].items():
                    assert np.max(np.abs(r.T@r-np.eye(4)))<1e-6
                    direction=basis@torch.tensor(r[:,axis].copy(),dtype=torch.float32)
                    p.injection=(delta@direction)*direction[None,:]
                    effects[name]=margin(model,before,a[0],b[0])-baseline
                    p.injection=None
                p.basis=None
                rows.append(dict(panel=panel,family=family,base_prompt=base_prompt,
                    edited_prompt=edited_prompt,full_rank4_path_effect=full,
                    single_axis_effects=effects))
                print(panel,family,len(rows),flush=True)
    summary={}
    for panel in ('lexical_holdout','format_shift'):
        for family in TARGETS:
            part=[x for x in rows if x['panel']==panel and x['family']==family]
            full=np.array([x['full_rank4_path_effect'] for x in part])
            for name in selection[family]:
                effect=np.array([x['single_axis_effects'][name] for x in part])
                summary[f'{panel}__{family}__{name}']=dict(n=len(part),
                    single_axis_vs_full_rank4_relative_mae=float(
                        np.mean(np.abs(effect-full))/np.mean(np.abs(full))),
                    sign_matches=int(np.sum(np.sign(effect)==np.sign(full))))
    result=dict(description='Finite forward patch of unlabeled-selected axes versus full rank-4 head path; all selections frozen before task panels',
                caveat='Small and large natural candidates are exploratory choices from a larger grid; supervised effect axis is a positive control.',
                summary=summary,rows=rows)
    (ART/'later_head_unlabeled_finite.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary,flush=True)


if __name__=='__main__':run()
