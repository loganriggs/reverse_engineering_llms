"""Frozen answer-free axes on new bracket and noun token-ID swaps."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_gauge_capture import TARGETS
from flagship_later_head_rank_pilot import HeadCut,capture_head_states,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


def proposed_pairs():
    out=[]
    for prefix in ('Try ','Compute ','If '):
        for a,op,b in ((2,'*',3),(4,'+',5),(7,'-',1)):
            out.append(('bracket_closure',f'{prefix}{{{a} {op} {b}',
                        f'{prefix}({a} {op} {b}','}',')'))
            out.append(('bracket_closure',f'{prefix}{{{a} {op} {b}',
                        f'{prefix}[{a} {op} {b}', '}', ']'))
    for noun,plural in (('dog','dogs'),('student','students'),
                        ('child','children'),('horse','horses')):
        for prep,distractor in (('near','trees'),('beside','houses'),
                               ('behind','people')):
            out.append(('subject_verb_agreement',
                f'The {noun} {prep} the {distractor}',
                f'The {plural} {prep} the {distractor}',' is',' are'))
    return out


def token_swap(ids_a,ids_b):
    assert len(ids_a)==len(ids_b)
    changed=[i for i,(a,b) in enumerate(zip(ids_a,ids_b)) if a!=b]
    assert len(changed)==1
    i=changed[0]
    return tuple(sorted((ids_a[i],ids_b[i])))


def margin(model,ids,a,b):
    value=final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:])[0]
    return float(value[a]-value[b])


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    seen={token_swap(r['base_ids'],r['edited_ids'])
          for panel in meta['panels'].values() for r in panel}
    cases=[]
    for family,base_text,edit_text,answer_a,answer_b in proposed_pairs():
        base=tokenizer.encode(base_text,add_special_tokens=False).ids
        edited=tokenizer.encode(edit_text,add_special_tokens=False).ids
        answer_ids=[tokenizer.encode(s,add_special_tokens=False).ids
                    for s in (answer_a,answer_b)]
        assert all(len(ids)==1 for ids in answer_ids)
        answer=[ids[0] for ids in answer_ids]
        swap=token_swap(base,edited)
        assert swap not in seen,(family,swap)
        cases.append(dict(family=family,base_text=base_text,edited_text=edit_text,
                          base_ids=base,edited_ids=edited,answer=answer,token_swap=swap))
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn)
            for layer in sorted(set(layer for layer,_ in TARGETS.values()))}
    bases=np.load(ART/'later_head_rank_pilot_bases.npz')
    rotations=np.load(ART/'later_head_unlabeled_edit_gauge_rotations.npz')
    selection=json.loads((ART/'later_head_unlabeled_edit_gauge.json').read_text())['records']
    rows=[]
    with torch.no_grad():
        for case in cases:
            family=case['family'];layer,head=TARGETS[family]
            p=probes[layer];p.head=head
            base,edited=case['base_ids'],case['edited_ids']
            a,b=case['answer']
            native_a=margin(model,base,a,b)
            native_b=margin(model,edited,a,b)
            z_a=capture_head_states(model,probes,base)[layer][head]
            z_b=capture_head_states(model,probes,edited)[layer][head]
            delta=(z_b-z_a)[None,:]
            p.basis=torch.eye(p.head_dim)
            full_base=margin(model,base,a,b)
            p.injection=delta
            native_path=margin(model,base,a,b)-full_base
            p.injection=None
            basis=torch.tensor(bases[f'{family}__weight'][:,:4].copy(),dtype=torch.float32)
            p.basis=basis
            cut_base=margin(model,base,a,b)
            p.injection=(delta@basis)@basis.T
            rank4_path=margin(model,base,a,b)-cut_base
            p.injection=None
            axes={}
            for panel in ('train','validation'):
                key=f'{family}__{panel}__all__raw_pca'
                r=rotations[key]
                record=next(x for x in selection if x['target']==family and
                    x['panel']==panel and x['condition']=='all' and
                    x['method']=='raw_pca' and x['selection_rule']=='global_energy')
                direction=basis@torch.tensor(r[:,record['selected_axis']].copy(),
                                               dtype=torch.float32)
                p.injection=(delta@direction)*direction[None,:]
                axes[panel]=margin(model,base,a,b)-cut_base
                p.injection=None
            p.basis=None
            rows.append(dict(**case,native_base_margin=native_a,
                native_edited_margin=native_b,
                both_native_correct=bool(native_a>0 and native_b<0),
                native_edit_effect=native_b-native_a,native_head_path=native_path,
                rank4_path=rank4_path,answer_free_axis_paths=axes))
            print(family,len(rows),'correct',rows[-1]['both_native_correct'],
                  'native',round(native_path,3),'rank4',round(rank4_path,3),flush=True)
    summary={}
    for family in TARGETS:
        part=[r for r in rows if r['family']==family]
        native=np.asarray([r['native_head_path'] for r in part])
        cut=np.asarray([r['rank4_path'] for r in part])
        s=dict(n=len(part),distinct_token_swaps=len(set(tuple(r['token_swap']) for r in part)),
            both_native_correct=sum(r['both_native_correct'] for r in part),
            mean_abs_native_head_path=float(np.mean(np.abs(native))),
            rank4_vs_native_relative_mae=float(np.mean(np.abs(cut-native))/
                max(np.mean(np.abs(native)),1e-20)))
        for panel in ('train','validation'):
            axis=np.asarray([r['answer_free_axis_paths'][panel] for r in part])
            s[f'{panel}_axis_vs_rank4_relative_mae']=float(np.mean(np.abs(axis-cut))/
                max(np.mean(np.abs(cut)),1e-20))
            s[f'{panel}_axis_vs_native_relative_mae']=float(np.mean(np.abs(axis-native))/
                max(np.mean(np.abs(native)),1e-20))
        summary[family]=s
    (ART/'unseen_token_edit_transfer.json').write_text(json.dumps(dict(
        description='Predeclared new token-ID substitution pairs, frozen rank4 bases and answer-free raw-PCA axes; model competence unfiltered',
        all_swaps_absent_from_both_gauge_panels=True,summary=summary,rows=rows),indent=2)+'\n')
    print(summary,flush=True)


if __name__=='__main__':run()
