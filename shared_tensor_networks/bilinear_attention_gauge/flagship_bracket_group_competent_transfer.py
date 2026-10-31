"""Frozen one- versus two-axis bracket group on competent unseen token IDs."""

import json

import numpy as np
import torch
from tokenizers import Tokenizer

from flagship_explicit_task_screen import TOKENIZER
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


def swap(a,b):
    assert len(a)==len(b)
    values=[tuple(sorted((x,y))) for x,y in zip(a,b) if x!=y]
    assert len(values)==1
    return values[0]


def margin(model,ids,a,b):
    z=final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:])[0]
    return float(z[a]-z[b])


def state(model,probe,ids):
    final_logits(model,torch.tensor(ids,dtype=torch.long)[None,:],stop_at=13)
    start=8*probe.head_dim
    return probe.last_input[0,-1,start:start+probe.head_dim].clone()


def run():
    torch.set_num_threads(4)
    tokenizer=Tokenizer.from_file(TOKENIZER)
    candidates=json.loads((ART/'competent_bracket_identities.json').read_text())
    tests=[x for x in candidates['rows'] if x['panel']=='validation']
    fit=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    original={swap(x['base_ids'],x['edited_ids']) for x in fit['panels']['train']
              if x['type']=='bracket'}
    unseen=json.loads((ART/'unseen_token_edit_transfer.json').read_text())
    brace={swap(x['base_ids'],x['edited_ids']) for x in unseen['rows']
           if x['family']=='bracket_closure' and
           not x['base_text'].startswith('If ')}
    fitted=original|brace
    assert all(swap(x['base_ids'],x['edited_ids']) not in fitted for x in tests)
    with np.load(ART/'bracket_edit_span_rotations.npz') as source:
        rotations={name:source[name].copy() for name in source.files}
    basis_array=np.load(ART/'later_head_rank_pilot_bases.npz')
    basis=torch.tensor(basis_array['bracket_closure__weight'][:,:4].copy(),dtype=torch.float32)
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state_dict=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state_dict);del state_dict
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[13].attn);probe.head=8
    a=tokenizer.encode(')',add_special_tokens=False).ids[0]
    b=tokenizer.encode(']',add_special_tokens=False).ids[0]
    rows=[]
    with torch.no_grad():
        for i,case in enumerate(tests):
            before=case['base_ids'];after=case['edited_ids']
            delta=(state(model,probe,after)-state(model,probe,before))[None,:]
            probe.basis=basis
            baseline=margin(model,before,a,b)
            probe.injection=(delta@basis)@basis.T
            rank4=margin(model,before,a,b)-baseline
            probe.injection=None
            assert abs(rank4-case['rank4_path'])<1e-4
            effects={}
            for name,rotation in rotations.items():
                subspace=basis@torch.tensor(rotation.copy(),dtype=torch.float32)
                probe.injection=(delta@subspace)@subspace.T
                effects[name]=margin(model,before,a,b)-baseline
                probe.injection=None
            probe.basis=None
            rows.append(dict(family=case['family'],base_text=case['base_text'],
                edited_text=case['edited_text'],token_swap=case['token_swap'],
                rank4_path=rank4,effects=effects))
            if (i+1)%7==0:print('done',i+1,'/',len(tests),flush=True)
    summary=[]
    for family in list(dict.fromkeys(x['family'] for x in rows))+['all']:
        group=rows if family=='all' else [x for x in rows if x['family']==family]
        denominator=np.mean([abs(x['rank4_path']) for x in group])
        summary.append(dict(family=family,n=len(group),
            mean_abs_rank4_path=float(denominator),
            relative_mae={name:float(np.mean([abs(x['effects'][name]-x['rank4_path'])
                         for x in group])/max(denominator,1e-20))
                          for name in rotations}))
    result=dict(description='Frozen rank-one and rank-two edit-response groups at fixed weight-derived rank-four head cut, tested on six competent token-ID families absent from group fitting',
        group_fit_token_swaps=[list(x) for x in sorted(fitted)],
        all_test_swaps_absent_from_group_fit=True,
        summary=summary,rows=rows)
    (ART/'bracket_group_competent_transfer.json').write_text(json.dumps(result,indent=2)+'\n')
    for item in summary:print(item,flush=True)


if __name__=='__main__':run()
