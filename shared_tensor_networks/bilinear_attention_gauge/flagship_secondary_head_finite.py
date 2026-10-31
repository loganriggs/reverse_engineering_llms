"""Finite-path gate for answer-free layer-11 bracket-response candidates."""

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


TARGETS=((11,3),(11,5),(11,6),(13,8))
KINDS=('bracket','noun_number','punctuation')
N_PER_KIND=20


def metrics(effect,target):
    a=effect.double().numpy();b=target.double().numpy()
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return dict(cosine=float(a@b/(na*nb)) if na*nb>1e-20 else None,
        signed_projection=float(a@b/(nb*nb)) if nb>1e-20 else None,
        relative_l2=float(na/nb) if nb>1e-20 else None,
        effect_l2=float(na),target_l2=float(nb))


def run():
    torch.set_num_threads(4)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    with np.load(ART/'all_head_unlabeled_edit_scan.npz') as source:
        arrays={layer:{part:source[f'layer{layer}__{part}'].copy()
                       for part in ('train','validation')}
                for layer in (11,13)}
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn) for layer in (11,13)}
    axes={}
    for layer,head in TARGETS:
        x=arrays[layer]['train'][:,head].astype(np.float64)
        eig,vec=np.linalg.eigh(x.T@x/len(x))
        axes[(layer,head)]=torch.tensor(vec[:,-1].copy(),dtype=torch.float32)
    cases=meta['panels']['validation']
    indices=[i for kind in KINDS for i in
             [j for j,c in enumerate(cases) if c['type']==kind][:N_PER_KIND]]
    rows=[]
    with torch.no_grad():
        for j,i in enumerate(indices):
            case=cases[i]
            before=torch.tensor(case['base_ids'],dtype=torch.long)[None,:]
            after=torch.tensor(case['edited_ids'],dtype=torch.long)[None,:]
            native_base=final_logits(model,before)[0].clone()
            native_edited=final_logits(model,after)[0].clone()
            for layer,head in TARGETS:
                p=probes[layer]
                p.head=head;p.basis=torch.eye(p.head_dim)
                baseline=final_logits(model,before)[0].clone()
                edited=final_logits(model,after)[0].clone()
                roundtrip=max(float((baseline-native_base).abs().max()),
                              float((edited-native_edited).abs().max()))
                assert roundtrip<1e-3,roundtrip
                target=edited-baseline
                delta=torch.tensor(arrays[layer]['validation'][i,head],dtype=torch.float32)
                axis=axes[(layer,head)]
                variants={'full_head':delta,'top_axis':(delta@axis)*axis}
                effects={}
                for name,inject in variants.items():
                    p.injection=inject[None,:]
                    patched=final_logits(model,before)[0].clone()
                    p.injection=None
                    effects[name]=metrics(patched-baseline,target)
                p.basis=None
                rows.append(dict(case_index=i,kind=case['type'],layer=layer,
                    head=head,identity_roundtrip_max_abs=roundtrip,
                    native_target_l2=float((native_edited-native_base).norm()),
                    full_head=effects['full_head'],top_axis=effects['top_axis']))
            if (j+1)%10==0:print('done',j+1,'/',len(indices),flush=True)
    summary=[]
    for kind in KINDS:
        for layer,head in TARGETS:
            subset=[r for r in rows if r['kind']==kind and r['layer']==layer
                    and r['head']==head]
            row=dict(kind=kind,layer=layer,head=head,n=len(subset))
            for name in ('full_head','top_axis'):
                for key in ('cosine','signed_projection','relative_l2'):
                    values=[r[name][key] for r in subset if r[name][key] is not None]
                    row[f'{name}__mean_{key}']=float(np.mean(values))
            summary.append(row)
    result=dict(description='Frozen unlabeled edit-PCA axes and full native head-response patches on independent validation edits, 20 per type; final-vocabulary finite effects',
        targets=TARGETS,kinds=KINDS,n_per_kind=N_PER_KIND,
        caveat='Same edit families and designed templates as fitting panel; path sufficiency, not necessity or independent task discovery.',
        summary=summary,rows=rows)
    (ART/'secondary_head_finite.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in summary:print(row,flush=True)


if __name__=='__main__':run()
