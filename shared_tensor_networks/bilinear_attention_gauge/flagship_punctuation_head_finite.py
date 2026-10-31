"""Finite downstream red team for unlabeled punctuation-head candidates."""

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import HeadCut,final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

LAYER=13
KINDS=('punctuation','number','bracket')
N_PER_KIND=20


def metrics(effect,target):
    a=effect.double().numpy();b=target.double().numpy()
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return dict(effect_l2=float(na),target_l2=float(nb),
        cosine=float(np.dot(a,b)/(na*nb)) if na*nb>1e-20 else None,
        signed_projection=float(np.dot(a,b)/(nb*nb)) if nb>1e-20 else None,
        relative_l2=float(na/nb) if nb>1e-20 else None)


def run():
    torch.set_num_threads(4)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    with np.load(ART/'all_head_unlabeled_edit_scan.npz') as saved:
        train=saved[f'layer{LAYER}__train'].copy()
        validation=saved[f'layer{LAYER}__validation'].copy()
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    p=HeadCut(model.transformer.h[LAYER].attn)
    p.basis=torch.eye(p.head_dim)
    p.head=0
    axes=[]
    for head in range(p.n_head):
        eig,vec=np.linalg.eigh(train[:,head,:].astype(np.float64).T@
                              train[:,head,:].astype(np.float64)/len(train))
        axes.append(torch.tensor(vec[:,-1].copy(),dtype=torch.float32))
    indices=[]
    cases=meta['panels']['validation']
    for kind in KINDS:
        indices.extend([i for i,c in enumerate(cases) if c['type']==kind][:N_PER_KIND])
    rows=[]
    with torch.no_grad():
        for j,i in enumerate(indices):
            case=cases[i]
            before=torch.tensor(case['base_ids'],dtype=torch.long)[None,:]
            after=torch.tensor(case['edited_ids'],dtype=torch.long)[None,:]
            # The identity basis is an exact full-head projector. It is kept
            # active in all reference and patched calls to cancel roundoff.
            base=final_logits(model,before)[0].clone()
            edited=final_logits(model,after)[0].clone()
            target=edited-base
            for head in range(p.n_head):
                delta=torch.tensor(validation[i,head],dtype=torch.float32)
                axis=axes[head]
                p.head=head
                effects={}
                for variant,add in [('full_head',delta),
                                    ('top_axis',(delta@axis)*axis)]:
                    p.injection=add[None,:]
                    logits=final_logits(model,before)[0].clone()
                    p.injection=None
                    effects[variant]=metrics(logits-base,target)
                rows.append(dict(case_index=i,kind=case['type'],head=head,
                                 full_head=effects['full_head'],
                                 top_axis=effects['top_axis']))
            if (j+1)%10==0:print('done',j+1,'/',len(indices),flush=True)
    summary=[]
    for kind in KINDS:
        for head in range(p.n_head):
            part=[r for r in rows if r['kind']==kind and r['head']==head]
            d=dict(kind=kind,head=head,n=len(part))
            for variant in ('full_head','top_axis'):
                for key in ('cosine','signed_projection','relative_l2'):
                    vals=[r[variant][key] for r in part if r[variant][key] is not None]
                    d[f'{variant}__mean_{key}']=float(np.mean(vals))
                # Signed projection is the fraction of total logit edit in
                # its own direction, averaged per case. It is not necessity.
            summary.append(d)
    (ART/'punctuation_head_finite.json').write_text(json.dumps(dict(
        description='Frozen unlabeled PCA axes and native full-head edit vectors transplanted into base prompts; full-vocabulary finite logit response',
        layer=LAYER,kinds=KINDS,n_per_kind=N_PER_KIND,
        caveat='Input edits are independent validation panel but use same designed edit types; signed projections are path sufficiency, not necessity or rule discovery.',
        summary=summary,rows=rows),indent=2)+'\n')
    for r in summary:
        if r['head'] in (3,5,8):print(r,flush=True)


if __name__=='__main__':run()
