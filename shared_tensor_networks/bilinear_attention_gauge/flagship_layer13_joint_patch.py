"""Native multi-head edit transplants to test distribution and cancellation."""

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

KINDS=('punctuation','number','bracket')
N_PER_KIND=20
HEAD_SETS={'head3':(3,), 'head5':(5,), 'heads3_5':(3,5),
           'head8':(8,), 'all9':tuple(range(9))}


class HeadInjection:
    def __init__(self,attention):
        self.native=attention.c_proj.forward
        self.n_head=attention.n_head
        self.head_dim=attention.head_dim
        self.delta=None
        attention.c_proj.forward=self.forward

    def forward(self,x):
        if self.delta is not None:
            x=x.clone()
            x[:,-1,:]+=self.delta
        return self.native(x)


def metrics(effect,target):
    a=effect.double().numpy();b=target.double().numpy()
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    return dict(effect_l2=float(na),target_l2=float(nb),
                cosine=float(a@b/(na*nb)) if na*nb>1e-20 else None,
                signed_projection=float(a@b/(nb*nb)) if nb>1e-20 else None)


def run():
    torch.set_num_threads(4)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    with np.load(ART/'all_head_unlabeled_edit_scan.npz') as saved:
        val=saved['layer13__validation'].copy()
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    patch=HeadInjection(model.transformer.h[13].attn)
    cases=meta['panels']['validation']
    indices=[]
    for kind in KINDS:
        indices.extend([i for i,c in enumerate(cases) if c['type']==kind][:N_PER_KIND])
    records=[]
    with torch.no_grad():
        for j,i in enumerate(indices):
            before=torch.tensor(cases[i]['base_ids'],dtype=torch.long)[None,:]
            after=torch.tensor(cases[i]['edited_ids'],dtype=torch.long)[None,:]
            base=final_logits(model,before)[0].clone()
            edited=final_logits(model,after)[0].clone()
            target=edited-base
            effects={}
            for name,heads in HEAD_SETS.items():
                delta=torch.zeros((1,patch.n_head,patch.head_dim),dtype=torch.float32)
                delta[:,list(heads),:]=torch.tensor(val[i,list(heads)],dtype=torch.float32)
                patch.delta=delta.reshape(1,-1)
                effects[name]=final_logits(model,before)[0].clone()-base
                patch.delta=None
            interaction=effects['heads3_5']-effects['head3']-effects['head5']
            records.append(dict(case_index=i,kind=cases[i]['type'],
                patches={k:metrics(v,target) for k,v in effects.items()},
                heads3_5_interaction_relative_l2=float(
                    torch.linalg.vector_norm(interaction)/
                    torch.linalg.vector_norm(target).clamp_min(1e-20))))
            if (j+1)%10==0:print('done',j+1,'/',len(indices),flush=True)
    summary=[]
    for kind in KINDS:
        part=[r for r in records if r['kind']==kind]
        row=dict(kind=kind,n=len(part),
            mean_heads3_5_interaction_relative_l2=float(np.mean(
                [r['heads3_5_interaction_relative_l2'] for r in part])))
        for name in HEAD_SETS:
            row[name]={k:float(np.mean([r['patches'][name][k] for r in part]))
                       for k in ('signed_projection','cosine','effect_l2','target_l2')}
        summary.append(row)
    (ART/'layer13_joint_patch.json').write_text(json.dumps(dict(
        description='Native layer13 finite multihead transplantation on validation input edits',
        head_sets=HEAD_SETS,kinds=KINDS,n_per_kind=N_PER_KIND,
        summary=summary,records=records),indent=2)+'\n')
    for r in summary:print(r,flush=True)


if __name__=='__main__':run()
