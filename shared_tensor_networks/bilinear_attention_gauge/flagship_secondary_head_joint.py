"""Joint finite patch of the two layer-11 bracket-response heads."""

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


HEADS=(5,6)
KINDS=('bracket','noun_number','punctuation')
N_PER_KIND=20


class MultiHeadInjection:
    def __init__(self,attention):
        self.native=attention.c_proj.forward
        self.width=attention.head_dim
        self.injections={}
        attention.c_proj.forward=self.forward

    def forward(self,x):
        if self.injections:
            x=x.clone()
            for head,delta in self.injections.items():
                start=head*self.width
                x[:,-1,start:start+self.width]+=delta
        return self.native(x)


def projection(effect,target):
    a=effect.double().numpy();b=target.double().numpy()
    return float(a@b/max(b@b,1e-30))


def run():
    torch.set_num_threads(4)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    with np.load(ART/'all_head_unlabeled_edit_scan.npz') as source:
        train=source['layer11__train'].copy()
        validation=source['layer11__validation'].copy()
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    patch=MultiHeadInjection(model.transformer.h[11].attn)
    axes={}
    for head in HEADS:
        x=train[:,head].astype(np.float64)
        _,vec=np.linalg.eigh(x.T@x/len(x))
        axes[head]=torch.tensor(vec[:,-1].copy(),dtype=torch.float32)
    cases=meta['panels']['validation']
    indices=[i for kind in KINDS for i in
             [j for j,c in enumerate(cases) if c['type']==kind][:N_PER_KIND]]
    rows=[]
    with torch.no_grad():
        for j,i in enumerate(indices):
            case=cases[i]
            before=torch.tensor(case['base_ids'],dtype=torch.long)[None,:]
            after=torch.tensor(case['edited_ids'],dtype=torch.long)[None,:]
            base=final_logits(model,before)[0].clone()
            target=final_logits(model,after)[0].clone()-base
            for form in ('full','axis'):
                deltas={head:torch.tensor(validation[i,head],dtype=torch.float32)
                        for head in HEADS}
                if form=='axis':
                    deltas={head:(delta@axes[head])*axes[head]
                            for head,delta in deltas.items()}
                effects={}
                for name,heads in (('head5',(5,)),('head6',(6,)),('joint',HEADS)):
                    patch.injections={head:deltas[head][None,:] for head in heads}
                    effects[name]=final_logits(model,before)[0].clone()-base
                patch.injections={}
                projections={name:projection(effect,target)
                             for name,effect in effects.items()}
                interaction=effects['joint']-effects['head5']-effects['head6']
                rows.append(dict(case_index=i,kind=case['type'],form=form,
                    projections=projections,
                    interaction_relative_l2=float(interaction.norm()/max(target.norm(),1e-30)),
                    joint_minus_sum_projection=projections['joint']-
                        projections['head5']-projections['head6']))
            if (j+1)%10==0:print('done',j+1,'/',len(indices),flush=True)
    summary=[]
    rng=np.random.default_rng(20261114)
    for kind in KINDS:
        for form in ('full','axis'):
            subset=[r for r in rows if r['kind']==kind and r['form']==form]
            indices=rng.integers(0,len(subset),(10000,len(subset)))
            joint=np.asarray([r['projections']['joint'] for r in subset])
            interaction=np.asarray([r['joint_minus_sum_projection'] for r in subset])
            summary.append(dict(kind=kind,form=form,n=len(subset),
                mean_projection={name:float(np.mean([r['projections'][name] for r in subset]))
                                 for name in ('head5','head6','joint')},
                mean_joint_minus_sum_projection=float(np.mean([
                    r['joint_minus_sum_projection'] for r in subset])),
                joint_projection_bootstrap95=np.quantile(
                    joint[indices].mean(axis=1),[.025,.975]).tolist(),
                interaction_projection_bootstrap95=np.quantile(
                    interaction[indices].mean(axis=1),[.025,.975]).tolist(),
                mean_interaction_relative_l2=float(np.mean([
                    r['interaction_relative_l2'] for r in subset]))))
    result=dict(description='Simultaneous layer-11 head 5+6 native-delta patches, comparing sum of separate finite effects to joint finite effect on independent validation edits',
        heads=HEADS,kinds=KINDS,n_per_kind=N_PER_KIND,
        caveat='Sufficiency-style final-token patch; same input edit families as axis fit; not path necessity.',
        summary=summary,rows=rows)
    (ART/'secondary_head_joint.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in summary:print(row,flush=True)


if __name__=='__main__':run()
