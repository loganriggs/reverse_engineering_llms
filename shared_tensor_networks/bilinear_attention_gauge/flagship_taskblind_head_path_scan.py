"""Label-free all-head ranking by finite full-vocabulary edit-path alignment."""

import json

import numpy as np
import torch

from flagship_all_head_unlabeled_edit_scan import LAYERS
from flagship_full_model_eval import GPT,GPTConfig
from flagship_later_head_rank_pilot import final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


N_PER_TYPE=12
TYPES=('bracket','noun_number','adjective','color','country','name',
       'number','punctuation','preposition','random_context')


class BatchHeadPatch:
    def __init__(self,attention):
        self.native=attention.c_proj.forward
        self.n_head=attention.n_head
        self.head_dim=attention.head_dim
        self.injection=None
        attention.c_proj.forward=self.forward

    def forward(self,x):
        if self.injection is not None:
            assert x.shape[0]==self.n_head
            assert self.injection.shape==(self.n_head,self.n_head*self.head_dim)
            x=x.clone()
            x[:,-1,:]+=self.injection
        return self.native(x)


def choose(cases):
    indices=[]
    for kind in TYPES:
        values=[i for i,x in enumerate(cases) if x['type']==kind]
        assert len(values)==80
        indices.extend(values[:N_PER_TYPE])
    return indices


def run():
    torch.set_num_threads(4)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    saved=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:BatchHeadPatch(model.transformer.h[layer].attn)
            for layer in LAYERS}
    assert {p.n_head for p in probes.values()}=={9}
    rows=[]
    with torch.no_grad():
        for panel,cases in meta['panels'].items():
            indices=choose(cases)
            for position,i in enumerate(indices):
                case=cases[i]
                before=torch.tensor(case['base_ids'],dtype=torch.long)[None,:]
                after=torch.tensor(case['edited_ids'],dtype=torch.long)[None,:]
                base=final_logits(model,before)[0].double()
                edited=final_logits(model,after)[0].double()
                target=edited-base
                target_norm=float(torch.linalg.norm(target))
                assert target_norm>1e-8
                for layer,p in probes.items():
                    delta=saved[f'layer{layer}__{panel}'][i]
                    injection=np.zeros((p.n_head,p.n_head*p.head_dim),dtype=np.float32)
                    for head in range(p.n_head):
                        injection[head,head*p.head_dim:(head+1)*p.head_dim]=delta[head]
                    p.injection=torch.tensor(injection)
                    patched=final_logits(model,before.repeat(p.n_head,1)).double()
                    p.injection=None
                    effect=patched-base[None,:]
                    effect_norm=torch.linalg.norm(effect,dim=-1)
                    aligned=(effect@target)/(target_norm**2)
                    cosine=(effect@target)/(effect_norm*target_norm).clamp_min(1e-30)
                    for head in range(p.n_head):
                        rows.append(dict(panel=panel,case_index=i,type=case['type'],
                            layer=layer,head=head,target_l2=target_norm,
                            head_state_delta_l2=float(np.linalg.norm(delta[head])),
                            path_l2=float(effect_norm[head]),
                            path_cosine=float(cosine[head]),
                            signed_projection=float(aligned[head])))
                if (position+1)%40==0:
                    print(panel,'finite paths',position+1,'/',len(indices),flush=True)
    summary=[]
    for panel in meta['panels']:
        for layer in LAYERS:
            for head in range(9):
                part=[x for x in rows if x['panel']==panel and
                      x['layer']==layer and x['head']==head]
                assert len(part)==len(TYPES)*N_PER_TYPE
                summary.append(dict(panel=panel,layer=layer,head=head,
                    mean_signed_projection=float(np.mean([x['signed_projection'] for x in part])),
                    mean_path_l2=float(np.mean([x['path_l2'] for x in part])),
                    mean_head_state_delta_l2=float(np.mean([
                        x['head_state_delta_l2'] for x in part])),
                    mean_by_type={kind:dict(
                        signed_projection=float(np.mean([
                            x['signed_projection'] for x in part if x['type']==kind])),
                        path_l2=float(np.mean([
                            x['path_l2'] for x in part if x['type']==kind])))
                        for kind in TYPES}))
    result=dict(description='All 36 heads in layers 9/11/13/15, 12 train and 12 independent validation edits per each of ten designed types: patch every native head delta, measure full-vocabulary finite path alignment with whole input-edit output difference, rank without answer labels',
        layers=LAYERS,types=TYPES,n_per_type=N_PER_TYPE,
        head_selection_uses_task_answer_labels=False,
        input_edit_distribution_is_designed_and_balanced=True,
        rows=rows,summary=summary)
    (ART/'taskblind_head_path_scan.json').write_text(json.dumps(result,indent=2)+'\n')
    for panel in ('train','validation'):
        ranked=sorted((x for x in summary if x['panel']==panel),
                      key=lambda x:x['mean_signed_projection'],reverse=True)
        print(panel,'top eight',[(x['layer'],x['head'],round(x['mean_signed_projection'],4))
                                 for x in ranked[:8]],flush=True)


if __name__=='__main__':run()
