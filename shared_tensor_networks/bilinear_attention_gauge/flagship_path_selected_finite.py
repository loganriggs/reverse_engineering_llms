"""Finite full-model check of train-path-selected rank cuts on held-out edits."""

import json

import numpy as np
import torch

from flagship_all_head_rank_frontier import top_basis
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

LAYER,HEAD,RANK=9,7,32
N_PER_TYPE=12
FLOOR=.1


def run():
    torch.set_num_threads(4)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    path=json.loads((ART/'taskblind_head_path_scan.json').read_text())
    scan=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    train=scan[f'layer{LAYER}__train'][:,HEAD].astype(np.float64)
    val=scan[f'layer{LAYER}__validation'][:,HEAD].astype(np.float64)
    train_types=np.array([r['type'] for r in meta['panels']['train']])
    names=sorted(set(train_types))
    val_types=np.array([r['type'] for r in meta['panels']['validation']])
    summary=next(r for r in path['summary'] if
                 (r['panel'],r['layer'],r['head'])==('train',LAYER,HEAD))
    chosen=sorted(names,key=lambda name:-summary['mean_by_type'][name]['path_l2'])[:3]
    heldout='bracket'
    assert heldout not in chosen
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[LAYER].attn)
    probe.head=HEAD
    projection=model.transformer.h[LAYER].attn.c_proj.weight.detach().numpy()
    weight=projection[:,HEAD*128:(HEAD+1)*128].astype(np.float64)
    gram=weight.T@weight
    def normalized(name):
        x=train[train_types==name]
        cov=x.T@x/len(x)
        return cov/max(np.trace(cov@gram),1e-30)
    balanced=sum(normalized(name) for name in names)/len(names)
    targeted=(1-FLOOR)*sum(normalized(name) for name in chosen)/len(chosen)+FLOOR*balanced
    moments={'pooled_edit':train.T@train/len(train),
             'balanced_all_types':balanced,
             'top_three_path_with_floor':targeted}
    bases={name:torch.tensor(top_basis(cov/np.trace(cov))[:,:RANK].copy(),
                              dtype=torch.float32) for name,cov in moments.items()}
    indices=[]
    for kind in [*chosen,heldout]:
        indices+=list(np.flatnonzero(val_types==kind)[:N_PER_TYPE])
    rows=[]
    with torch.no_grad():
        for index in indices:
            case=meta['panels']['validation'][index]
            ids=torch.tensor(case['base_ids'],dtype=torch.long)[None,:]
            delta=torch.tensor(val[index],dtype=torch.float32)
            probe.basis=torch.eye(128);probe.injection=None
            native_base=final_logits(model,ids)[0].double()
            probe.injection=delta
            native_patched=final_logits(model,ids)[0].double()
            probe.injection=None
            native_effect=native_patched-native_base
            denom=native_effect.square().sum().item()
            assert denom>1e-12
            for name,basis in bases.items():
                probe.basis=basis
                cut_base=final_logits(model,ids)[0].double()
                p=torch.log_softmax(native_base,dim=-1)
                q=torch.log_softmax(cut_base,dim=-1)
                base_kl=float((p.exp()*(p-q)).sum())
                probe.injection=(delta@basis)@basis.T
                cut_patched=final_logits(model,ids)[0].double()
                probe.injection=None
                effect=cut_patched-cut_base
                rows.append(dict(index=int(index),type=case['type'],method=name,
                    relative_path_mse=float((effect-native_effect).square().sum()/denom),
                    cosine=float((effect@native_effect)/
                                 (effect.norm()*native_effect.norm()).clamp_min(1e-30)),
                    path_norm_ratio=float(effect.norm()/native_effect.norm()),
                    base_forward_kl=base_kl,
                    native_path_l2=float(native_effect.norm())))
            probe.basis=None
            print('case',index,case['type'],flush=True)
    result=dict(description='L9H7 rank32 finite full-model path check. Twelve held-out validation cases from each of train-path top-three edit types plus bracket as unselected guardrail. Each method projects base head state and edit delta, then compares patched-minus-base final logits to native patched-minus-base final logits.',
                layer=LAYER,head=HEAD,rank=RANK,n_per_type=N_PER_TYPE,
                train_selected_types=chosen,unselected_guardrail=heldout,
                fit_uses_answer_labels=False,rows=rows)
    (ART/'path_selected_finite.json').write_text(json.dumps(result,indent=2)+'\n')
    for kind in [*chosen,heldout]:
        for method in moments:
            a=[r['relative_path_mse'] for r in rows if r['type']==kind and r['method']==method]
            kl=[r['base_forward_kl'] for r in rows if r['type']==kind and r['method']==method]
            print(kind,method,round(float(np.mean(a)),4),
                  'base KL',round(float(np.mean(kl)),4),flush=True)


if __name__=='__main__':run()
