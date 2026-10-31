"""Choose L9H7 rank and fit measure from training-only finite paths."""

import json

import numpy as np
import torch

from flagship_all_head_rank_frontier import top_basis
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

LAYER,HEAD=9,7
RANKS=(32,64,96,112)
N_PER_TYPE=4
MAX_FAMILY_PATH_MSE=.10
MAX_MEAN_BASE_KL=.001


def run():
    torch.set_num_threads(4)
    metadata=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    path=json.loads((ART/'taskblind_head_path_scan.json').read_text())
    scan=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    cases=metadata['panels']['train']
    types=np.array([r['type'] for r in cases])
    names=sorted(set(types))
    indices=[int(i) for name in names for i in np.flatnonzero(types==name)[:N_PER_TYPE]]
    train=scan[f'layer{LAYER}__train'][:,HEAD].astype(np.float64)
    finite_delta=scan[f'layer{LAYER}__train'][:,HEAD]
    summary=next(r for r in path['summary'] if
                 (r['layer'],r['head'],r['panel'])==(LAYER,HEAD,'train'))
    chosen=sorted(names,key=lambda name:-summary['mean_by_type'][name]['path_l2'])[:3]
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[LAYER].attn);probe.head=HEAD
    projection=model.transformer.h[LAYER].attn.c_proj.weight.detach().numpy()
    weight=projection[:,HEAD*128:(HEAD+1)*128].astype(np.float64)
    gram=weight.T@weight
    def normalized(name):
        x=train[types==name];cov=x.T@x/len(x)
        return cov/max(np.trace(cov@gram),1e-30)
    balanced=sum(normalized(name) for name in names)/len(names)
    targeted=sum(normalized(name) for name in chosen)/len(chosen)
    moments={'weight_only':gram,'pooled_edit':train.T@train/len(train),
             'selected_floor_0.1':.9*targeted+.1*balanced,
             'selected_floor_0.5':.5*targeted+.5*balanced,
             'balanced_all':balanced}
    bases={name:top_basis(moment/np.trace(moment))
           for name,moment in moments.items()}
    batches=[]
    for length in sorted({len(cases[i]['base_ids']) for i in indices}):
        for start in range(0,len([i for i in indices if len(cases[i]['base_ids'])==length]),8):
            group=[i for i in indices if len(cases[i]['base_ids'])==length][start:start+8]
            batches.append(group)
    reference={}
    identity=torch.eye(128)
    with torch.no_grad():
        for group in batches:
            before=torch.tensor([cases[i]['base_ids'] for i in group],dtype=torch.long)
            delta=torch.tensor(finite_delta[group],dtype=torch.float32)
            probe.basis=identity;probe.injection=None
            native_base=final_logits(model,before).double()
            probe.injection=delta
            native_patched=final_logits(model,before).double()
            probe.injection=None
            native_effect=native_patched-native_base
            for j,i in enumerate(group):
                reference[i]=(native_base[j].clone(),native_effect[j].clone())
        rows=[]
        for rank in RANKS:
            for name,full_basis in bases.items():
                basis=torch.tensor(full_basis[:,:rank].copy(),dtype=torch.float32)
                probe.basis=basis
                for group in batches:
                    before=torch.tensor([cases[i]['base_ids'] for i in group],dtype=torch.long)
                    delta=torch.tensor(finite_delta[group],dtype=torch.float32)
                    cut_base=final_logits(model,before).double()
                    probe.injection=(delta@basis)@basis.T
                    cut_patched=final_logits(model,before).double()
                    probe.injection=None
                    effect=cut_patched-cut_base
                    native_base=torch.stack([reference[i][0] for i in group])
                    native_effect=torch.stack([reference[i][1] for i in group])
                    path_mse=(effect-native_effect).square().sum(-1)/native_effect.square().sum(-1)
                    p=torch.log_softmax(native_base,dim=-1)
                    q=torch.log_softmax(cut_base,dim=-1)
                    kl=(p.exp()*(p-q)).sum(-1)
                    for j,i in enumerate(group):
                        rows.append(dict(rank=rank,method=name,index=i,type=cases[i]['type'],
                            relative_path_mse=float(path_mse[j]),
                            base_forward_kl=float(kl[j]),
                            native_path_l2=float(native_effect[j].norm())))
                print('candidate',rank,name,flush=True)
        probe.basis=None
    summary_rows=[]
    for rank in RANKS:
        for name in moments:
            part=[r for r in rows if (r['rank'],r['method'])==(rank,name)]
            family={kind:float(np.mean([r['relative_path_mse'] for r in part
                                        if r['type']==kind])) for kind in names}
            max_family=max(family.values())
            mean_kl=float(np.mean([r['base_forward_kl'] for r in part]))
            summary_rows.append(dict(rank=rank,method=name,
                family_path_mse=family,max_family_path_mse=max_family,
                mean_base_forward_kl=mean_kl,
                pass_gate=max_family<=MAX_FAMILY_PATH_MSE and
                          mean_kl<=MAX_MEAN_BASE_KL))
    passing=[r for r in summary_rows if r['pass_gate']]
    selected=min(passing,key=lambda r:(r['rank'],r['max_family_path_mse'],
                                      r['mean_base_forward_kl'])) if passing else None
    result=dict(description='Training-only downstream gate for L9H7. Four cases per ten familiar edit types used to choose from four ranks and five spectral fit measures. Every designed type must have mean finite final-logit path relative MSE <= .10, and mean base-prompt forward KL <= .001. The minimal passing rank wins; within rank, minimize worst family path error. New edit families and validation cases are excluded from selection. Training cases also contributed to covariance fitting, so independent evaluation is required.',
        layer=LAYER,head=HEAD,ranks=RANKS,methods=list(moments),
        n_per_type=N_PER_TYPE,thresholds=dict(max_family_path_mse=MAX_FAMILY_PATH_MSE,
                                             mean_base_forward_kl=MAX_MEAN_BASE_KL),
        path_selected_types=chosen,selected=selected,summary=summary_rows,rows=rows)
    (ART/'train_finite_fidelity_gate.json').write_text(json.dumps(result,
                                                          separators=(',',':'))+'\n')
    for r in summary_rows:
        print(r['rank'],r['method'],'worst',round(r['max_family_path_mse'],4),
              'KL',round(r['mean_base_forward_kl'],5),'pass',r['pass_gate'])
    print('selected',selected,flush=True)


if __name__=='__main__':run()
