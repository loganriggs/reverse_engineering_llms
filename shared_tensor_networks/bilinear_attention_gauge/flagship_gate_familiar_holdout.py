"""Finite full-model validation of the frozen training-only L9H7 gate."""

import json

import numpy as np
import torch

from flagship_all_head_rank_frontier import top_basis
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

N_PER_TYPE=12


def run():
    torch.set_num_threads(4)
    gate=json.loads((ART/'train_finite_fidelity_gate.json').read_text())
    selected=gate['selected']
    assert selected is not None
    layer,head=gate['layer'],gate['head']
    rank,method=selected['rank'],selected['method']
    assert method=='pooled_edit'
    metadata=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    scan=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    train=scan[f'layer{layer}__train'][:,head].astype(np.float64)
    validation=scan[f'layer{layer}__validation'][:,head]
    basis=torch.tensor(top_basis(train.T@train/len(train))[:,:rank].copy(),
                       dtype=torch.float32)
    cases=metadata['panels']['validation']
    types=np.array([r['type'] for r in cases])
    names=sorted(set(types))
    indices=[int(i) for name in names for i in np.flatnonzero(types==name)[:N_PER_TYPE]]
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[layer].attn);probe.head=head
    batches=[]
    for length in sorted({len(cases[i]['base_ids']) for i in indices}):
        group=[i for i in indices if len(cases[i]['base_ids'])==length]
        batches.extend(group[start:start+8] for start in range(0,len(group),8))
    rows=[]
    with torch.no_grad():
        for group in batches:
            before=torch.tensor([cases[i]['base_ids'] for i in group],dtype=torch.long)
            delta=torch.tensor(validation[group],dtype=torch.float32)
            probe.basis=torch.eye(128);probe.injection=None
            native_base=final_logits(model,before).double()
            probe.injection=delta
            native_patched=final_logits(model,before).double()
            probe.injection=None
            native_effect=native_patched-native_base
            probe.basis=basis
            cut_base=final_logits(model,before).double()
            probe.injection=(delta@basis)@basis.T
            cut_patched=final_logits(model,before).double()
            probe.injection=None
            cut_effect=cut_patched-cut_base
            path_mse=(cut_effect-native_effect).square().sum(-1)/native_effect.square().sum(-1)
            p=torch.log_softmax(native_base,dim=-1)
            q=torch.log_softmax(cut_base,dim=-1)
            kl=(p.exp()*(p-q)).sum(-1)
            for j,i in enumerate(group):
                rows.append(dict(index=i,type=cases[i]['type'],
                    relative_path_mse=float(path_mse[j]),
                    base_forward_kl=float(kl[j]),
                    native_path_l2=float(native_effect[j].norm())))
            print('batch',len(rows),'/',len(indices),flush=True)
        probe.basis=None
    family={name:float(np.mean([r['relative_path_mse'] for r in rows
                                if r['type']==name])) for name in names}
    result=dict(description='Frozen training-only gate evaluated on 12 independent validation edits per each of ten familiar designed types. Rank and fit selected before viewing validation or new families. Native versus reduced head-patch finite final-logit path MSE and base-prompt forward KL. No answer labels.',
        layer=layer,head=head,rank=rank,method=method,n_per_type=N_PER_TYPE,
        thresholds=gate['thresholds'],family_path_mse=family,
        max_family_path_mse=max(family.values()),
        mean_base_forward_kl=float(np.mean([r['base_forward_kl'] for r in rows])),
        rows=rows)
    (ART/'gate_familiar_holdout.json').write_text(json.dumps(result,indent=2)+'\n')
    print('max family path MSE',result['max_family_path_mse'])
    print('mean base KL',result['mean_base_forward_kl'])
    print('families',family)


if __name__=='__main__':run()
