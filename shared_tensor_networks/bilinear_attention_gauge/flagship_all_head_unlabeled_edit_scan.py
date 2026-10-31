"""Answer-free edit-response scan over all heads in four later layers (CPU)."""

import json

import numpy as np
import torch

from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

LAYERS=(9, 11, 13, 15)
SEED=20261031


def capture(model, probes, cases):
    nhead=next(iter(probes.values())).n_head
    width=next(iter(probes.values())).head_dim
    out={layer:np.empty((len(cases),nhead,width),dtype=np.float32) for layer in LAYERS}
    by_length={}
    for i,case in enumerate(cases):
        by_length.setdefault(len(case['base_ids']),[]).append(i)
    with torch.no_grad():
        for length,indices in sorted(by_length.items()):
            for start in range(0,len(indices),16):
                ix=indices[start:start+16]
                before=torch.tensor([cases[i]['base_ids'] for i in ix],dtype=torch.long)
                after=torch.tensor([cases[i]['edited_ids'] for i in ix],dtype=torch.long)
                final_logits(model,before,stop_at=max(LAYERS))
                base={layer:probes[layer].last_input[:,-1].reshape(len(ix),nhead,width)
                      for layer in LAYERS}
                final_logits(model,after,stop_at=max(LAYERS))
                for layer in LAYERS:
                    edited=probes[layer].last_input[:,-1].reshape(len(ix),nhead,width)
                    out[layer][ix]=(edited-base[layer]).numpy()
    return out


def evaluate(arrays,meta):
    records=[]
    types=np.array([r['type'] for r in meta['panels']['validation']])
    for layer in LAYERS:
        train=arrays[f'layer{layer}__train'].astype(np.float64)
        val=arrays[f'layer{layer}__validation'].astype(np.float64)
        for head in range(train.shape[1]):
            x=train[:,head,:];y=val[:,head,:]
            moment=x.T@x/len(x)
            eig,vec=np.linalg.eigh(moment)
            axis=vec[:,-1]
            train_energy=float(eig[-1])
            val_energy=float(np.mean((y@axis)**2))
            # The axis and head are ranked without edit-type labels.
            score=np.abs(y@axis)
            norm=np.linalg.norm(y,axis=-1)
            top_axis=score>=np.quantile(score,.9)
            top_norm=norm>=np.quantile(norm,.9)
            counts_axis={kind:int(np.sum(top_axis & (types==kind))) for kind in sorted(set(types))}
            counts_norm={kind:int(np.sum(top_norm & (types==kind))) for kind in sorted(set(types))}
            records.append(dict(layer=layer,head=head,train_top_axis_energy=train_energy,
                validation_top_axis_energy=val_energy,
                train_total_energy=float(np.mean(np.sum(x*x,axis=-1))),
                validation_total_energy=float(np.mean(np.sum(y*y,axis=-1))),
                train_validation_axis_energy_ratio=val_energy/max(train_energy,1e-30),
                top_decile_axis_type_counts=counts_axis,
                top_decile_head_norm_type_counts=counts_norm))
    return sorted(records,key=lambda r:r['train_top_axis_energy'],reverse=True)


def run():
    torch.set_num_threads(4)
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probes={layer:HeadCut(model.transformer.h[layer].attn) for layer in LAYERS}
    arrays={}
    for panel,cases in meta['panels'].items():
        captured=capture(model,probes,cases)
        arrays.update({f'layer{layer}__{panel}':captured[layer] for layer in LAYERS})
        print('captured',panel,len(cases),flush=True)
    records=evaluate(arrays,meta)
    np.savez_compressed(ART/'all_head_unlabeled_edit_scan.npz',**arrays)
    (ART/'all_head_unlabeled_edit_scan.json').write_text(json.dumps(dict(
        description='Train-axis unsupervised all-head scan; edit types used only to evaluate top decile',
        layers=LAYERS,seed=SEED,rank_rule='largest uncentered edit-response PCA eigenvalue',
        records=records),indent=2)+'\n')
    print('top heads',[(r['layer'],r['head'],round(r['train_top_axis_energy'],2),
                        r['top_decile_axis_type_counts']) for r in records[:8]],flush=True)


if __name__=='__main__':run()
