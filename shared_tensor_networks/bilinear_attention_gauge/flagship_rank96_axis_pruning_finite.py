"""Finite-path consequences of top-k axes in exact rank-96 gauges."""

import json

import numpy as np
import torch

from flagship_all_head_rank_frontier import top_basis
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

SEED=20261105
K_VALUES=(8,16,32)
N_PER_FAMILIAR_TYPE=12
RANDOM_CONTROLS=4


def run():
    torch.set_num_threads(4)
    gate=json.loads((ART/'train_finite_fidelity_gate.json').read_text())
    assert (gate['layer'],gate['head'],gate['selected']['rank'],
            gate['selected']['method'])==(9,7,96,'pooled_edit')
    metadata=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    first=json.loads((ART/'new_family_path_transfer_rank96.json').read_text())
    second=json.loads((ART/'gate_second_family_stress.json').read_text())
    saved=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    train=saved['layer9__train'][:,7].astype(np.float64)
    val=saved['layer9__validation'][:,7]
    val_cases=metadata['panels']['validation']
    val_types=np.array([r['type'] for r in val_cases])
    indices=[int(i) for name in sorted(set(val_types))
             for i in np.flatnonzero(val_types==name)[:N_PER_FAMILIAR_TYPE]]
    cases=[dict(panel='familiar',source_index=i,**val_cases[i]) for i in indices]
    deltas=[val[i] for i in indices]
    new_cases=first['cases']+second['cases']
    novel=np.load(ART/'rank96_exact_gauge_novel_deltas.npz')['novel']
    assert novel.shape==(len(new_cases),128)
    for i,c in enumerate(new_cases):
        cases.append(dict(panel='unseen',source_index=i,**c))
        deltas.append(novel[i])
    deltas=np.stack(deltas).astype(np.float64)
    u=top_basis(train.T@train/len(train))[:,:96]
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[9].attn);probe.head=7
    w=model.transformer.h[9].attn.c_proj.weight.detach().numpy()[:,7*128:8*128].astype(np.float64)
    g=u.T@w.T@w@u
    eig,v=np.linalg.eigh((g+g.T)/2)
    assert eig.min()>1e-8
    root=(v*np.sqrt(eig))@v.T
    invroot=(v/np.sqrt(eig))@v.T
    projection_delta=deltas@u@u.T
    saved_rotations=np.load(ART/'rank96_exact_gauge_rotations.npz')
    names=(['output_weight_varimax','edit_activation_varimax',
            'edit_type_signature','pooled_edit_pca','identity']+
           [f'random_{i}' for i in range(RANDOM_CONTROLS)])
    gauges={name:saved_rotations[name] for name in names}
    group_deltas={};energy_retained={}
    for name,r in gauges.items():
        coords=deltas@u@root@r
        decoder=u@invroot@r
        assert np.linalg.norm(coords@decoder.T-projection_delta)/\
               np.linalg.norm(projection_delta)<1e-10
        order=np.argsort(-coords*coords,axis=1)
        for k in K_VALUES:
            keep=np.zeros_like(coords)
            np.put_along_axis(keep,order[:,:k],
                              np.take_along_axis(coords,order[:,:k],axis=1),axis=1)
            group_deltas[name,k]=(keep@decoder.T).astype(np.float32)
            energy_retained[name,k]=np.sum(keep*keep,axis=1)/np.sum(coords*coords,axis=1)
    grouped={}
    for i,c in enumerate(cases):
        grouped.setdefault((c['panel'],len(c['base_ids'])),[]).append(i)
    batches=[group[start:start+8] for group in grouped.values()
             for start in range(0,len(group),8)]
    rows=[]
    with torch.no_grad():
        for group in batches:
            before=torch.tensor([cases[i]['base_ids'] for i in group],dtype=torch.long)
            probe.basis=torch.eye(128);probe.injection=None
            base=final_logits(model,before).double()
            probe.injection=torch.tensor(projection_delta[group],dtype=torch.float32)
            full_projected=final_logits(model,before).double()-base
            probe.injection=torch.tensor(deltas[group],dtype=torch.float32)
            full_native=final_logits(model,before).double()-base
            probe.injection=None
            assert torch.all(full_projected.square().sum(-1)>1e-12)
            for name in names:
                for k in K_VALUES:
                    probe.injection=torch.tensor(group_deltas[name,k][group])
                    effect=final_logits(model,before).double()-base
                    probe.injection=None
                    relative=(effect-full_projected).square().sum(-1)/full_projected.square().sum(-1)
                    native_relative=(effect-full_native).square().sum(-1)/full_native.square().sum(-1)
                    for j,i in enumerate(group):
                        rows.append(dict(index=i,panel=cases[i]['panel'],type=cases[i]['type'],
                            method=name,k=k,immediate_output_energy_retained=float(
                                energy_retained[name,k][i]),
                            reduced_path_relative_mse=float(relative[j]),
                            native_path_relative_mse=float(native_relative[j]),
                            reduced_path_l2=float(full_projected[j].norm()),
                            native_path_l2=float(full_native[j].norm())))
            print('batch',len(group),'processed',len(set(r['index'] for r in rows)),
                  '/',len(cases),flush=True)
    result=dict(description='For frozen L9H7 rank96 pooled-edit cut, exact output-isometric gauges encode each native edit delta, retain its k largest absolute output-coordinate axes, reconstruct only those axes in head-state space, and patch them into the same native base model. Selectors use no family or answer labels. Compare final-logit path to full rank96 projected-delta patch and to full native-delta patch. Familiar panel has 12 independent cases per ten known types; unseen panel has 12 per ten whole-held-out designed types. All gauges share the same full rank96 reduced function.',
        layer=9,head=7,rank=96,seed=SEED,k_values=K_VALUES,
        methods=names,n_familiar=120,n_unseen=120,cases=cases,rows=rows)
    (ART/'rank96_axis_pruning_finite.json').write_text(json.dumps(result,
                                                    separators=(',',':'))+'\n')
    for panel in ('familiar','unseen'):
        for k in K_VALUES:
            for name in names[:5]:
                part=[r for r in rows if (r['panel'],r['k'],r['method'])==(panel,k,name)]
                print(panel,k,name,'energy',round(float(np.mean([
                    r['immediate_output_energy_retained'] for r in part])),3),
                    'path',round(float(np.mean([r['reduced_path_relative_mse']
                    for r in part])),3))


if __name__=='__main__':run()
