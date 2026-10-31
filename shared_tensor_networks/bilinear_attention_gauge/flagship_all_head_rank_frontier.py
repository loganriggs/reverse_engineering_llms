"""Weight-only versus edit-data rank frontier for 36 native attention heads.

The immediate-output metric applies each head's learned c_proj block to
saved held-out native edit responses. It is not a final-logit or KL test.
"""

import json

import numpy as np
import torch

from flagship_joint_edit_geometry import LAYERS, HEADS, ART
from flagship_matrix_free import SNAPSHOT


RANKS=(4,8,16,32,64,80,96,112,128)
MIXES=(0.,.1,.5,.9,1.)
EDIT_MEASURES=('frequency','equal_input_type','equal_output_type')


def top_basis(moment):
    eig,vec=np.linalg.eigh((moment+moment.T)/2)
    return vec[:,np.argsort(eig)[::-1]]


def metrics(basis,cov,gram):
    projector=basis@basis.T
    residual=np.eye(len(projector))-projector
    input_total=float(np.trace(cov))
    output_total=float(np.trace(cov@gram))
    input_error=float(np.trace(residual@cov)/input_total)
    output_error=float(np.trace(residual@cov@residual@gram)/output_total)
    assert input_error>-1e-8 and output_error>-1e-8
    return dict(input_relative_mse=max(0.,input_error),
                immediate_output_relative_mse=max(0.,output_error),
                input_energy=input_total,immediate_output_energy=output_total)


def run():
    saved=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    train_types=np.array([r['type'] for r in meta['panels']['train']])
    types=np.array([r['type'] for r in meta['panels']['validation']])
    names=sorted(set(types))
    assert set(names)==set(train_types)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    rows=[]
    for layer in LAYERS:
        train=saved[f'layer{layer}__train'].astype(np.float64)
        val=saved[f'layer{layer}__validation'].astype(np.float64)
        full=state[f'transformer.h.{layer}.attn.c_proj.weight'].numpy()
        for head in range(HEADS):
            x=train[:,head,:]
            y=val[:,head,:]
            weight=full[:,head*128:(head+1)*128].astype(np.float64)
            gram=weight.T@weight
            edit_moment=x.T@x/len(x)
            val_cov=y.T@y/len(y)
            train_by_type={name:(x[train_types==name].T@x[train_types==name])/
                           np.sum(train_types==name) for name in names}
            by_type={name:(y[types==name].T@y[types==name])/
                     np.sum(types==name) for name in names}
            normalized_gram=gram/np.trace(gram)
            edit_moments={
                'frequency':edit_moment,
                'equal_input_type':sum(cov/np.trace(cov)
                    for cov in train_by_type.values())/len(names),
                'equal_output_type':sum(cov/np.trace(cov@gram)
                    for cov in train_by_type.values())/len(names)}
            for edit_measure,target in edit_moments.items():
                normalized_edit=target/np.trace(target)
                for mix in MIXES:
                    basis=top_basis((1-mix)*normalized_gram+mix*normalized_edit)
                    for rank in RANKS:
                        selected=basis[:,:rank]
                        aggregate=metrics(selected,val_cov,gram)
                        per_type={name:metrics(selected,cov,gram)
                                  for name,cov in by_type.items()}
                        rows.append(dict(layer=layer,head=head,
                            edit_measure=edit_measure,mix=mix,rank=rank,
                            aggregate=aggregate,per_type=per_type,
                            max_type_input_relative_mse=float(max(
                                v['input_relative_mse'] for v in per_type.values())),
                            max_type_immediate_output_relative_mse=float(max(
                                v['immediate_output_relative_mse']
                                for v in per_type.values())),
                            weight_frobenius_fraction=float(np.trace(
                                selected.T@gram@selected)/np.trace(gram))))
            print('layer',layer,'head',head,'complete',flush=True)
    result=dict(description='All 36 native heads, train edit-moment vs weight-Gram spectral mixture at nine ranks, with frequency, equal-input-type, and equal-output-type edit measures. Evaluated on independent 800-edit validation panel. Immediate-output metric uses head c_proj block, not full-model logits.',
        layers=LAYERS,heads=HEADS,ranks=RANKS,mixes=MIXES,
        edit_measures=EDIT_MEASURES,
        val_edit_types=names,n_train=800,n_validation=800,rows=rows)
    (ART/'all_head_rank_frontier.json').write_text(
        json.dumps(result,separators=(',',':'))+'\n')
    for rank in (4,16,64,96):
        for measure,mix in (('frequency',1.),('equal_input_type',1.),
                            ('equal_output_type',1.),('frequency',0.)):
            subset=[r for r in rows if r['rank']==rank and r['mix']==mix and
                    r['edit_measure']==measure]
            print(rank,measure,mix,'head MSE',round(np.mean([
                r['aggregate']['input_relative_mse'] for r in subset]),4),
                'c_proj MSE',round(np.mean([
                r['aggregate']['immediate_output_relative_mse']
                for r in subset]),4),'max-type c_proj',round(np.mean([
                r['max_type_immediate_output_relative_mse']
                for r in subset]),4))


if __name__=='__main__':run()
