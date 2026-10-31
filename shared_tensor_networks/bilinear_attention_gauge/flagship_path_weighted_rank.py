"""Task-blind finite-path weighting for later-head reduction metrics.

Native train finite-path magnitudes select three edit families per head.
No answer tokens or task labels select the families. The downstream path
measure is used only to define the fit measure; the saved validation panel
tests immediate c_proj fidelity, not final-logit fidelity of a reduced cut.
"""

import json

import numpy as np
import torch

from flagship_all_head_rank_frontier import metrics,top_basis
from flagship_joint_edit_geometry import ART,LAYERS,HEADS
from flagship_matrix_free import SNAPSHOT


RANKS=(4,8,16,32,64,80,96)
FLOOR=.1


def run():
    data=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    path=json.loads((ART/'taskblind_head_path_scan.json').read_text())
    train_types=np.array([r['type'] for r in meta['panels']['train']])
    val_types=np.array([r['type'] for r in meta['panels']['validation']])
    names=sorted(set(train_types))
    assert names==sorted(set(val_types))
    summary={(r['layer'],r['head'],r['panel']):r for r in path['summary']}
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    rows=[]
    for layer in LAYERS:
        train=data[f'layer{layer}__train'].astype(np.float64)
        validation=data[f'layer{layer}__validation'].astype(np.float64)
        projection=state[f'transformer.h.{layer}.attn.c_proj.weight'].numpy()
        for head in range(HEADS):
            weight=projection[:,head*128:(head+1)*128].astype(np.float64)
            gram=weight.T@weight
            x=train[:,head,:];y=validation[:,head,:]
            train_cov={name:x[train_types==name].T@x[train_types==name]/
                       np.sum(train_types==name) for name in names}
            val_cov={name:y[val_types==name].T@y[val_types==name]/
                     np.sum(val_types==name) for name in names}
            original=y.T@y/len(y)
            path_train=summary[layer,head,'train']['mean_by_type']
            path_val=summary[layer,head,'validation']['mean_by_type']
            chosen=tuple(sorted(names,key=lambda name:
                -path_train[name]['path_l2'])[:3])
            val_top=tuple(sorted(names,key=lambda name:
                -path_val[name]['path_l2'])[:3])
            def normalized_output_cov(name):
                cov=train_cov[name]
                return cov/max(np.trace(cov@gram),1e-30)
            balanced=sum(normalized_output_cov(name) for name in names)/len(names)
            targeted=sum(normalized_output_cov(name) for name in chosen)/len(chosen)
            targeted=(1-FLOOR)*targeted+FLOOR*balanced
            frequency=x.T@x/len(x)
            measures={'pooled_edit':frequency,'balanced_all_types':balanced,
                      'top_three_path_with_floor':targeted}
            for method,moment in measures.items():
                basis=top_basis(moment/np.trace(moment))
                for rank in RANKS:
                    selected=basis[:,:rank]
                    by_type={name:metrics(selected,cov,gram)
                             for name,cov in val_cov.items()}
                    selected_errors=[by_type[name]['immediate_output_relative_mse']
                                     for name in chosen]
                    rows.append(dict(layer=layer,head=head,method=method,rank=rank,
                        train_top_three=chosen,validation_top_three=val_top,
                        selected_type_overlap=len(set(chosen)&set(val_top)),
                        aggregate_immediate_output_relative_mse=metrics(
                            selected,original,gram)['immediate_output_relative_mse'],
                        max_selected_type_relative_mse=float(max(selected_errors)),
                        mean_selected_type_relative_mse=float(np.mean(selected_errors)),
                        max_unselected_type_relative_mse=float(max(
                            by_type[name]['immediate_output_relative_mse']
                            for name in names if name not in chosen)),
                        per_type_relative_mse={name:by_type[name][
                            'immediate_output_relative_mse'] for name in names}))
            print('layer',layer,'head',head,'top',chosen,flush=True)
    result=dict(description='Native train-panel finite path magnitude chooses three edit families per head. A 10% equal-output-type floor is mixed into their normalized edit covariances. Rank-r spectral cuts evaluated on independent same-type validation native edit responses after c_proj; final logits of reduced cuts not evaluated.',
        ranks=RANKS,floor=FLOOR,n_train=800,n_validation=800,
        methods=['pooled_edit','balanced_all_types','top_three_path_with_floor'],
        rows=rows)
    (ART/'all_head_path_weighted_rank.json').write_text(
        json.dumps(result,separators=(',',':'))+'\n')
    for rank in RANKS:
        for method in ('pooled_edit','balanced_all_types','top_three_path_with_floor'):
            part=[r for r in rows if r['rank']==rank and r['method']==method]
            print(rank,method,'selected max',round(np.mean([
                r['max_selected_type_relative_mse'] for r in part]),4),
                'aggregate',round(np.mean([
                r['aggregate_immediate_output_relative_mse'] for r in part]),4),
                'unselected max',round(np.mean([
                r['max_unselected_type_relative_mse'] for r in part]),4))


if __name__=='__main__':run()
