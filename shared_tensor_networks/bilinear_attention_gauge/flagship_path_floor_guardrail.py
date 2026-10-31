"""Training-only coverage guardrail for path-selected head subspaces."""

import json

import numpy as np
import torch

from flagship_all_head_rank_frontier import metrics, top_basis
from flagship_joint_edit_geometry import ART, LAYERS, HEADS
from flagship_matrix_free import SNAPSHOT

RANKS=(16,32,64,96)
FLOORS=(0,.05,.1,.2,.35,.5,.75,1.)
TOLERANCE=.02


def run():
    saved=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    path=json.loads((ART/'taskblind_head_path_scan.json').read_text())
    types={panel:np.array([r['type'] for r in cases])
           for panel,cases in meta['panels'].items()}
    names=sorted(set(types['train']))
    summary={(r['layer'],r['head'],r['panel']):r for r in path['summary']}
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    rows=[]
    for layer in LAYERS:
        train=saved[f'layer{layer}__train'].astype(np.float64)
        validation=saved[f'layer{layer}__validation'].astype(np.float64)
        projection=state[f'transformer.h.{layer}.attn.c_proj.weight'].numpy()
        for head in range(HEADS):
            weight=projection[:,head*128:(head+1)*128].astype(np.float64)
            gram=weight.T@weight
            x=train[:,head,:];y=validation[:,head,:]
            cov={panel:{name:z[types[panel]==name].T@z[types[panel]==name]
                        /int(np.sum(types[panel]==name)) for name in names}
                 for panel,z in [('train',x),('validation',y)]}
            path_train=summary[layer,head,'train']['mean_by_type']
            chosen=tuple(sorted(names,key=lambda name:
                         -path_train[name]['path_l2'])[:3])
            unselected=[name for name in names if name not in chosen]
            normalized={name:cov['train'][name]/max(
                np.trace(cov['train'][name]@gram),1e-30) for name in names}
            balanced=sum(normalized.values())/len(names)
            targeted=sum(normalized[name] for name in chosen)/len(chosen)
            pooled=x.T@x/len(x)
            moments={'pooled':pooled}
            moments.update({f'floor_{floor:g}':(1-floor)*targeted+floor*balanced
                            for floor in FLOORS})
            bases={key:top_basis(moment/np.trace(moment))
                   for key,moment in moments.items()}
            for rank in RANKS:
                candidate=[]
                for key,basis in bases.items():
                    p=basis[:,:rank]
                    error={panel:{name:metrics(p,cov[panel][name],gram)[
                        'immediate_output_relative_mse'] for name in names}
                        for panel in ('train','validation')}
                    candidate.append(dict(method=key,rank=rank,
                        train_selected_mean=float(np.mean([error['train'][name]
                            for name in chosen])),
                        train_unselected_max=float(max(error['train'][name]
                            for name in unselected)),
                        validation_selected_mean=float(np.mean([
                            error['validation'][name] for name in chosen])),
                        validation_unselected_max=float(max([
                            error['validation'][name] for name in unselected])),
                        validation_aggregate=metrics(p,y.T@y/len(y),gram)[
                            'immediate_output_relative_mse']))
                pooled_row=candidate[0]
                feasible=[r for r in candidate if r['train_unselected_max']<=
                          pooled_row['train_unselected_max']+TOLERANCE+1e-12]
                assert feasible
                policy=min(feasible,key=lambda r:r['train_selected_mean'])
                rows.append(dict(layer=layer,head=head,rank=rank,
                    train_top_three=chosen,policy=policy,pooled=pooled_row,
                    candidates=candidate))
            print('layer',layer,'head',head,flush=True)
    result=dict(description='For each head and rank, select path-targeted floor or pooled baseline solely using training edit responses. Minimize mean selected-type immediate-output relative MSE subject to worst unselected-type training error <= pooled baseline + absolute .02. Evaluate same-type independent validation edits. Native train full-model path magnitude selects three types; no answer labels used. This is a local c_proj proxy, not a finite full-model or new-type test.',
        ranks=RANKS,floors=FLOORS,tolerance=TOLERANCE,rows=rows)
    (ART/'path_floor_guardrail.json').write_text(json.dumps(result,
                                                    separators=(',',':'))+'\n')
    for rank in RANKS:
        part=[r for r in rows if r['rank']==rank]
        print('rank',rank,'selected',np.mean([r['policy'][
            'validation_selected_mean'] for r in part]),'pooled',np.mean([
            r['pooled']['validation_selected_mean'] for r in part]),
            'unselected',np.mean([r['policy']['validation_unselected_max']
            for r in part]),'pooled',np.mean([r['pooled'][
            'validation_unselected_max'] for r in part]),'nonpooled',sum(
            r['policy']['method']!='pooled' for r in part))


if __name__=='__main__':run()
