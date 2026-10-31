"""Answer-free edit-fidelity rank selection on ten trained toy-language models."""

import json
from itertools import combinations

import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_metric import pair_weights,weighted_fit
from three_count_task_fit import count_features
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


WITHIN=((0,1),(2,3),(4,5),(6,7))
CROSS=tuple(e for e in combinations(range(8),2) if e not in WITHIN)
RANKS=(1,2,3,4,5,6)
MEASURES=('uniform','empirical','half_uniform_empirical')
THRESHOLDS=(.001,.01,.05)


def response_error(reference,cut,pairs,query):
    r=reference.reshape(2,3,8)[query]
    c=cut.reshape(2,3,8)[query]
    target=np.stack([r[:,j]-r[:,i] for i,j in pairs])
    predicted=np.stack([c[:,j]-c[:,i] for i,j in pairs])
    return float(np.sum((predicted-target)**2)/max(np.sum(target**2),1e-30))


def task_scores(cut,q,bias,pb,tokens,target,attrs):
    counts=count_features(tokens).astype(np.float64)
    query=(tokens[:,-1]-8).astype(int)
    bits=bias+np.einsum('nk,nok->no',counts,cut.reshape(2,3,8)[query])
    parity=np.einsum('ni,ij,nj->n',bits,q,bits)+pb
    pred=np.c_[bits,parity]>0
    truth=np.c_[target.astype(bool),np.logical_xor(target[:,0],target[:,1])]
    out={}
    for name,mask in [('all',np.ones(len(tokens),dtype=bool)),
                      ('rare_seen',attrs['rare']&~attrs['heldout']),
                      ('rare_heldout',attrs['heldout'])]:
        out[name]=dict(n=int(mask.sum()),all_four_accuracy=float(np.mean(
            np.all(pred[mask]==truth[mask],axis=1))),
            parity_accuracy=float(np.mean(pred[mask,3]==truth[mask,3])))
    return out


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        matrix=flatten(sum(head_tables(model.backbone)))
        train,_,_=examples(20000,110000+seed,True)
        frequency=pair_weights(train.numpy())
        uniform=np.full_like(frequency,1/frequency.size)
        weights={'uniform':uniform,'empirical':frequency,
                 'half_uniform_empirical':.5*(uniform+frequency)}
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        bias=model.backbone.read.bias.detach().numpy()
        pb=float(model.parity_bias.detach())
        test,bits,attrs=examples(30000,130000+seed,False)
        test=test.numpy();bits=bits.numpy()
        for measure,weight in weights.items():
            for rank in RANKS:
                if measure=='uniform':
                    u,s,v=np.linalg.svd(matrix,full_matrices=False)
                    cut=(u[:,:rank]*s[:rank])@v[:rank]
                    info={'method':'SVD'}
                else:
                    cut,info=weighted_fit(matrix,weight,rank)
                edit={f'{panel}_{query}':response_error(matrix,cut,pairs,query)
                      for panel,pairs in [('within',WITHIN),('cross',CROSS)]
                      for query in [0,1]}
                rows.append(dict(seed=seed,measure=measure,rank=rank,
                    uniform_relative_table_mse=float(np.sum((cut-matrix)**2)/
                                                      np.sum(matrix**2)),
                    empirical_relative_table_mse=float(np.sum((cut-matrix)**2*frequency)/
                                                        np.sum(matrix**2*frequency)),
                    edit_relative_mse=edit,
                    task=task_scores(cut,q,bias,pb,test,bits,attrs),fit=info))
        print('seed',seed,flush=True)
    choices=[]
    for seed in range(10):
        for measure in MEASURES:
            pool=[r for r in rows if r['seed']==seed and r['measure']==measure]
            for threshold in THRESHOLDS:
                selected=next((r for r in pool if max(r['edit_relative_mse'][
                    f'within_{q}'] for q in [0,1])<=threshold),pool[-1])
                choices.append(dict(seed=seed,measure=measure,threshold=threshold,
                    selected_rank=selected['rank'],
                    rare_heldout_accuracy=selected['task']['rare_heldout']['all_four_accuracy'],
                    all_accuracy=selected['task']['all']['all_four_accuracy']))
    result=dict(description='Rank selection from answer-free exact token-edit response fidelity on ten trained three-variable quadratic-readout bilinear attention models',
        ranks=RANKS,measures=MEASURES,thresholds=THRESHOLDS,
        within_pair_edits=WITHIN,cross_pair_edits=CROSS,
        reduction_uses_task_labels=False,rank_selection_uses_task_labels=False,
        task_labels_used_only_for_evaluation=True,
        caveat='Weight-only and weighted token-table cuts both see all eight key-token cells; the edit panel is a separate selection metric but not an unseen-token test.',
        rows=rows,choices=choices)
    (HERE/'three_count_rank_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    for measure in MEASURES:
        for threshold in THRESHOLDS:
            subset=[x for x in choices if x['measure']==measure and
                    x['threshold']==threshold]
            print(measure,threshold,'ranks',[x['selected_rank'] for x in subset],
                  'rare',round(float(np.mean([x['rare_heldout_accuracy'] for x in subset])),4),
                  flush=True)


if __name__=='__main__':run()
