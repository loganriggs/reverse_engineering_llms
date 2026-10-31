"""Choose exact gauges by direct post-pruning function error, not node count."""

import json
import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_graph_prior import one_node_deletion_error,terms
from three_count_metric import pair_weights,weighted_fit
from three_count_nonlinear_gauge import edit_panel,key_alignment,rotation
from three_count_task_fit import count_features
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


def uniform_panel(key_length):
    keys=np.indices((8,)*key_length).reshape(key_length,-1).T
    counts=np.zeros((len(keys),8))
    for j in range(key_length):
        counts[np.arange(len(keys)),keys[:,j]]+=1
    return dict(counts=np.tile(counts,(2,1)),
                query=np.repeat(np.arange(2),len(keys)))


def deletion_curve(matrix,q,dictionaries,panel):
    errors=np.empty((len(dictionaries),3))
    counts=panel['counts'];query=panel['query']
    for start in range(0,len(dictionaries),64):
        d=dictionaries[start:start+64]
        z=np.einsum('nk,bck->bnc',counts,d)
        coefficients=np.array([terms(matrix,q,axis) for axis in d])[:,query,:]
        features=np.stack((z[:,:,0]**2,np.sqrt(2)*z[:,:,0]*z[:,:,1],
                           z[:,:,1]**2),axis=2)
        contributions=coefficients*features
        node_variance=np.var(contributions,axis=1)
        whole_variance=np.var(contributions.sum(axis=2),axis=1)
        errors[start:start+len(d)]=node_variance/np.maximum(whole_variance[:,None],1e-15)
    return errors


def evaluate(matrix,q,d,node,panel):
    return one_node_deletion_error(matrix,q,d,panel,node)[0]


def run(seeds=range(10,20),grid=1024,epsilons=(.01,.03,.1)):
    torch.set_num_threads(1)
    angles=np.arange(grid)*np.pi/(2*grid)
    rotations=np.array([rotation(a) for a in angles])
    uniform_fit=uniform_panel(3)
    uniform_test=uniform_panel(4)
    rows=[]
    for seed in seeds:
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        matrix=flatten(sum(head_tables(model.backbone)))
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        train,_,_=examples(20000,110000+seed,True)
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        reductions={'uniform_rank2':(u[:,:2]*s[:2])@v[:2]}
        reductions['empirical_rank2'],_=weighted_fit(matrix,pair_weights(train.numpy()),2)
        natural_fit=edit_panel(4096,91000+seed,True)
        natural_test=edit_panel(4096,92000+seed,False)
        shifted=edit_panel(4096,93000+seed,False,rate=1.2)
        for reduction,m in reductions.items():
            _,sv,base=np.linalg.svd(m,full_matrices=False)
            assert sv[2]<1e-7
            dictionaries=np.einsum('aij,jk->aik',rotations,base[:2])
            key=np.abs(dictionaries).sum(axis=(1,2))
            curves={'uniform':deletion_curve(m,q,dictionaries,uniform_fit),
                    'natural':deletion_curve(m,q,dictionaries,natural_fit)}
            selections={}
            for measure,curve in curves.items():
                for epsilon in epsilons:
                    feasible=np.where(curve.min(axis=1)<=epsilon)[0]
                    if len(feasible):
                        idx=int(feasible[np.argmin(key[feasible])])
                    else:
                        idx=int(np.argmin(curve.min(axis=1)))
                    node=int(np.argmin(curve[idx]))
                    d=dictionaries[idx]
                    selections[f'{measure}_eps{epsilon}']=dict(
                        angle=float(angles[idx]),deleted_node=node,
                        feasible=bool(len(feasible)),key_l1=float(key[idx]),
                        fit_relative_mse=float(curve[idx,node]),
                        alignment=key_alignment(d)['mean_distinct_family_cosine'],
                        uniform_length4_relative_mse=evaluate(m,q,d,node,uniform_test),
                        natural_test_relative_mse=evaluate(m,q,d,node,natural_test),
                        shifted_test_relative_mse=evaluate(m,q,d,node,shifted))
            key_index=int(np.argmin(key))
            for measure,curve in curves.items():
                node=int(np.argmin(curve[key_index]))
                d=dictionaries[key_index]
                selections[f'{measure}_key_l1']=dict(angle=float(angles[key_index]),
                    deleted_node=node,feasible=True,key_l1=float(key[key_index]),
                    fit_relative_mse=float(curve[key_index,node]),
                    alignment=key_alignment(d)['mean_distinct_family_cosine'],
                    uniform_length4_relative_mse=evaluate(m,q,d,node,uniform_test),
                    natural_test_relative_mse=evaluate(m,q,d,node,natural_test),
                    shifted_test_relative_mse=evaluate(m,q,d,node,shifted))
            rows.append(dict(seed=seed,reduction=reduction,selections=selections,
                best_fit_relative_mse={measure:float(curve.min())
                                       for measure,curve in curves.items()}))
        print(seed,flush=True)
    result=dict(seeds=list(seeds),grid=grid,epsilons=list(epsilons),
        measures={'uniform':'exhaustive 8^3 key triples x 2 queries',
                  'natural':'4096 unlabeled variable-count training-law contexts'},
        test={'uniform':'exhaustive 8^4 key quadruples x 2 queries',
              'natural':'4096 independent held-out contexts',
              'shift':'4096 independent contexts at higher key-count rate'},
        labels_used_in_gauge_fit=False,rows=rows)
    (HERE/'three_count_pruning_budget.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()
