"""Out-of-sample seed check for the fixed quadratic graph prior."""

import json
import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_composed_analysis import score
from three_count_graph_prior import one_node_deletion_error,pruning_scores,terms
from three_count_metric import pair_weights,weighted_fit
from three_count_nonlinear_gauge import edit_panel,key_alignment,rotation
from three_count_task_fit import count_features
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


def run(seeds=range(10,20),grid=1024,tau=.1,lam=.3):
    torch.set_num_threads(1)
    angles=np.arange(grid)*np.pi/(2*grid)
    rotations=np.array([rotation(a) for a in angles])
    rows=[]
    for seed in seeds:
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        matrix=flatten(sum(head_tables(model.backbone)))
        bias=model.backbone.read.bias.detach().numpy()
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        pb=float(model.parity_bias.detach())
        train,_,_=examples(20000,110000+seed,True)
        test,bits,attrs=examples(30000,130000+seed,False)
        test_np=test.numpy();bits_np=bits.numpy().astype(bool)
        with torch.no_grad():direct=model(test[:256]).numpy()
        _,proxy=score(matrix,bias,q,pb,test_np[:256],bits_np[:256],
            {k:v[:256] for k,v in attrs.items()},direct)
        assert np.max(np.abs(proxy-direct))<1e-9
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        reductions={'uniform_rank2':(u[:,:2]*s[:2])@v[:2]}
        reductions['empirical_rank2'],_=weighted_fit(matrix,pair_weights(train.numpy()),2)
        fit=edit_panel(4096,91000+seed,True)
        edit_test=edit_panel(4096,92000+seed,False)
        shifted=edit_panel(4096,93000+seed,False,rate=1.2)
        counts=count_features(test_np).astype(np.float64)
        query=(test_np[:,-1]-8).astype(int)
        truth=np.c_[bits_np,np.logical_xor(bits_np[:,0],bits_np[:,1])]
        for reduction,m in reductions.items():
            _,sv,base=np.linalg.svd(m,full_matrices=False)
            assert sv[2]<1e-7
            d_all=np.einsum('aij,jk->aik',rotations,base[:2])
            key=np.abs(d_all).sum(axis=(1,2))
            monomials=np.array([terms(m,q,d) for d in d_all])
            strength=np.linalg.norm(monomials,axis=1)
            strength/=np.maximum(np.linalg.norm(strength,axis=1,keepdims=True),1e-15)
            graph_cost=np.minimum(1,strength/tau).sum(axis=1)
            indexes={'key_l1':int(np.argmin(key)),
                     'graph_prior':int(np.argmin(key/key.min()+lam*graph_cost))}
            logits=bias+np.einsum('nk,nok->no',counts,m.reshape(2,3,8)[query])
            parity=np.einsum('ni,ij,nj->n',logits,q,logits)+pb
            original=np.c_[logits,parity]>0
            masks={'all':np.ones(len(test_np),dtype=bool),'rare_heldout':attrs['heldout']}
            methods={}
            for name,index in indexes.items():
                d=d_all[index]
                pruning=pruning_scores(m,q,d,fit,edit_test,shifted)
                node=pruning['deleted_node']
                z=counts@d.T
                features=np.stack((z[:,0]**2,np.sqrt(2)*z[:,0]*z[:,1],z[:,1]**2),axis=1)
                removed=terms(m,q,d)[query][:,node]*features[:,node]
                pred=np.c_[logits,parity-removed]>0
                methods[name]=dict(angle=float(angles[index]),
                    key_l1=float(key[index]),node_cost=float(graph_cost[index]),
                    alignment=key_alignment(d)['mean_distinct_family_cosine'],
                    pruning=pruning,
                    task={group:dict(parity_accuracy=float(np.mean(pred[mask,3]==truth[mask,3])),
                        all_four_accuracy=float(np.mean(np.all(pred[mask]==truth[mask],axis=1))))
                        for group,mask in masks.items()})
            rows.append(dict(seed=seed,reduction=reduction,
                native_rare_accuracy=float(json.loads((HERE/f'three_count_composed_scratch_seed{seed}.json').read_text())['test']['rare_heldout']['all_four_accuracy']),
                unpruned={group:dict(parity_accuracy=float(np.mean(original[mask,3]==truth[mask,3])),
                    all_four_accuracy=float(np.mean(np.all(original[mask]==truth[mask],axis=1))))
                    for group,mask in masks.items()},methods=methods))
        print(seed,flush=True)
    result=dict(seeds=list(seeds),tau=tau,lam=lam,grid=grid,fit_labels_used=False,
                selection='tau and lambda fixed from earlier seeds 0-9; fresh models 10-19',rows=rows)
    (HERE/'three_count_graph_prior_fresh.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()
