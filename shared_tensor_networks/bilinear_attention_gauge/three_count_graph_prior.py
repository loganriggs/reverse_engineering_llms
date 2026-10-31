"""Label-free sparse-key plus quadratic-node gauge sweep on scratch toys."""

import json
import numpy as np

from dual_joint_function_canonical import HERE
from three_count_nonlinear_gauge import edit_panel, key_alignment, rotation


def terms(matrix, q, dictionary):
    c = (matrix @ dictionary.T).reshape(2, 3, 2)
    h = np.einsum('qic,ij,qjd->qcd', c, q, c)
    # Frobenius-normalized quadratic monomials, grouped over query types.
    return np.stack((h[:, 0, 0], np.sqrt(2)*h[:, 0, 1], h[:, 1, 1]), axis=1)


def one_node_deletion_error(matrix,q,dictionary,panel,deleted_node=None):
    z=panel['counts']@dictionary.T
    query=panel['query']
    coef=terms(matrix,q,dictionary)[query]
    features=np.stack((z[:,0]**2,np.sqrt(2)*z[:,0]*z[:,1],z[:,1]**2),axis=1)
    contributions=coef*features
    whole=contributions.sum(axis=1)
    # Best single node to delete, chosen using this panel's unlabeled function.
    # Center both signals so fixed offsets cannot distort the relative error.
    energy=np.mean((contributions-contributions.mean(axis=0))**2,axis=0)
    denominator=np.var(whole)
    node=int(np.argmin(energy)) if deleted_node is None else deleted_node
    return float(energy[node]/max(denominator,1e-15)),node


def pruning_scores(matrix,q,dictionary,fit,test,shifted):
    _,node=one_node_deletion_error(matrix,q,dictionary,fit)
    return dict(deleted_node=node,
        test_one_node_deletion_relative_mse=one_node_deletion_error(
            matrix,q,dictionary,test,node)[0],
        shifted_one_node_deletion_relative_mse=one_node_deletion_error(
            matrix,q,dictionary,shifted,node)[0])


def run(grid=1024):
    analysis=json.loads((HERE/'three_count_composed_scratch_analysis.json').read_text())
    gauges=json.loads((HERE/'three_count_composed_scratch_gauge.json').read_text())
    lookup={(r['seed'],r['reduction']):r for r in gauges['rows']}
    angles=np.arange(grid)*np.pi/(2*grid)
    rows=[]
    for row in analysis['rows']:
        q=np.asarray(row['symmetric_quadratic_effective'])
        seed=row['seed']
        fit=edit_panel(4096,91000+seed,True)
        test=edit_panel(4096,92000+seed,False)
        shifted=edit_panel(4096,93000+seed,False,rate=1.2)
        for reduction in ('uniform_rank2','empirical_rank2'):
            matrix=np.asarray(row['rank2_matrices'][reduction])
            _,s,v=np.linalg.svd(matrix,full_matrices=False)
            assert s[2]<1e-7
            base=v[:2]
            dictionaries=np.einsum('aij,jk->aik',np.array([rotation(a) for a in angles]),base)
            for d in (dictionaries[0],dictionaries[grid//3]):
                z=test['counts']@d.T
                query=test['query']
                monomial=terms(matrix,q,d)[query]
                reconstructed=(monomial[:,0]*z[:,0]**2+
                    np.sqrt(2)*monomial[:,1]*z[:,0]*z[:,1]+
                    monomial[:,2]*z[:,1]**2)
                logits=np.einsum('nk,nok->no',test['counts'],
                    matrix.reshape(2,3,8)[query])
                direct=np.einsum('ni,ij,nj->n',logits,q,logits)
                assert np.max(np.abs(reconstructed-direct))<1e-9
            key=np.abs(dictionaries).sum(axis=(1,2))
            monomials=np.array([terms(matrix,q,d) for d in dictionaries])
            norm=np.linalg.norm(monomials,axis=1)
            norm=norm/np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-15)
            # Saturating support surrogate: a node reaches unit cost once its
            # coefficient reaches tau times the total quadratic coefficient norm.
            candidates={}
            for tau in (.03,.1,.2):
                node_cost=np.minimum(1,norm/tau).sum(axis=1)
                for lam in (0,.1,.3,1,3,10):
                    objective=key/key.min()+lam*node_cost
                    idx=int(np.argmin(objective))
                    d=dictionaries[idx]
                    candidates[f'tau{tau}_lambda{lam}']=dict(angle=float(angles[idx]),
                        key_l1=float(key[idx]),node_cost=float(node_cost[idx]),
                        node_strengths=norm[idx].tolist(),
                        pruning=pruning_scores(matrix,q,d,fit,test,shifted),
                        alignment=key_alignment(d)['mean_distinct_family_cosine'])
            old=lookup[(row['seed'],reduction)]['gauges']
            # A sparsity count is only meaningful together with the approximation
            # error incurred by deleting graph nodes. Score on a fixed isotropic
            # coordinate distribution, independent of task labels.
            rows.append(dict(seed=row['seed'],reduction=reduction,candidates=candidates,
                baselines={k:dict(angle=v['angle'],alignment=v['alignment']['mean_distinct_family_cosine'],
                                  pruning=pruning_scores(matrix,q,rotation(v['angle'])@base,
                                                         fit,test,shifted))
                           for k,v in old.items()}))
        print(row['seed'],flush=True)
    result=dict(grid=grid,fit_labels_used=False,rows=rows)
    (HERE/'three_count_graph_prior.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    run()
