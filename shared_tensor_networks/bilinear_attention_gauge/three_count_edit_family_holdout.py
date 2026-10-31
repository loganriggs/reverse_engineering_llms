"""Hold out within-variable edit edges when fitting rank-three key gauges."""

import json
from itertools import combinations

import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from shared_pair_dense_edit_gauge import fit_edit
from three_count_metric import pair_weights,weighted_fit
from three_count_scratch_edit_gauge import alignment,task_score
from train_three_count_composed import ComposedCountModel
from train_three_count_language import examples


HELDOUT=((0,1),(2,3),(4,5),(6,7))
FIT=tuple(pair for pair in combinations(range(8),2) if pair not in HELDOUT)
COVERAGE=(7,10,14,18,24)


def connected_orders():
    orders=[]
    for split in range(5):
        rng=np.random.default_rng(20261113+split)
        shuffled=[FIT[i] for i in rng.permutation(len(FIT))]
        parent=list(range(8))
        def root(i):
            while parent[i]!=i:i=parent[i]
            return i
        tree=[];extra=[]
        for i,j in shuffled:
            ri,rj=root(i),root(j)
            if ri!=rj:
                parent[ri]=rj;tree.append((i,j))
            else:extra.append((i,j))
        assert len(tree)==7 and len(extra)==17
        orders.append(tuple(tree+extra))
    return orders


def edit_l1(dictionary,pairs):
    return float(np.mean([np.abs(dictionary[:,j]-dictionary[:,i]).sum()
                          for i,j in pairs]))


def run():
    torch.set_num_threads(1)
    saved=json.loads((HERE/'three_count_scratch_edit_gauge.json').read_text())
    rows=[];coverage_rows=[]
    orders=connected_orders()
    for seed in range(10):
        model=ComposedCountModel().double()
        model.load_state_dict(torch.load(HERE/f'three_count_composed_scratch_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        matrix=flatten(sum(head_tables(model.backbone)))
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        uniform=(u[:,:3]*s[:3])@v[:3]
        train,_,_=examples(20000,110000+seed,True)
        empirical,_=weighted_fit(matrix,pair_weights(train.numpy()),3)
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.T)/scale[:,None]/scale[None,:]
        bias=model.backbone.read.bias.detach().numpy()
        pb=float(model.parity_bias.detach())
        tokens,bits,attrs=examples(30000,130000+seed,False)
        tokens=tokens.numpy();target=bits.numpy().astype(bool)
        for reduction,cut in (('uniform_rank3',uniform),
                              ('empirical_rank3',empirical)):
            _,cs,cv=np.linalg.svd(cut,full_matrices=False)
            assert cs[3]<1e-8
            base=cv[:3]
            heldout_dictionary,fit_value=fit_edit(base,FIT,20261112+100*seed+
                                                    (0 if reduction=='uniform_rank3' else 1))
            original=next(r for r in saved['rows'] if r['seed']==seed and
                          r['reduction']==reduction)['gauges']
            choices={'svd':np.asarray(original['svd']['dictionary']),
                     'key_l1':np.asarray(original['key_l1']['dictionary']),
                     'all_pairs':np.asarray(original['all_pairs']['dictionary']),
                     'fit_without_bit_flips':heldout_dictionary}
            outcomes={}
            for name,d in choices.items():
                assert np.linalg.norm(cut@d.T@d-cut)/np.linalg.norm(cut)<1e-8
                task=task_score(cut,q,bias,pb,d,tokens,target,attrs)
                outcomes[name]=dict(dictionary=d.tolist(),
                    matched_variable_cosine=alignment(d),
                    fit_edit_l1=edit_l1(d,FIT),
                    heldout_bit_flip_l1=edit_l1(d,HELDOUT),
                    task_after_two_node_prune=task)
            rows.append(dict(seed=seed,reduction=reduction,
                             fit_objective=fit_value,gauges=outcomes))
            for split,order in enumerate(orders):
                for edge_count in COVERAGE:
                    selected=order[:edge_count]
                    d,objective=fit_edit(base,selected,
                        20261114+10000*seed+1000*(reduction=='empirical_rank3')+
                        100*split+edge_count)
                    assert np.linalg.norm(cut@d.T@d-cut)/np.linalg.norm(cut)<1e-8
                    task=task_score(cut,q,bias,pb,d,tokens,target,attrs)
                    coverage_rows.append(dict(seed=seed,reduction=reduction,
                        split=split,edge_count=edge_count,fit_edges=selected,
                        fit_objective=objective,
                        matched_variable_cosine=alignment(d),
                        heldout_bit_flip_l1=edit_l1(d,HELDOUT),
                        full_two_node_accuracy=task['scores']['all']['all_four_accuracy'],
                        rare_two_node_accuracy=task['scores']['rare_heldout']['all_four_accuracy'],
                        top_two_monomial_energy_fraction=
                            task['top_two_monomial_energy_fraction']))
        print('seed',seed,flush=True)
    result=dict(description='Ten trained three-variable quadratic-readout attention models: fit sparse edits on 24 cross-pair token swaps; hold out the four within-pair bit flips including filler; evaluate known-variable alignment and two-node task pruning after fit',
        fit_edges=FIT,heldout_edges=HELDOUT,
        coverage_edge_counts=COVERAGE,coverage_orders=orders,
        model_training_uses_task_labels=True,
        gauge_fit_uses_task_labels=False,
        edge_partition_uses_known_input_token_pairing=True,
        rows=rows,coverage_rows=coverage_rows)
    (HERE/'three_count_edit_family_holdout.json').write_text(json.dumps(result,indent=2)+'\n')
    for reduction in ('uniform_rank3','empirical_rank3'):
        for name in ('svd','key_l1','all_pairs','fit_without_bit_flips'):
            subset=[r['gauges'][name] for r in rows if r['reduction']==reduction]
            print(reduction,name,'alignment',round(np.mean([x['matched_variable_cosine'] for x in subset]),4),
                  'fit',round(np.mean([x['fit_edit_l1'] for x in subset]),4),
                  'heldout',round(np.mean([x['heldout_bit_flip_l1'] for x in subset]),4),
                  'full',round(np.mean([x['task_after_two_node_prune']['scores']['all']['all_four_accuracy'] for x in subset]),4),
                  'rare',round(np.mean([x['task_after_two_node_prune']['scores']['rare_heldout']['all_four_accuracy'] for x in subset]),4),flush=True)
        for edge_count in COVERAGE:
            subset=[r for r in coverage_rows if r['reduction']==reduction and
                    r['edge_count']==edge_count]
            print(reduction,'coverage',edge_count,
                'alignment',round(np.mean([r['matched_variable_cosine'] for r in subset]),4),
                'heldout',round(np.mean([r['heldout_bit_flip_l1'] for r in subset]),4),
                'full',round(np.mean([r['full_two_node_accuracy'] for r in subset]),4),
                'rare',round(np.mean([r['rare_two_node_accuracy'] for r in subset]),4),flush=True)


if __name__=='__main__':run()
