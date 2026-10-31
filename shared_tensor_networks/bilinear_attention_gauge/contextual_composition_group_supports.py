"""Learn overlapping support groups from unlabeled sparse codes.

The 12-atom shared dictionary is a flat top-four autoencoder fitted on
primitive+AND head states. Binary overlapping groups are learned from its
top-four activation supports. Inference chooses one or two groups by
reconstruction error and counts every shared atom once. XOR is excluded
from dictionary fitting, group fitting, and model selection.
"""

from itertools import combinations
import json

import numpy as np
import torch
from scipy.cluster.vq import kmeans2

from contextual_composition_reduction import FIT_QUERIES, exact_head, head_table
from contextual_composition_shared_groups import FlatSparse
from dual_joint_function_canonical import HERE
from train_contextual_composition_attention import (
    ContextualCompositionAttention, FAMILIES, examples)


def flat_factors(model):
    with torch.no_grad():
        return (model.encoder.numpy().T,
                torch.nn.functional.normalize(model.decoder,dim=1).numpy())


def top_support(a, k=4, allowed=None):
    score=np.abs(a) if allowed is None else np.where(allowed[None],np.abs(a),-np.inf)
    indices=np.argpartition(score,-k,axis=1)[:,-k:]
    mask=np.zeros_like(a,dtype=np.float32)
    np.put_along_axis(mask,indices,1.,axis=1)
    if allowed is not None:mask*=allowed[None]
    return mask


def fit_groups(a, seed, size, groups=8):
    supports=top_support(a)
    # kmeans2's sampling is seeded locally and never reads held-out XOR.
    centers,labels=kmeans2(supports,groups,iter=60,minit='++',seed=seed)
    membership=np.zeros((groups,a.shape[1]),dtype=bool)
    for group,center in enumerate(centers):
        membership[group,np.argsort(center)[-size:]]=True
    assert np.all(membership.sum(axis=1)==size)
    return membership,dict(cluster_sizes=np.bincount(labels,minlength=groups).tolist(),
                           distinct_groups=int(len(np.unique(membership,axis=0))),
                           link_count=int(membership.sum()),
                           atoms_used_by_multiple_groups=int((membership.sum(0)>1).sum()))


def infer(a,h,decoder,membership,n_groups):
    choices=list(combinations(range(len(membership)),n_groups))
    union=np.stack([np.any(membership[list(combo)],axis=0) for combo in choices])
    best_error=np.full(len(a),np.inf)
    best_h=np.zeros_like(h)
    best_choice=np.zeros(len(a),dtype=np.int64)
    best_active=np.zeros(len(a),dtype=np.float32)
    for index,allowed in enumerate(union):
        mask=top_support(a,allowed=allowed)
        reconstructed=(a*mask)@decoder
        error=np.sum((reconstructed-h)**2,axis=1)
        better=error<best_error
        best_error[better]=error[better]
        best_h[better]=reconstructed[better]
        best_choice[better]=index
        best_active[better]=mask.sum(axis=1)[better]
    return best_h,best_choice,best_active,len(choices)


def check():
    assert not set(FIT_QUERIES).intersection(FAMILIES['xor'])
    a=np.array([[1.,2.,3.,4.]],dtype=np.float32)
    membership=np.ones((2,4),dtype=bool)
    decoded,_,active,choices=infer(a,a,np.eye(4,dtype=np.float32),
                                   membership,2)
    # Both active groups contain all four atoms; the union decodes once.
    np.testing.assert_array_equal(decoded,a)
    np.testing.assert_array_equal(active,[4])
    assert choices==1


def predictions(native,table,encoder,decoder,membership,x):
    h=exact_head(table,x)
    a=h@encoder
    flat_h=(a*top_support(a))@decoder
    variants={'flat_top4':(flat_h,np.full(len(h),4.),1),
              'physical_top3':(h*top_support(h,3),np.full(len(h),3.),1),
              'physical_top4':(h*top_support(h,4),np.full(len(h),4.),1)}
    for n_groups in (1,2):
        estimated,choice,active,candidates=infer(a,h,decoder,membership,n_groups)
        variants[f'group_top4_from_{n_groups}']=(estimated,active,candidates)
    q=native.embed(x[:,-1])
    with torch.no_grad():
        h_t=torch.from_numpy(h).float()
        reference=native.read(torch.cat((h_t,q),dim=1)).squeeze(-1)
        output_variants={}
        for name,(pred,active,candidates) in variants.items():
            pred_t=torch.from_numpy(pred).float()
            output=native.read(torch.cat((pred_t,q),dim=1)).squeeze(-1)
            output_variants[name]=(pred_t,output,active,candidates)
    return h_t,reference,output_variants


def evaluate(native,table,encoder,decoder,membership,x,y,query):
    h_t,reference,variants=predictions(native,table,encoder,decoder,
                                       membership,x)
    with torch.no_grad():
        results={}
        for name,(pred_t,output,active,candidates) in variants.items():
            results[name]={}
            for family,ids in FAMILIES.items():
                mask=np.isin(query,ids)
                m=torch.from_numpy(mask)
                results[name][family]=dict(
                    n=int(mask.sum()),
                    accuracy=float(((output[m]>0)==y[m].bool()).float().mean()),
                    head_relative_mse=float((pred_t[m]-h_t[m]).square().sum()/
                                            h_t[m].square().sum()),
                    relative_logit_mse=float((output[m]-reference[m]).square().sum()/
                                             reference[m].square().sum()),
                    mean_active_atoms=float(np.mean(active[mask])),
                    candidate_group_sets=candidates)
    return results


def evaluate_edits(native,table,encoder,decoder,membership,x,query,seed):
    rng=np.random.default_rng(seed)
    edited=x.clone()
    key_position=rng.integers(0,7,size=len(x))
    bit=rng.integers(0,4,size=len(x))
    edited[np.arange(len(x)),key_position] ^= torch.from_numpy(1<<bit)
    _,reference,base=predictions(native,table,encoder,decoder,membership,x)
    _,edited_reference,changed=predictions(native,table,encoder,decoder,
                                           membership,edited)
    native_effect=edited_reference-reference
    rows={}
    for name,(_,before,_,_) in base.items():
        after=changed[name][1]
        effect=after-before
        rows[name]={}
        for family,ids in FAMILIES.items():
            mask=torch.from_numpy(np.isin(query,ids))
            rows[name][family]=dict(
                edit_path_relative_mse=float((effect[mask]-native_effect[mask]).square().sum()/
                                             native_effect[mask].square().sum().clamp_min(1e-12)),
                edit_effect_sign_agreement=float(((effect[mask]>0)==
                                                  (native_effect[mask]>0)).float().mean()),
                n=int(mask.sum()))
    return rows


def run():
    torch.set_num_threads(1)
    check()
    rows=[]
    for seed in range(5):
        native=ContextualCompositionAttention()
        native.load_state_dict(torch.load(HERE/f'contextual_composition_attention_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        native.eval()
        table=head_table(native)
        flat=FlatSparse(12)
        flat.load_state_dict(torch.load(HERE/f'contextual_composition_groups_seed{seed}_flat_12.pt',
                                        map_location='cpu',weights_only=True))
        flat.eval()
        encoder,decoder=flat_factors(flat)
        fit_x,_,fit_q=examples(20000,20263000+seed)
        fit_h=exact_head(table,fit_x[np.isin(fit_q,FIT_QUERIES)])
        fit_a=fit_h@encoder
        test_x,test_y,test_q=examples(10000,20263100+seed)
        p=np.arange(1,17,dtype=float);p/=p.sum()
        shifted_x,shifted_y,shifted_q=examples(10000,20263200+seed,
                                               token_prob=p)
        methods={}
        for size in (4,6,8):
            membership,fit=fit_groups(fit_a,20263300+seed*10+size,size)
            methods[f'group_size{size}']=dict(fit=fit,
                membership=membership.astype(int).tolist(),
                test=evaluate(native,table,encoder,decoder,membership,
                              test_x,test_y,test_q),
                edits=evaluate_edits(native,table,encoder,decoder,membership,
                                     test_x,test_q,20263400+seed),
                shifted=evaluate(native,table,encoder,decoder,membership,
                                 shifted_x,shifted_y,shifted_q))
        rows.append(dict(seed=seed,methods=methods))
        print('seed',seed,'complete',flush=True)
    summary={}
    for size in (4,6,8):
        name=f'group_size{size}'
        summary[name]={}
        for split in ('test','shifted'):
            summary[name][split]={}
            for variant in rows[0]['methods'][name][split]:
                summary[name][split][variant]={}
                for family in FAMILIES:
                    summary[name][split][variant][family]={key:float(np.mean([
                        row['methods'][name][split][variant][family][key]
                        for row in rows])) for key in ('accuracy','head_relative_mse',
                                                       'relative_logit_mse','mean_active_atoms')}
        summary[name]['edits']={}
        for variant in rows[0]['methods'][name]['edits']:
            summary[name]['edits'][variant]={family:{key:float(np.mean([
                row['methods'][name]['edits'][variant][family][key]
                for row in rows])) for key in ('edit_path_relative_mse',
                                              'edit_effect_sign_agreement')}
                for family in FAMILIES}
    record=dict(description='Five trained 16-task bilinear attention models; flat 12-atom top-four autoencoder dictionary trained only on primitive+AND query activations. Eight overlapping binary groups learned by kmeans of unlabeled top-four supports from new primitive+AND sequences. Group inference selects one or two sets by head reconstruction error, then top-four atoms in their union; shared atoms decoded once. No XOR in fitting or selection. Size-4/6/8 groups; physical top-three/top-four, flat top-four baselines; independent uniform and shifted-token tests; finite single-key bit-toggle logit edit paths.',
                rows=rows,summary=summary)
    (HERE/'contextual_composition_group_supports.json').write_text(
        json.dumps(record,indent=2)+'\n')
    for name,split in summary.items():
        print(name,{variant:(round(families['xor']['accuracy'],4),
                             round(families['xor']['head_relative_mse'],4))
                    for variant,families in split['test'].items()})
    return record


if __name__=='__main__':run()
