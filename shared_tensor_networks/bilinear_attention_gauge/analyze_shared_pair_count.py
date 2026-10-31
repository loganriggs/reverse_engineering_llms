"""Exact table and rank analysis for two parity rules sharing B."""

import json
import numpy as np
import torch

from analyze_three_count_language import flatten,head_tables
from dual_joint_function_canonical import HERE
from three_count_metric import pair_weights,weighted_fit
from three_count_task_fit import count_features
from train_shared_pair_count import SharedPairCountModel,targets
from train_three_count_language import examples


def logits(matrix,bias,q,pb,tokens):
    counts=count_features(tokens).astype(np.float64)
    query=(tokens[:,-1]-8).astype(int)
    bits=bias+np.einsum('nk,nok->no',counts,matrix.reshape(2,3,8)[query])
    parity=np.einsum('ni,pij,nj->np',bits,q,bits)+pb
    return np.c_[bits,parity]


def score(output,truth,attrs,reference=None):
    result={}
    for name,mask in (('all',np.ones(len(truth),dtype=bool)),
                      ('rare_heldout',attrs['heldout'])):
        pred=output[mask]>0
        result[name]=dict(n=int(mask.sum()),
            five_output_accuracy=float(np.mean(np.all(pred==truth[mask],axis=1))),
            bits_accuracy=float(np.mean(np.all(pred[:,:3]==truth[mask,:3],axis=1))),
            pair_accuracy=[float(np.mean(pred[:,3+j]==truth[mask,3+j])) for j in range(2)])
    if reference is not None:
        result['relative_output_mse']=float(np.sum((output-reference)**2)/
                                             np.maximum(np.sum(reference**2),1e-20))
    return result


def run(seeds=range(10)):
    torch.set_num_threads(1)
    rows=[]
    for seed in seeds:
        train,_,_=examples(20000,510000+seed,True)
        test,bits,attrs=examples(30000,530000+seed,False)
        tokens=test.numpy();truth=targets(bits).numpy().astype(bool)
        model=SharedPairCountModel().double()
        model.load_state_dict(torch.load(HERE/f'shared_pair_count_seed{seed}.pt',
                                         map_location='cpu',weights_only=True))
        model.eval()
        matrix=flatten(sum(head_tables(model.backbone)))
        bias=model.backbone.read.bias.detach().numpy()
        scale=model.logit_scale.detach().numpy()
        q=model.quadratic.detach().numpy()
        q=.5*(q+q.transpose(0,2,1))/scale[None,:,None]/scale[None,None,:]
        pb=model.parity_bias.detach().numpy()
        with torch.no_grad():direct=model(test[:256]).numpy()
        proxy=logits(matrix,bias,q,pb,tokens[:256])
        gate=float(np.max(np.abs(direct-proxy)))
        assert gate<1e-9,gate
        native=logits(matrix,bias,q,pb,tokens)
        u,s,v=np.linalg.svd(matrix,full_matrices=False)
        cuts={'native':matrix}
        for rank in (2,3,4):
            cuts[f'uniform_rank{rank}']=(u[:,:rank]*s[:rank])@v[:rank]
            cuts[f'empirical_rank{rank}'],_=weighted_fit(
                matrix,pair_weights(train.numpy()),rank)
        scores={}
        for name,cut in cuts.items():
            output=logits(cut,bias,q,pb,tokens)
            result=score(output,truth,attrs,native)
            result['uniform_table_mse']=float(np.sum((cut-matrix)**2)/
                                              np.sum(matrix**2))
            scores[name]=result
        rows.append(dict(seed=seed,exact_forward_gate=gate,
            singular_values=s.tolist(),quadratic_effective=q.tolist(),
            scores=scores,
            matrices={k:cut.tolist() for k,cut in cuts.items()
                      if k!='native'}))
        print(seed,scores['native']['all']['five_output_accuracy'],
              scores['uniform_rank3']['rare_heldout']['five_output_accuracy'],flush=True)
    result=dict(seeds=list(seeds),task='count A/B/C, A XOR B, B XOR C',
        exact_object='6x8 first-layer attention table plus two learned quadratic readouts',
        labels_used_in_reduction=False,rows=rows)
    (HERE/'shared_pair_count_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':run()
