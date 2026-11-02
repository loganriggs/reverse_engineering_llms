"""Compare weight-only and broad-stream data-aware rank reductions of layer 0.

Only one fixed query/key position pair is considered. Fit/dev/test streams
are disjoint synthetic v8 draws; uniform-vocabulary and unseen-pair errors
are reported separately. This does not evaluate downstream model behavior.
"""

import argparse
import json
from pathlib import Path

import torch

from surface_mixture_v8 import SurfaceMixtureV8


def pairs(seed, n, length, query_position, key_position, kwargs):
    gen=SurfaceMixtureV8(seed=seed,**kwargs)
    ids,_,_=gen.sample_batch(n,length)
    return torch.tensor([(row[query_position],row[key_position]) for row in ids],
                        dtype=torch.long)


def factors_from_svd(matrix, rank):
    u,s,vh=torch.linalg.svd(matrix.double(),full_matrices=False)
    scale=s[:rank].sqrt()
    left=(u[:,:rank]*scale).float().contiguous()
    right=(vh[:rank].T*scale).float().contiguous()
    return left,right


def approximations(score, value_norm, rank):
    score_left,score_right=factors_from_svd(score,rank)
    weighted=score*value_norm[None,:]
    weighted_left,weighted_right=factors_from_svd(weighted,rank)
    output_right=weighted_right/value_norm[:,None].clamp_min(1e-12)
    return dict(score_svd=(score_left,score_right),
                output_svd=(weighted_left,output_right))


def full_matrix(factors):
    return factors[0]@factors[1].T


def relative_error(score, value_norm, factors, panel=None, unseen=None):
    estimate=full_matrix(factors)
    if panel is None:
        weight=value_norm[None,:].square()
        numerator=((estimate-score).square()*weight).sum()
        denominator=(score.square()*weight).sum()
        score_denominator=score.square().sum()
        score_error=(estimate-score).square().sum()/score_denominator.clamp_min(1e-30)
        return dict(n=int(score.numel()),relative_output_l2=float(
            (numerator/denominator.clamp_min(1e-30)).sqrt()),
                    relative_score_l2=float(score_error.sqrt()))
    if unseen is not None:
        panel=panel[unseen]
    if not len(panel):
        return dict(n=0,relative_output_l2=None,relative_score_l2=None)
    a,b=panel[:,0],panel[:,1]
    error=estimate[a,b]-score[a,b]
    weight=value_norm[b].square()
    target=score[a,b]
    output=(error.square()*weight).mean()/(target.square()*weight).mean().clamp_min(1e-30)
    score_ratio=error.square().mean()/target.square().mean().clamp_min(1e-30)
    return dict(n=len(panel),relative_output_l2=float(output.sqrt()),
                relative_score_l2=float(score_ratio.sqrt()))


def fit_data(score, value_norm, rank, initial, train, dev, steps, lr, prior):
    left=torch.nn.Parameter(initial[0].clone())
    right=torch.nn.Parameter(initial[1].clone())
    optimizer=torch.optim.Adam((left,right),lr=lr)
    a,b=train[:,0],train[:,1]
    target=score[a,b]
    weight=value_norm[b].square()
    train_scale=(target.square()*weight).mean().clamp_min(1e-30)
    uniform_scale=(score.square()*value_norm[None,:].square()).mean().clamp_min(1e-30)
    best=(float('inf'),0,(left.detach().clone(),right.detach().clone()))
    curve=[]
    for step in range(steps+1):
        if step%25==0 or step==steps:
            with torch.no_grad():
                dev_error=relative_error(score,value_norm,(left,right),dev)
            value=dev_error['relative_output_l2']
            curve.append(dict(step=step,dev_relative_output_l2=value))
            if value<best[0]:
                best=(value,step,(left.detach().clone(),right.detach().clone()))
        if step==steps:break
        optimizer.zero_grad()
        prediction=(left[a]*right[b]).sum(-1)
        loss=((prediction-target).square()*weight).mean()/train_scale
        if prior:
            dense=left@right.T
            regularizer=((dense-score).square()*value_norm[None,:].square()).mean()/uniform_scale
            loss=loss+prior*regularizer
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite data fit at step {step}')
        loss.backward()
        optimizer.step()
    return best[2],dict(best_dev_relative_output_l2=best[0],best_step=best[1],
                        curve=curve)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-dir',default='language_task_suite/results/v8_scope_balanced_continue_seed101')
    p.add_argument('--tables',default='language_task_suite/results/v8_layer0_token_function_seed1_q31_k7.pt')
    p.add_argument('--query-position',type=int,default=31)
    p.add_argument('--key-position',type=int,default=7)
    p.add_argument('--ranks',default='8,16')
    p.add_argument('--train-streams',type=int,default=4096)
    p.add_argument('--dev-streams',type=int,default=2048)
    p.add_argument('--test-streams',type=int,default=4096)
    p.add_argument('--train-seed',type=int,default=60001)
    p.add_argument('--dev-seed',type=int,default=60002)
    p.add_argument('--test-seed',type=int,default=61001)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--priors',default='0,0.1')
    p.add_argument('--torch-threads',type=int,default=1,
                   help='Single-thread CPU math gives identical fit factors across test panels')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.torch_threads<1:
        p.error('--torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    saved=json.loads((Path(args.model_dir)/'results.json').read_text())
    kwargs=saved['generator_kwargs']
    assert saved['config']['generator']=='v8'
    table_path=Path(args.tables)
    table_json=json.loads(table_path.with_suffix('.json').read_text())
    assert table_json['config']['query_position']==args.query_position
    assert table_json['config']['key_position']==args.key_position
    tables=torch.load(table_path,map_location='cpu',weights_only=True)
    length=args.query_position+1
    train=pairs(args.train_seed,args.train_streams,length,
                args.query_position,args.key_position,kwargs)
    dev=pairs(args.dev_seed,args.dev_streams,length,
              args.query_position,args.key_position,kwargs)
    test=pairs(args.test_seed,args.test_streams,length,
               args.query_position,args.key_position,kwargs)
    vocab=tables['score'].shape[-1]
    train_seen=torch.zeros((vocab,vocab),dtype=torch.bool)
    train_seen[train[:,0],train[:,1]]=True
    unseen=~train_seen[test[:,0],test[:,1]]
    ranks=[int(x) for x in args.ranks.split(',')]
    priors=[float(x) for x in args.priors.split(',')]
    results={}
    checkpoints={}
    for h in range(tables['score'].shape[0]):
        score=tables['score'][h]
        value_norm=tables['ov'][h].norm(dim=-1)
        for rank in ranks:
            key=f'head{h}_rank{rank}'
            initial=approximations(score,value_norm,rank)
            methods=dict(initial)
            metadata={}
            for prior in priors:
                name=f'data_prior{prior:g}'
                methods[name],metadata[name]=fit_data(
                    score,value_norm,rank,initial['output_svd'],train,dev,
                    args.steps,args.lr,prior)
            scores={}
            for name,factors in methods.items():
                scores[name]=dict(
                    train=relative_error(score,value_norm,factors,train),
                    dev=relative_error(score,value_norm,factors,dev),
                    test=relative_error(score,value_norm,factors,test),
                    unseen_test=relative_error(score,value_norm,factors,test,unseen),
                    uniform_vocab=relative_error(score,value_norm,factors))
            results[key]=dict(head=h,rank=rank,optimization=metadata,scores=scores)
            checkpoints[key]=methods
            print(key,{name:round(row['test']['relative_output_l2'],4)
                       for name,row in scores.items()},flush=True)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint=path.with_suffix('.pt')
    torch.save(checkpoints,checkpoint)
    output=dict(config=vars(args),factor_checkpoint=str(checkpoint),
                sample_counts=dict(train=len(train),dev=len(dev),test=len(test),
                                   train_unique_pairs=int(train_seen.sum()),
                                   unseen_test_occurrences=int(unseen.sum()),
                                   unseen_test_unique_pairs=len(set(map(tuple,test[unseen].tolist())))),
                results=results,
                caveat='Data-aware fitting uses broad v8 streams at one fixed '
                       'position pair and output-vector norm weights; dev '
                       'selects checkpoints, test and uniform vocab do not. '
                       'This is pair-function error, not full-model KL or '
                       'causal intervention fidelity.')
    path.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
