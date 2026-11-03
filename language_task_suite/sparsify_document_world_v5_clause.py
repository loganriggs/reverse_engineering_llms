"""Attention-only L1/prune/no-L1 recovery for learned source attribution."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

OLD_REPO=Path(__file__).resolve().parents[2]/'reverse_engineer/tensor_language'
sys.path.insert(0,str(OLD_REPO))
from deep_model import DeepModel,SPECS  # noqa: E402
from document_world_v5_clause_realizations import sample_documents  # noqa: E402
from sparsify_original_v3_for_rule_graphs import (  # noqa: E402
    enforce,global_prune,mask_stats,
)
from train_document_world_v5_clause import encode,evaluate  # noqa: E402


ATTN_PARTS={'q1','k1','q2','k2','v','o'}


def prunable(model):
    return {name:p for name,p in model.named_parameters()
            if name.startswith('layers.') and len(name.split('.'))==4
            and name.split('.')[2] in ATTN_PARTS and name.endswith('.weight')}


def phase(model,params,mask,data,device,steps,batch_size,lr,l1,seed):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    rng=random.Random(seed)
    losses=[]
    model.train()
    for _ in range(steps):
        idx=torch.tensor(rng.sample(range(data[0].shape[0]),batch_size))
        x,y=(v[idx].to(device) for v in data)
        opt.zero_grad(set_to_none=True)
        logits=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100)
        if l1:
            loss=loss+l1*sum(p[mask[name]].abs().sum()
                                 for name,p in params.items())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite sparse training loss')
        loss.backward()
        for name,p in params.items():
            p.grad.mul_(mask[name])
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        enforce(params,mask)
        losses.append(loss.item())
    model.eval()
    return sum(losses[-50:])/min(len(losses),50)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--targets',default='.4,.6')
    p.add_argument('--steps-per-round',type=int,default=80)
    p.add_argument('--final-steps',type=int,default=200)
    p.add_argument('--train-n',type=int,default=8000)
    p.add_argument('--test-n',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--l1',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=34441)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    torch.set_num_threads(4)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert saved['depth']==2
    assert a.train_n==saved['config']['train_n']
    assert a.test_n==saved['config']['test_n']
    kwargs=saved['generator_kwargs']
    vocab=saved['vocabulary'];n_ctx=saved['n_ctx']
    train=sample_documents(a.train_n,saved['config']['seed'],**kwargs)
    test=sample_documents(a.test_n,saved['config']['seed']+100000,**kwargs)
    train_data=encode(train,vocab,n_ctx)
    test_data=encode(test,vocab,n_ctx)
    model=DeepModel(len(vocab),saved['model_kwargs']['d_model'],
                    saved['model_kwargs']['n_head'],SPECS['block2'],n_ctx,
                    norm=saved['model_kwargs']['norm'],
                    residual=saved['model_kwargs']['residual'],
                    attention=saved['model_kwargs']['attention']).to(a.device)
    model.load_state_dict(saved['state_dict'])
    params=prunable(model)
    assert len(params)==12
    mask={name:torch.ones_like(weight,dtype=torch.bool)
          for name,weight in params.items()}
    targets=[float(v) for v in a.targets.split(',')]
    assert targets==[.4,.6]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    clock=time.time();rows=[]

    def record(stage,train_loss=None):
        scores=evaluate(model,test,test_data,vocab,a.device)
        stats=mask_stats(mask)
        total=sum(p.numel() for p in model.parameters())
        live=total-(stats['total']-stats['live'])
        row=dict(stage=stage,train_loss_last50=train_loss,
                 attention_mask=stats,total_params=total,total_live=live,
                 evaluation=scores,seconds=round(time.time()-clock,2))
        rows.append(row)
        payload=dict(state_dict={k:v.detach().cpu() for k,v in
                                 model.state_dict().items()},
                     mask={k:v.detach().cpu() for k,v in mask.items()},
                     vocabulary=vocab,n_ctx=n_ctx,depth=2,
                     model_kwargs=saved['model_kwargs'],
                     generator_kwargs=kwargs,config=saved['config'],
                     sparsity_config={**vars(a),'checkpoint':str(a.checkpoint),
                                      'output_dir':str(a.output_dir)},
                     evaluation=row)
        torch.save(payload,a.output_dir/f'{stage}.pt')
        (a.output_dir/'results.json').write_text(json.dumps(dict(
            config=payload['sparsity_config'],rows=rows,
            caveat='Only q1/k1/q2/k2/v/o attention matrices are pruned; '
                   'MLP, embedding, unembedding and norms stay dense. '
                   'Fresh causal edits are scored separately.'),indent=2)+'\n')
        print(stage,'attention live',stats['live'],
              'source',scores['by_site'].get('source_attribution'),flush=True)

    model.eval();record('dense')
    for i,target in enumerate(targets):
        loss=phase(model,params,mask,train_data,a.device,a.steps_per_round,
                   a.batch_size,a.lr,a.l1,a.seed+i)
        global_prune(params,mask,target)
        enforce(params,mask)
        record(f'pruned_{round(target*100)}',loss)
    loss=phase(model,params,mask,train_data,a.device,a.final_steps,
               a.batch_size,a.lr,0.,a.seed+100)
    record('final_no_l1',loss)


if __name__=='__main__':main()
