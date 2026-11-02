"""Can a different linear readout recover held-out answers from residuals?"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from document_world import sample_world
from train import DeepModel, SPECS, encode


@torch.no_grad()
def residuals(model, examples, vocab, device):
    rows = [[] for _ in range(len(model.layers)+1)]
    labels=[]
    for start in range(0,len(examples),128):
        cases=examples[start:start+128]
        x,last,y=encode(cases,vocab,device)[:3]
        index=torch.arange(len(cases),device=device)
        h=model.embed(x)
        rows[0].append(h[index,last].cpu())
        for i,layer in enumerate(model.layers):
            h=layer(h)
            rows[i+1].append(h[index,last].cpu())
        labels.append(y.cpu())
    return [torch.cat(part).double() for part in rows],torch.cat(labels)


def ridge_fit(x,y,n_vocab,lam):
    x=torch.cat((x,torch.ones(len(x),1,dtype=x.dtype)),dim=1)
    target=F.one_hot(y,num_classes=n_vocab).double()
    penalty=torch.eye(x.shape[1],dtype=x.dtype)*lam
    penalty[-1,-1]=0
    return torch.linalg.solve(x.T@x+penalty,x.T@target)


def ridge_accuracy(x,y,w):
    x=torch.cat((x,torch.ones(len(x),1,dtype=x.dtype)),dim=1)
    return float(((x@w).argmax(dim=1)==y).double().mean())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--train-n',type=int,default=12000)
    p.add_argument('--test-n',type=int,default=2000)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',default='language_task_suite/world_residual_probe.json')
    a=p.parse_args()
    torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    k=ck['model_kwargs'];device=torch.device(a.device)
    model=DeepModel(len(ck['vocabulary']),k['d_model'],k['n_head'],
                    SPECS[f"block{ck['depth']}"],ck['n_ctx'],
                    norm=k['norm'],residual=k['residual'],attention=k['attention']).to(device)
    model.load_state_dict(ck['state_dict']);model.eval()
    train=sample_world(a.train_n,9042,genres=(0,1))
    iid=sample_world(a.test_n,9043,genres=(0,1))
    comp_cal=sample_world(a.test_n,9044,split='composition',genres=(0,1))
    comp=sample_world(a.test_n,9045,split='composition',genres=(0,1))
    genre=sample_world(a.test_n,9046,genres=(2,))
    data={name:residuals(model,examples,ck['vocabulary'],device)
          for name,examples in [('train',train),('iid',iid),('comp_cal',comp_cal),
                                ('composition',comp),('held_genre',genre)]}
    rows=[]
    for i in range(len(model.layers)+1):
        train_x,train_y=data['train'][0][i],data['train'][1]
        # Ridge scale tracks feature variance and training count.
        lam=float((train_x.square().sum()/train_x.shape[1]).item())*0.01
        w=ridge_fit(train_x,train_y,len(ck['vocabulary']),lam)
        oracle_x,oracle_y=data['comp_cal'][0][i],data['comp_cal'][1]
        oracle_lam=float((oracle_x.square().sum()/oracle_x.shape[1]).item())*0.01
        oracle=ridge_fit(oracle_x,oracle_y,len(ck['vocabulary']),oracle_lam)
        row=dict(site='embedding' if i==0 else f'after_layer_{i}',
                 train_fit={name:ridge_accuracy(x[i],y,w)
                            for name,(x,y) in data.items() if name!='comp_cal'},
                 composition_oracle_probe=ridge_accuracy(
                     data['composition'][0][i],data['composition'][1],oracle))
        rows.append(row)
        print(row,flush=True)
    output=dict(config=vars(a),rows=rows,
                caveat='The oracle probe uses held-out-combination labels and is diagnostic only; '
                       'its success would not establish blind compositional generalization.')
    Path(a.output).write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
