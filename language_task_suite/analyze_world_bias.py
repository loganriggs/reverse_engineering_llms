"""Check whether held-out-combination errors favor allowed training pairs."""

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from document_world import composition_fold, sample_world
from train import DeepModel, SPECS, encode


@torch.no_grad()
def predictions(model, examples, vocab, device):
    inverse = {i:w for w,i in vocab.items()}
    out=[]
    for start in range(0,len(examples),128):
        batch=examples[start:start+128]
        x,last=encode(batch,vocab,device)[:2]
        logits=model(x)[torch.arange(len(batch),device=device),last]
        out.extend(inverse[i] for i in logits.argmax(-1).tolist())
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--n',type=int,default=3000)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    k=ck['model_kwargs'];device=torch.device(a.device)
    model=DeepModel(len(ck['vocabulary']),k['d_model'],k['n_head'],
                    SPECS[f"block{ck['depth']}"],ck['n_ctx'],
                    norm=k['norm'],residual=k['residual'],attention=k['attention']).to(device)
    model.load_state_dict(ck['state_dict']);model.eval()
    results={}
    for split,seed in [('iid',2043),('composition',2044)]:
        examples=sample_world(a.n,seed,split='composition' if split=='composition' else 'train')
        pred=predictions(model,examples,ck['vocabulary'],device)
        rows=defaultdict(lambda:[0,0,0,0])
        for ex,answer in zip(examples,pred):
            row=rows[ex.family]
            row[0]+=1
            row[1]+=int(answer==ex.target)
            row[2]+=int(answer in ex.candidates)
            if answer in ex.candidates:
                row[3]+=int(composition_fold(replace(ex,target=answer))==0)
        results[split]={f:dict(n=n,accuracy=c/n,answer_type_rate=t/n,
                               predicted_heldout_pair_rate=h/max(t,1))
                        for f,(n,c,t,h) in rows.items()}
    output=dict(checkpoint=a.checkpoint,results=results,
                note='On composition examples the true pair is always in held-out fold 0; '
                     'predicted_heldout_pair_rate measures whether the model emits an allowed training-pair answer instead.')
    print(json.dumps(output,indent=2))
    if a.output:Path(a.output).write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
