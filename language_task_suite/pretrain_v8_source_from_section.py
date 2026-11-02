"""Short source-key curriculum from an already section-trained v8 teacher.

This is a curriculum intervention, not a natural-language sample. Its
speaker/object pair split applies only to this stage: the initializing
joint model has already seen all pairings.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pretrain_joint_source_lookup import cases, check_oracle, encode, evaluate
from surface_mixture_v8 import SurfaceMixtureV8


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--init-model-dir',required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--dev-cases',type=int,default=512)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--seed',type=int,default=401)
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0,a.source)
    from model import BilinearGPT,Config
    source_dir=Path(a.init_model_dir)
    saved=json.loads((source_dir/'results.json').read_text())
    assert saved['config']['generator']=='v8'
    record=saved['records'][0]
    gen=SurfaceMixtureV8(seed=0,**saved['generator_kwargs'])
    config=Config(**record['model_config'])
    assert config.vocab_size==len(gen.vocab) and config.seq_len>=56
    device=torch.device(a.device)
    model=BilinearGPT(config).to(device)
    model.load_state_dict(torch.load(source_dir/f"model_depth{record['depth']}.pt",
                                     map_location=device,weights_only=True))
    allowed=cases(a.dev_cases,17001+a.seed,'train')
    heldout=cases(a.dev_cases,18001+a.seed,'pair_holdout')
    check_oracle(gen,allowed[:64])
    check_oracle(gen,heldout[:64])
    a.output_dir.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    rng=np.random.default_rng(20000+a.seed)
    snapshots=[]
    for step in range(a.steps+1):
        if step==0 or step%a.eval_every==0 or step==a.steps:
            with torch.no_grad():
                scores=dict(allowed=evaluate(model,gen,allowed,device),
                            heldout=evaluate(model,gen,heldout,device))
            snapshots.append(dict(step=step,scores=scores))
            torch.save(model.state_dict(),
                       a.output_dir/f'model_depth{record["depth"]}_step{step}.pt')
            print(step,{kind:(round(row['accuracy'],3),
                              round(row['query']['effect_l1'],3))
                        for kind,row in scores.items()},flush=True)
        if step==a.steps:break
        model.train()
        sample=cases(a.batch,int(rng.integers(2**31)),'train')
        x,y=encode(gen,sample)
        xt=torch.tensor(x,dtype=torch.long,device=device)
        yt=torch.tensor(y,dtype=torch.long,device=device)
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(xt)[0][:,-1].float(),yt)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite short-source loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
    torch.save(model.state_dict(),
               a.output_dir/f'model_depth{record["depth"]}.pt')
    (a.output_dir/'results.json').write_text(json.dumps(dict(
        config={**vars(a),'output_dir':str(a.output_dir)},
        model_config=record['model_config'],
        generator_kwargs=saved['generator_kwargs'],
        snapshots=snapshots,
        caveat='Short V8-supported source curriculum. The initializing '
               'model already saw all speaker/object combinations in joint '
               'training; stage-specific pair splits are not whole-training '
               'holdouts. Short-case success does not establish long-context '
               'source binding or retained section competence.'),indent=2)+'\n')


if __name__=='__main__':main()
