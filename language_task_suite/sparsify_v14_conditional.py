"""Iterative attention-only L1/prune/no-L1 recovery on competent V14."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
from model import BilinearGPT, Config
from sparsify_original_v3_for_rule_graphs import (
    enforce, global_prune, mask_stats, prunable,
)
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from train_v12_price_faq import batch, evaluate
from train_v13_attributes import COMMON, REFERENCE


def train_phase(model,params,mask,gen,device,steps,batch_size,length,lr,l1):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    model.train()
    for step in range(steps):
        x,y,_=batch(gen,batch_size,length,device)
        opt.zero_grad(set_to_none=True)
        logits=model(x)[0].float()
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        if l1:
            loss=loss+l1*sum(p[mask[name]].abs().sum()
                             for name,p in params.items())
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss in phase at step {step}')
        loss.backward()
        for name,p in params.items():
            p.grad.mul_(mask[name])
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        enforce(params,mask)
    model.eval()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--targets',default='.4,.6')
    p.add_argument('--steps-per-round',type=int,default=80)
    p.add_argument('--final-steps',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--l1',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=99441)
    p.add_argument('--conditional-rate',type=float,
                   help='Override V14 rate for older checkpoints without generator kwargs')
    p.add_argument('--eval-streams',type=int,default=128)
    p.add_argument('--resume-stage',choices=('dense','pruned_40','pruned_60'))
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    saved=torch.load(args.checkpoint,map_location=device,weights_only=True)
    assert saved['generator']=='SurfaceMixtureV14Conditional'
    recorded_rate=saved.get('generator_kwargs',{}).get(
        'conditional_continuation_rate')
    if (args.conditional_rate is not None and recorded_rate is not None
            and args.conditional_rate!=recorded_rate):
        p.error('Requested conditional rate disagrees with checkpoint')
    conditional_rate=(args.conditional_rate if args.conditional_rate is not None
                      else recorded_rate if recorded_rate is not None else .35)
    if not 0 <= conditional_rate < 1:
        p.error('--conditional-rate must be in [0, 1)')
    model=BilinearGPT(Config(**saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    params=prunable(model)
    mask={name:torch.ones_like(weight,dtype=torch.bool)
          for name,weight in params.items()}
    kwargs=dict(COMMON,**REFERENCE,
                conditional_continuation_rate=conditional_rate)
    gen=SurfaceMixtureV14Conditional(seed=args.seed,**kwargs)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    targets=[float(v) for v in args.targets.split(',')]
    assert targets==[.4,.6]
    stages=['dense','pruned_40','pruned_60','final_no_l1']
    rows=[]; start_index=0; elapsed=0.
    if args.resume_stage:
        payload=torch.load(args.output_dir/f'{args.resume_stage}.pt',
                           map_location=device,weights_only=False)
        assert payload['generator']=='SurfaceMixtureV14Conditional'
        assert payload.get('generator_kwargs',kwargs)==kwargs
        assert payload['config']==saved['config']
        assert payload['targets']==targets
        assert payload['steps_per_round']==args.steps_per_round
        assert payload['final_steps']==args.final_steps
        assert payload['batch_size']==args.batch_size
        assert payload['length']==args.length
        assert payload['lr']==args.lr and payload['l1']==args.l1
        model.load_state_dict(payload['model'])
        mask=payload['mask']
        gen.rng.setstate(payload['generator_rng'])
        torch.set_rng_state(payload['torch_rng'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in payload['cuda_rng']])
        rows=payload['rows']
        start_index=stages.index(args.resume_stage)+1
        elapsed=payload['elapsed_seconds']
    clock=time.time()-elapsed

    def record(stage):
        scores=evaluate(model,SurfaceMixtureV14Conditional(
            seed=args.seed+100,**kwargs),args.eval_streams,args.length,device)
        stats=mask_stats(mask)
        total_params=sum(p.numel() for p in model.parameters())
        total_live=total_params-(stats['total']-stats['live'])
        rows.append(dict(stage=stage,attention_mask=stats,
                         total_params=total_params,total_live=total_live,
                         scores=scores,elapsed_seconds=time.time()-clock))
        print(stage,'attention live',stats['live'],'total live',total_live,
              'conditional',scores.get('conditional_action'),
              'attribute object',scores.get('attribute_object'),flush=True)
        payload=dict(stage=stage,model=model.state_dict(),mask=mask,
            config=saved['config'],generator='SurfaceMixtureV14Conditional',
            generator_kwargs=kwargs,
            targets=targets,steps_per_round=args.steps_per_round,
            final_steps=args.final_steps,batch_size=args.batch_size,
            length=args.length,lr=args.lr,l1=args.l1,
            generator_rng=gen.rng.getstate(),torch_rng=torch.get_rng_state(),
            cuda_rng=(torch.cuda.get_rng_state_all()
                      if device.type=='cuda' else None),
            rows=rows,elapsed_seconds=time.time()-clock)
        temporary=args.output_dir/f'{stage}.pt.tmp'
        torch.save(payload,temporary)
        temporary.replace(args.output_dir/f'{stage}.pt')
        (args.output_dir/'results.json').write_text(json.dumps(dict(
            args={**vars(args),'checkpoint':str(args.checkpoint),
                  'output_dir':str(args.output_dir),
                  'conditional_continuation_rate':conditional_rate},rows=rows,
            caveat='Only attention projection entries are pruned. Embedding, '
                   'unembedding, and norm weights stay dense. Fixed-panel '
                   'CE does not establish fresh causal fidelity.'),indent=2)+'\n')

    if start_index==0:
        model.eval();record('dense')
        start_index=1
    for index in range(start_index,len(stages)):
        stage=stages[index]
        if stage=='final_no_l1':
            train_phase(model,params,mask,gen,device,args.final_steps,
                        args.batch_size,args.length,args.lr,0.)
        else:
            train_phase(model,params,mask,gen,device,args.steps_per_round,
                        args.batch_size,args.length,args.lr,args.l1)
            global_prune(params,mask,targets[index-1])
            enforce(params,mask)
        record(stage)


if __name__=='__main__':main()
