"""Resumable small bilinear LM baseline on the jointly sampled V13 stream."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
from model import BilinearGPT, Config
from surface_mixture_v13_attributes import SurfaceMixtureV13Attributes
from surface_mixture_v14_conditional import SurfaceMixtureV14Conditional
from train_v12_price_faq import batch, evaluate


COMMON = dict(counterbalanced_queries=True, question_rate=.05,
              count_rate=0., quote_rate=.08, source_rate=.15,
              section_rate=.06, detail_rate=.8, scope_question_rate=.08,
              source_object_mass=.15, source_subject_mass=.15,
              bracket_rate=.06, linked_subject_mass=.65,
              linked_object_mass=.65, linked_question_mass=.30,
              linked_query_key_mass=.65)
REFERENCE = dict(price_fact_rate=.06, price_question_rate=.12,
                 price_object_mass=.45, attribute_fact_rate=.10,
                 attribute_adjective_mass=.65, attribute_object_mass=.65)
DIAGNOSTIC = dict(price_fact_rate=.12, price_question_rate=.24,
                  price_object_mass=.65, attribute_fact_rate=.20,
                  attribute_adjective_mass=.80, attribute_object_mass=.80)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--length',type=int,default=256)
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=3)
    p.add_argument('--regime',choices=('reference','diagnostic'),
                   default='reference')
    p.add_argument('--generator',choices=('v13','v14'),default='v13')
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--eval-streams',type=int,default=128)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume-checkpoint',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device=torch.device(args.device)
    task_kwargs=dict(COMMON,**(REFERENCE if args.regime=='reference'
                               else DIAGNOSTIC))
    generator_cls=(SurfaceMixtureV13Attributes if args.generator=='v13'
                   else SurfaceMixtureV14Conditional)
    if args.generator=='v14':
        task_kwargs['conditional_continuation_rate']=.35
    train_gen=generator_cls(seed=args.seed+1000,**task_kwargs)
    config=Config(vocab_size=len(train_gen.vocab),n_layer=args.depth,
                  n_head=args.heads,n_embd=args.width,
                  seq_len=args.length-1,
                  block_has_mlp=(False,)*args.depth,
                  use_rmsnorm=True,use_qk_norm=False,use_final_norm=False)
    model=BilinearGPT(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[]; start=0; elapsed=0.
    if args.resume_checkpoint:
        saved=torch.load(args.resume_checkpoint,map_location=device,
                         weights_only=False)
        assert saved['generator']==generator_cls.__name__
        assert saved['config']==vars(config)
        assert saved['regime']==args.regime
        assert saved['batch_size']==args.batch_size
        assert saved['length']==args.length
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        train_gen.rng.setstate(saved['generator_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        records=saved['records'];start=saved['step']
        elapsed=saved['elapsed_seconds']
    clock=time.time()-elapsed

    def snapshot(step):
        fresh=generator_cls(seed=args.seed+2000,**task_kwargs)
        scores=evaluate(model,fresh,args.eval_streams,args.length,device)
        records.append(dict(step=step,elapsed_seconds=time.time()-clock,
                            scores=scores))
        print('step',step,'attribute adjective',
              scores.get('attribute_adjective'),'attribute object',
              scores.get('attribute_object'),'conditional action',
              scores.get('conditional_action'),'mixed verb',
              scores.get('mixed_verb'),flush=True)
        payload=dict(step=step,model=model.state_dict(),
            optimizer=optimizer.state_dict(),config=vars(config),
            generator=generator_cls.__name__,regime=args.regime,
            batch_size=args.batch_size,length=args.length,
            generator_rng=train_gen.rng.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=(torch.cuda.get_rng_state_all()
                      if device.type=='cuda' else None),
            records=records,elapsed_seconds=time.time()-clock)
        temporary=args.output_dir/'resume_latest.pt.tmp'
        torch.save(payload,temporary)
        temporary.replace(args.output_dir/'resume_latest.pt')
        (args.output_dir/'results.json').write_text(json.dumps(dict(
            args={**vars(args),'output_dir':str(args.output_dir),
                  'resume_checkpoint':str(args.resume_checkpoint)
                  if args.resume_checkpoint else None},
            generator_kwargs=task_kwargs,records=records),indent=2)+'\n')

    if start==0 and args.resume_checkpoint is None:
        snapshot(0)
    model.train()
    for step in range(start+1,args.steps+1):
        x,y,_=batch(train_gen,args.batch_size,args.length,device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)[0].float()
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%args.eval_every==0 or step==args.steps:
            snapshot(step)
    torch.save(dict(model=model.state_dict(),config=vars(config),
                    generator=generator_cls.__name__,step=args.steps),
               args.output_dir/'model.pt')


if __name__=='__main__':
    main()
