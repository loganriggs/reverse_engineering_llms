"""Resumable bilinear LM training on the enriched joint V12 stream."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
from model import BilinearGPT, Config
from surface_mixture_v12_price_faq import SurfaceMixtureV12PriceFAQ


COMMON = dict(counterbalanced_queries=True, question_rate=.05, count_rate=0.,
              quote_rate=.08, source_rate=.15, section_rate=.06,
              detail_rate=.8, scope_question_rate=.08,
              source_object_mass=.15, source_subject_mass=.15,
              bracket_rate=.06, linked_subject_mass=.65,
              linked_object_mass=.65, linked_question_mass=.30,
              linked_query_key_mass=.65)
ENRICHED = dict(price_fact_rate=.18, price_question_rate=.40,
                price_object_mass=.80)
REFERENCE = dict(price_fact_rate=.08, price_question_rate=.18,
                 price_object_mass=.55)


def batch(gen, n, length, device):
    examples = [gen.sample_sequence(length) for _ in range(n)]
    ids = torch.tensor([row[0] for row in examples], device=device)
    return ids[:, :-1], ids[:, 1:], [row[1][1:] for row in examples]


@torch.no_grad()
def evaluate(model, gen, nstreams, length, device):
    model.eval()
    totals = defaultdict(lambda: [0., 0])
    for start in range(0, nstreams, 32):
        x, y, labels = batch(gen, min(32,nstreams-start), length, device)
        logits = model(x)[0].float()
        ce = F.cross_entropy(logits.flatten(0,1), y.flatten(), reduction='none')
        ce = ce.reshape_as(y)
        for b, row in enumerate(labels):
            for t, label in enumerate(row):
                totals[label][0] += float(ce[b,t])
                totals[label][1] += 1
    model.train()
    return {name:dict(ce=total/count,n=count)
            for name,(total,count) in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--price-answer-weight', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--eval-streams', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.price_answer_weight < 1:
        parser.error('price-answer-weight must be at least 1')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_gen = SurfaceMixtureV12PriceFAQ(seed=args.seed+1000,
                                           **COMMON, **ENRICHED)
    config = Config(vocab_size=len(train_gen.vocab), n_layer=args.depth,
                    n_head=args.heads, n_embd=args.width,
                    seq_len=args.length-1,
                    block_has_mlp=(False,)*args.depth,
                    use_rmsnorm=True, use_qk_norm=False,
                    use_final_norm=False)
    model = BilinearGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,
                                  weight_decay=0.)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    records = []
    start = 0
    elapsed = 0.
    if args.resume_checkpoint:
        saved = torch.load(args.resume_checkpoint,map_location=device,
                           weights_only=False)
        assert saved['generator'] == 'SurfaceMixtureV12PriceFAQ'
        assert saved['config'] == vars(config)
        assert saved['batch_size'] == args.batch_size
        assert saved['length'] == args.length
        assert saved.get('price_answer_weight',1.) == args.price_answer_weight
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        train_gen.rng.setstate(saved['generator_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        records = saved['records']
        start = saved['step']
        elapsed = saved['elapsed_seconds']
    clock = time.time()-elapsed

    def snapshot(step):
        scores = evaluate(model,SurfaceMixtureV12PriceFAQ(
            seed=args.seed+2000,**COMMON,**ENRICHED),
            args.eval_streams,args.length,device)
        records.append(dict(step=step,elapsed_seconds=time.time()-clock,
                            scores=scores))
        print('step',step,'price-answer',scores.get('price_answer'),
              'priced-object',scores.get('priced_item_object'),
              'mixed-verb',scores.get('mixed_verb'),flush=True)
        payload=dict(step=step,model=model.state_dict(),
            optimizer=optimizer.state_dict(),config=vars(config),
            generator='SurfaceMixtureV12PriceFAQ',
            batch_size=args.batch_size,length=args.length,
            price_answer_weight=args.price_answer_weight,
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
            generator_kwargs=dict(COMMON,**ENRICHED),
            reference_kwargs=dict(COMMON,**REFERENCE),
            records=records),indent=2)+'\n')

    if start == 0:
        snapshot(0)
    model.train()
    for step in range(start+1,args.steps+1):
        x,y,labels = batch(train_gen,args.batch_size,args.length,device)
        optimizer.zero_grad(set_to_none=True)
        logits=model(x)[0].float()
        if args.price_answer_weight == 1:
            loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        else:
            ce=F.cross_entropy(logits.flatten(0,1),y.flatten(),
                               reduction='none').reshape_as(y)
            weights=torch.tensor([
                [args.price_answer_weight if label=='price_answer' else 1.
                 for label in row] for row in labels],device=device)
            loss=(ce*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            snapshot(step)
    torch.save(dict(model=model.state_dict(),config=vars(config),
                    generator='SurfaceMixtureV12PriceFAQ',step=args.steps),
               args.output_dir/'model.pt')


if __name__ == '__main__':
    main()
