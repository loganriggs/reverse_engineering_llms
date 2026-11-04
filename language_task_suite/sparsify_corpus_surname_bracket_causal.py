"""Matched attention L1/prune/recovery with or without bracket-effect loss.

Only training-source bracket pairs and background rows fit the sparse weights.
The evaluation source panel is read after the fit and never contributes loss.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_ov_rank import fidelity
from eval_corpus_rule_reference_transfer import one_kind, pair_for_text
from sparsify_corpus_rule_reference_attention import attention_parameters
from sparsify_original_v3_for_rule_graphs import enforce, global_prune, mask_stats
from train_corpus_delimiter_bilinear import _model


def bracket_pairs(rows, tokenizer, limit, parity_odd=False):
    result = []
    for row in rows:
        if parity_odd and int(row['sha256'][:8], 16) % 2 == 0:
            continue
        pair, _ = pair_for_text(row['preview'], tokenizer, row['index'],
                                'bracket', 256, 2048)
        if pair is not None:
            result.append(pair)
            if len(result) == limit:
                break
    if len(result) < limit:
        raise ValueError(f'Only {len(result)} of {limit} bracket pairs')
    return result


def picked_pairs(pairs, n, generator, device):
    indices = torch.randint(len(pairs), (n,), generator=generator).tolist()
    selected = [pairs[i] for i in indices]
    return {key: torch.stack([row[key] for row in selected]).to(device)
            if key.startswith(('x_', 'y_')) else
            torch.tensor([row[key] for row in selected], device=device)
            for key in ('x_natural', 'x_edit', 'y_natural',
                        'pos_natural', 'pos_edit',
                        'token_natural', 'token_edit')}


def sampled_background(background, n, generator, device):
    indices = torch.randint(len(background['x']), (n,), generator=generator)
    return (background['x'][indices].to(device),
            background['y'][indices].to(device))


def divergence(reference, candidate):
    return F.kl_div(candidate.log_softmax(-1), reference.softmax(-1),
                    reduction='batchmean')


def training_step(teacher, student, params, masks, pairs, background,
                  optimizer, generator, gamma, l1, device, batch_pairs):
    picked = picked_pairs(pairs, batch_pairs, generator, device)
    xb, yb = sampled_background(background, batch_pairs, generator, device)
    x = torch.cat((picked['x_natural'], picked['x_edit'], xb))
    with torch.no_grad():
        target = teacher(x).float()
    output = student(x).float()
    n = batch_pairs
    ix = torch.arange(n, device=device)
    pn, pe = picked['pos_natural'], picked['pos_edit']
    tn, te = picked['token_natural'], picked['token_edit']
    target_n, target_e = target[:n][ix, pn], target[n:2*n][ix, pe]
    output_n, output_e = output[:n][ix, pn], output[n:2*n][ix, pe]
    site_kl = divergence(torch.cat((target_n, target_e)),
                         torch.cat((output_n, output_e)))
    positions = torch.arange(0, xb.shape[1], 32, device=device)
    valid = yb[:, positions] != -100
    background_kl = divergence(target[2*n:, positions][valid],
                               output[2*n:, positions][valid])
    def effect(a, b):
        return ((a[ix, tn]-a[ix, te])-
                (b[ix, tn]-b[ix, te]))
    teacher_effect = effect(target_n, target_e)
    student_effect = effect(output_n, output_e)
    causal_loss = ((student_effect-teacher_effect).square().mean() /
                   (1.+teacher_effect.square().mean().detach()))
    penalty = sum(p[masks[name]].abs().sum() for name,p in params.items())
    loss = site_kl + .25*background_kl + gamma*causal_loss + l1*penalty
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite sparse distillation loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name,param in params.items():
        param.grad.mul_(masks[name])
    torch.nn.utils.clip_grad_norm_(list(params.values()), 1.)
    optimizer.step()
    enforce(params, masks)
    return {name:float(value.detach()) for name,value in
            (('total',loss),('site_kl',site_kl),
             ('background_kl',background_kl),('causal',causal_loss))}


def phase(teacher, student, params, masks, pairs, background,
          generator, gamma, l1, steps, args):
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr,
                                  weight_decay=0.)
    losses = []
    student.train()
    for _ in range(steps):
        losses.append(training_step(teacher, student, params, masks, pairs,
                                    background, optimizer, generator, gamma,
                                    l1, args.device, args.batch_pairs))
    student.eval()
    return {key:sum(row[key] for row in losses[-50:]) /
            min(len(losses),50) for key in losses[0]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--train-preview',type=Path,required=True)
    p.add_argument('--eval-preview',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--train-pairs',type=int,default=256)
    p.add_argument('--eval-pairs',type=int,default=200)
    p.add_argument('--steps-per-round',type=int,default=100)
    p.add_argument('--recovery-steps',type=int,default=250)
    p.add_argument('--batch-pairs',type=int,default=2)
    p.add_argument('--lr',type=float,default=.0002)
    p.add_argument('--l1',type=float,default=1e-5)
    p.add_argument('--sample-seed',type=int,default=85001)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    torch.set_num_threads(4)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if Path(state['identity'][0])!=args.cache.resolve():
        raise ValueError('Checkpoint/cache mismatch')
    _,depth,width,heads=state['identity'][:4]
    teacher=_model(50257,256,depth,width,heads,args.device,'rms')
    teacher.load_state_dict(state['model'])
    teacher.eval().requires_grad_(False)
    cache=torch.load(args.cache,map_location='cpu',weights_only=True)
    tokenizer=Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    train=bracket_pairs(json.loads(args.train_preview.read_text()),
                        tokenizer,args.train_pairs)
    # This is an already-used development source range, never a fit range.
    evaluation=bracket_pairs(json.loads(args.eval_preview.read_text()),
                             tokenizer,args.eval_pairs,parity_odd=True)
    baseline=one_kind(teacher,evaluation,args.device,8,tokenizer)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'config':{k:str(v) if isinstance(v,Path) else v
                      for k,v in vars(args).items()},
            'teacher_step':state['step'],
            'train_pairs':len(train),'eval_pairs':len(evaluation),
            'teacher_behavior':baseline,'arms':{},
            'scope':'Attention q1/k1/q2/k2/v/o only. Fixed 40%, then 60% '
                    'global attention-edge zeros; 100 L1 steps before '
                    'each pruning stage, then 250 masked no-L1 recovery '
                    'steps. Both arms use the same examples, masks rule, '
                    'optimizer, and budget. Causal arm additionally '
                    'matches teacher bracket-edit effect. Evaluation '
                    'source panel is not fit; other weights remain dense. '
                    'This tests sparsity/fidelity, not human legibility.'}
    output=args.output_dir/'results.json'
    for label,gamma in (('distribution',0.),('causal',1.)):
        student=deepcopy(teacher)
        params=attention_parameters(student)
        if len(params)!=6*depth:
            raise AssertionError('Unexpected attention projection count')
        for name,param in student.named_parameters():
            param.requires_grad_(name in params)
        masks={name:torch.ones_like(param,dtype=torch.bool)
               for name,param in params.items()}
        generator=torch.Generator().manual_seed(args.sample_seed)
        stages=[]
        for target in (.4,.6):
            losses=phase(teacher,student,params,masks,train,
                         cache['train']['background'],generator,gamma,
                         args.l1,args.steps_per_round,args)
            global_prune(params,masks,target)
            enforce(params,masks)
            stats=mask_stats(masks)
            stages.append({'stage':f'pruned_{round(target*100)}',
                           'losses':losses,'attention_edges':stats})
            print(label,stages[-1]['stage'],stats,flush=True)
        losses=phase(teacher,student,params,masks,train,
                     cache['train']['background'],generator,gamma,0.,
                     args.recovery_steps,args)
        behavior=one_kind(student,evaluation,args.device,8,tokenizer)
        preserved=fidelity(teacher,student,evaluation,args.device,8)
        stats=mask_stats(masks)
        stages.append({'stage':'recovered_no_l1','losses':losses,
                       'attention_edges':stats,'behavior':behavior,
                       'teacher_fidelity':preserved})
        report['arms'][label]=stages
        torch.save({'model':{k:v.detach().cpu() for k,v in
                             student.state_dict().items()},
                    'masks':{k:v.detach().cpu() for k,v in masks.items()},
                    'teacher_checkpoint':str(args.checkpoint),
                    'config':report['config']},
                   args.output_dir/f'{label}_recovered.pt')
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(label,'recovered','two-way',
              round(behavior['paired_two_way_fraction'],3),
              'effect_MAE',round(preserved['effect_mean_abs_error'],3),
              'background_KL',
              round(preserved['sampled_background_teacher_kl'],3),flush=True)
    print('saved',output,flush=True)


if __name__=='__main__':
    main()
