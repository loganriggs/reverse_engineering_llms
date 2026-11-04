"""Small bilinear LM on corpus-carried surname reuse, with paired edit gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model

@torch.no_grad()
def evaluate(model, data, device, batch_size=8):
    model.eval()
    n = len(data[0]['x'])
    paired_top1 = paired_rank = response = 0
    top1 = rank = 0
    site_nll = token_nll = token_count = tv = 0.
    for start in range(0,n,batch_size):
        arms = [{k:v[start:start+batch_size].to(device)
                 for k,v in arm.items()} for arm in data]
        outputs = [model(arm['x']).float() for arm in arms]
        b = len(arms[0]['x'])
        ix = torch.arange(b,device=device)
        logits = [out[ix,arm['site']] for out,arm in zip(outputs,arms)]
        target_a,target_b = (arm['target'] for arm in arms)
        good_top1 = [(z.argmax(-1)==arm['target'])
                     for z,arm in zip(logits,arms)]
        before = logits[0][ix,target_a]-logits[0][ix,target_b]
        after = logits[1][ix,target_a]-logits[1][ix,target_b]
        good_rank = (before>0,after<0)
        top1 += sum(g.sum().item() for g in good_top1)
        rank += sum(g.sum().item() for g in good_rank)
        paired_top1 += (good_top1[0]&good_top1[1]).sum().item()
        paired_rank += (good_rank[0]&good_rank[1]).sum().item()
        response += (before>after).sum().item()
        tv += .5*(logits[0].softmax(-1)-
                  logits[1].softmax(-1)).abs().sum(-1).sum().item()
        for arm,out,z in zip(arms,outputs,logits):
            site_nll += F.cross_entropy(z,arm['target'],
                                       reduction='sum').item()
            token_nll += F.cross_entropy(out.flatten(0,1),
                                         arm['y'].flatten(),
                                         ignore_index=-100,
                                         reduction='sum').item()
            token_count += (arm['y']!=-100).sum().item()
    return {'paired_examples':n,'paired_top1':paired_top1/n,
            'top1_mean':top1/(2*n),
            'candidate_rank_paired':paired_rank/n,
            'candidate_rank_mean':rank/(2*n),
            'edit_logit_direction':response/n,
            'answer_site_nll':site_nll/(2*n),
            'token_nll':token_nll/token_count,
            'mean_edit_tv':tv/n}


def _save(path, model, optimizer, rng, step, history, identity):
    state = {'identity': identity, 'step': step, 'history': history,
             'model': {k: v.detach().cpu()
                       for k,v in model.state_dict().items()},
             'optimizer': optimizer.state_dict(),
             'rng': rng.get_state(),
             'cuda_rng': torch.cuda.get_rng_state_all()
                         if next(model.parameters()).is_cuda else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--site-weight', type=float, default=1.)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--seed', type=int, default=45251)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    if a.site_weight < 1 or a.depth not in (1,2,3,4):
        raise ValueError('Invalid site weight or depth')
    torch.set_num_threads(4)
    cache = torch.load(a.cache, map_location='cpu', weights_only=True)
    n_ctx, vocab = cache['n_ctx'], cache['vocab_size']
    train, val, test = (cache['pairs'][name]
                        for name in ('train','val','test'))
    flat = {key: torch.cat([arm[key] for arm in train])
            for key in ('x','y','site','target')}
    torch.manual_seed(a.seed)
    model = _model(vocab, n_ctx, a.depth, a.d_model,
                   a.heads, a.device, 'rms')
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = torch.Generator().manual_seed(a.seed+1000)
    identity = (str(a.cache.resolve()), a.depth, a.d_model, a.heads,
                a.batch_size, a.site_weight, a.lr, a.seed, vocab, n_ctx)
    if a.state.exists():
        saved = torch.load(a.state, map_location='cpu', weights_only=True)
        if tuple(saved['identity']) != identity or saved['step'] > a.steps:
            raise ValueError('Resume identity mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng'])
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step_start, history = saved['step'], saved['history']
        print('resumed', step_start, flush=True)
    else:
        history = {'0': {'val': evaluate(model, val, a.device,
                                        a.eval_batch_size),
                         'test': evaluate(model, test, a.device,
                                          a.eval_batch_size)}}
        step_start = 0
    report = {'config':{k:str(v) if isinstance(v,Path) else v
                        for k,v in vars(a).items()},
              'parameters':model.n_params,
              'source_counts':cache['counts'],
              'history':history,
              'scope':cache.get('task_description',
                      'Whole-source heldout corpus surname substitution. '
                      'Full next-token CE with specified extra site weight; '
                      'target pair rank and earlier-name edit response are '
                      'separate behavioral gates. Natural background text '
                      'has no all-token oracle, and source identity was '
                      'screened heuristically.')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if step_start == 0:
        _save(a.state, model, optimizer, rng, 0, history, identity)
    for step in range(step_start, a.steps):
        model.train()
        ids = torch.randint(len(flat['x']), (a.batch_size,), generator=rng)
        x, y = flat['x'][ids].to(a.device), flat['y'][ids].to(a.device)
        sites = flat['site'][ids].to(a.device)
        logits = model(x).float()
        per = F.cross_entropy(logits.flatten(0,1), y.flatten(),
                              ignore_index=-100,
                              reduction='none').view_as(y)
        weights = (y != -100).float()
        weights[torch.arange(a.batch_size, device=a.device), sites] = \
            a.site_weight
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss step {step+1}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        completed = step+1
        if completed % a.eval_every == 0 or completed == a.steps:
            history[str(completed)] = {
                'train_loss_last':float(loss.detach()),
                'val':evaluate(model,val,a.device,a.eval_batch_size),
                'test':evaluate(model,test,a.device,a.eval_batch_size)}
            report['history'] = history
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            _save(a.state, model, optimizer, rng, completed, history, identity)
            print(completed, 'test paired rank',
                  round(history[str(completed)]['test'][
                      'candidate_rank_paired'],3),
                  'test paired top1',
                  round(history[str(completed)]['test'][
                      'paired_top1'],3),flush=True)
    report['history']=history
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
