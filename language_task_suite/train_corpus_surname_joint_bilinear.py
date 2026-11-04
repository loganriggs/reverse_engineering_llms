"""Resumable bilinear LM on mixed FineWeb rule sites and ordinary prose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model
from train_fineweb_surname_copy_bilinear import evaluate as evaluate_name
from corpus_surname_joint_dgp import KINDS


TRAIN_KIND_BATCH = {'background':2, 'name_pairs':1,
                    'bracket':2, 'literal_copy':1, 'natural_style':1}
SITE_WEIGHTS = {'surname_copy':500., 'bracket':200.,
                'literal_copy':200., 'natural_style':100., 'list':100.}


def sample(rows, count, rng):
    if rows is None:
        raise ValueError('Empty training stratum')
    indices = torch.randint(len(rows['x']),(count,),generator=rng)
    return {key:value[indices] for key,value in rows.items()}


def sampled_batch(section, rng):
    out = [sample(section['background'],TRAIN_KIND_BATCH['background'],rng)]
    a,b = section['name_pairs']
    i = torch.randint(len(a['x']),(TRAIN_KIND_BATCH['name_pairs'],),
                      generator=rng)
    out += [{key:value[i] for key,value in arm.items()} for arm in (a,b)]
    for kind in ('bracket','literal_copy','natural_style'):
        out.append(sample(section['sites'][kind],TRAIN_KIND_BATCH[kind],rng))
    return {key:torch.cat([part[key] for part in out])
            for key in ('x','y','positions','targets')}


def weighted_loss(model,batch,device,pair_loss_weight=0.):
    x,y,positions = (batch[key].to(device) for key in
                     ('x','y','positions'))
    output = model(x).float()
    per = F.cross_entropy(output.flatten(0,1),y.flatten(),
                          ignore_index=-100,reduction='none').view_as(y)
    weights = (y != -100).float()
    for j,kind in enumerate(KINDS):
        valid = positions[:,j] >= 0
        if valid.any():
            rows = torch.arange(len(x),device=device)[valid]
            cols = positions[valid,j]
            weights[rows,cols] = torch.maximum(weights[rows,cols],
                torch.full_like(weights[rows,cols],SITE_WEIGHTS[kind]))
    loss = (per*weights).sum()/weights.sum()
    if pair_loss_weight:
        # The fixed eight-row allocation puts the two counterfactual surname
        # arms at rows 2 and 3. Compare their *same two* answer tokens.
        name_a,name_b = 2,3
        pos_a,pos_b = positions[name_a,0],positions[name_b,0]
        target_a,target_b = (batch['targets'][i,0].to(device)
                             for i in (name_a,name_b))
        if pos_a < 0 or pos_b < 0 or target_a == target_b:
            raise AssertionError('Invalid paired surname batch')
        logits_a,logits_b = output[name_a,pos_a],output[name_b,pos_b]
        paired = .5*(F.softplus(logits_a[target_b]-logits_a[target_a])+
                     F.softplus(logits_b[target_a]-logits_b[target_b]))
        loss = loss + pair_loss_weight*paired
    return loss


@torch.no_grad()
def evaluate_site(model,rows,j,device,batch_size):
    if rows is None:
        return None
    total = top1 = 0
    nll = 0.
    model.eval()
    for start in range(0,len(rows['x']),batch_size):
        x = rows['x'][start:start+batch_size].to(device)
        pos = rows['positions'][start:start+batch_size,j].to(device)
        target = rows['targets'][start:start+batch_size,j].to(device)
        assert bool((pos>=0).all())
        logits = model(x).float()[torch.arange(len(x),device=device),pos]
        top1 += (logits.argmax(-1)==target).sum().item()
        nll += F.cross_entropy(logits,target,reduction='sum').item()
        total += len(x)
    return {'n':total,'top1':top1/total,'site_nll':nll/total}


@torch.no_grad()
def evaluate_background(model,rows,device,batch_size,max_rows=128):
    total_loss = total_tokens = 0.
    model.eval()
    n = min(len(rows['x']),max_rows)
    for start in range(0,n,batch_size):
        x = rows['x'][start:start+batch_size].to(device)
        y = rows['y'][start:start+batch_size].to(device)
        logits = model(x).float()
        total_loss += F.cross_entropy(logits.flatten(0,1),y.flatten(),
                                      ignore_index=-100,
                                      reduction='sum').item()
        total_tokens += (y!=-100).sum().item()
    return {'rows':n,'token_nll':total_loss/total_tokens}


@torch.no_grad()
def evaluate(model,section,device,batch_size):
    names = section['name_pairs']
    named = ({key:value for key,value in arm.items()
              if key in ('x','y')} |
             {'site':arm['positions'][:,0],
              'target':arm['targets'][:,0]}
             for arm in names)
    name_result = evaluate_name(model,tuple(named),device,batch_size) \
        if all(arm is not None for arm in names) else None
    return {'background':evaluate_background(model,section['background'],
                                             device,batch_size),
            'surname_copy':name_result,
            'sites':{kind:evaluate_site(model,section['sites'][kind],j,
                                        device,batch_size)
                     for j,kind in enumerate(KINDS) if j>0}}


def save(path,model,optimizer,rng,step,history,identity):
    state = {'identity':identity,'step':step,'history':history,
             'model':{key:value.detach().cpu()
                      for key,value in model.state_dict().items()},
             'optimizer':optimizer.state_dict(),'rng':rng.get_state(),
             'cuda_rng':torch.cuda.get_rng_state_all()
                        if next(model.parameters()).is_cuda else []}
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    torch.save(state,temp)
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--d-model',type=int,default=64)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--pair-loss-weight',type=float,default=0.)
    p.add_argument('--seed',type=int,default=46451)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a=p.parse_args()
    if (a.batch_size != 8 or a.depth not in (1,2,3,4) or
            a.pair_loss_weight < 0):
        raise ValueError('Only fixed eight-row allocation and depth1–4')
    torch.set_num_threads(4)
    cache=torch.load(a.cache,map_location='cpu',weights_only=True)
    for split in ('train','val','test'):
        assert cache[split]['background'] is not None
        assert all(arm is not None for arm in cache[split]['name_pairs'])
    torch.manual_seed(a.seed)
    model=_model(50257,256,a.depth,a.d_model,a.heads,a.device,'rms')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    rng=torch.Generator().manual_seed(a.seed+1000)
    identity=(str(a.cache.resolve()),a.depth,a.d_model,a.heads,a.lr,a.seed,
              tuple(TRAIN_KIND_BATCH.items()),tuple(SITE_WEIGHTS.items()))
    if a.pair_loss_weight:
        identity += (a.pair_loss_weight,)
    if a.state.exists():
        saved=torch.load(a.state,map_location='cpu',weights_only=True)
        if tuple(saved['identity'])!=identity or saved['step']>a.steps:
            raise ValueError('Resume identity mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['rng'])
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start,history=saved['step'],saved['history']
        print('resumed',start,flush=True)
    else:
        start=0
        history={'0':{'val':evaluate(model,cache['val'],a.device,a.batch_size),
                      'test':evaluate(model,cache['test'],a.device,a.batch_size)}}
    report={'config':{key:str(value) if isinstance(value,Path) else value
                      for key,value in vars(a).items()},
            'parameters':model.n_params,
            'train_allocation':TRAIN_KIND_BATCH,
            'site_weights':SITE_WEIGHTS,'history':history,
            'scope':'One ordinary background chunk from every accepted '
                    'source plus stratified marked-site rows. All eight '
                    'rows have full next-token CE. Surname rows are paired '
                    'edits; other-rule site top1 is observational and is '
                    'not a causal circuit gate. Heldout sources disjoint.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if start==0:
        save(a.state,model,optimizer,rng,0,history,identity)
    for step in range(start,a.steps):
        model.train()
        batch=sampled_batch(cache['train'],rng)
        loss=weighted_loss(model,batch,a.device,a.pair_loss_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at {step+1}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        completed=step+1
        if completed%a.eval_every==0 or completed==a.steps:
            history[str(completed)]={'last_train_loss':float(loss.detach()),
                'val':evaluate(model,cache['val'],a.device,a.batch_size),
                'test':evaluate(model,cache['test'],a.device,a.batch_size)}
            report['history']=history
            a.output.write_text(json.dumps(report,indent=2)+'\n')
            save(a.state,model,optimizer,rng,completed,history,identity)
            test=history[str(completed)]['test']
            print(completed,'surname paired rank',
                  round(test['surname_copy']['candidate_rank_paired'],3),
                  'surname paired top1',
                  round(test['surname_copy']['paired_top1'],3),
                  'bracket top1',
                  None if test['sites']['bracket'] is None else
                  round(test['sites']['bracket']['top1'],3),flush=True)


if __name__=='__main__':
    main()
