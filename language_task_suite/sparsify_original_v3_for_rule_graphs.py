"""Iterative L1/prune/final-no-L1 control on the original joint v3 toy language."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


RULES=('noun_bigram','place_bigram','tok_trigram','cat_trigram',
       'skip_bigram','induction','skip_trigram')


def model_and_generator(source,device):
    sys.path.insert(0,str(source/'attn_circuits'))
    from model import BilinearGPT,Config
    from generator_v3 import LanguageV3
    tag='v3_all_2L_4H_d64_5000steps'
    saved=json.loads((source/'configs'/f'bilinear_gpt_{tag}.json').read_text())
    cfg=Config(**{k:v for k,v in saved.items() if k in Config.__dataclass_fields__})
    model=BilinearGPT(cfg).to(device)
    model.load_state_dict(torch.load(source/'models'/f'bilinear_gpt_{tag}.pt',
                                     map_location=device,weights_only=True))
    return model,LanguageV3,cfg


def make_data(generator,n,length,device):
    from generator_v3 import make_batch
    return make_batch(generator,n,length,device=device)


@torch.no_grad()
def evaluate(model,batch):
    x,y,labels,_=batch
    logits=model(x)[0]
    loss=F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='none')
    ce=loss.reshape_as(y).cpu()
    groups=defaultdict(list)
    for b,row in enumerate(labels):
        for t,label in enumerate(row):
            groups[label].append(float(ce[b,t]))
    return dict(overall_ce=float(ce.mean()),
                rule_ce={rule:sum(values)/len(values)
                         for rule,values in groups.items() if values},
                rule_n={rule:len(values) for rule,values in groups.items()})


def prunable(model):
    return {name:p for name,p in model.named_parameters()
            if '.attn.' in name and p.ndim==2}


def mask_stats(mask):
    total=sum(x.numel() for x in mask.values())
    live=sum(int(x.sum()) for x in mask.values())
    return dict(total=total,live=live,sparsity=1-live/total)


def enforce(params,mask):
    with torch.no_grad():
        for name,p in params.items():
            p.mul_(mask[name])


def global_prune(params,mask,target):
    total=sum(x.numel() for x in mask.values())
    keep=round(total*(1-target))
    current=sum(int(x.sum()) for x in mask.values())
    n_drop=current-keep
    if n_drop<=0:
        return
    names=sorted(params)
    magnitudes=torch.cat([params[name].detach().abs()[mask[name]]
                          for name in names])
    threshold_indices=magnitudes.argsort()[:n_drop]
    selection=torch.zeros(current,dtype=torch.bool,device=magnitudes.device)
    selection[threshold_indices]=True
    offset=0
    for name in names:
        live=mask[name].nonzero(as_tuple=True)
        count=len(live[0])
        drop=selection[offset:offset+count]
        mask[name][tuple(index[drop] for index in live)]=False
        offset+=count
    enforce(params,mask)
    assert mask_stats(mask)['live']==keep


def train_phase(model,params,mask,gen,device,steps,batch,length,lr,l1):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    model.train()
    for step in range(steps):
        x,y,_,_=make_data(gen,batch,length,device)
        opt.zero_grad(set_to_none=True)
        logits=model(x)[0]
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        if l1:
            loss=loss+l1*sum(p[mask[name]].abs().sum()
                             for name,p in params.items())
        loss.backward()
        for name,p in params.items():
            p.grad.mul_(mask[name])
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        enforce(params,mask)
    model.eval()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--target-sparsities',default='.4,.6,.75,.85')
    p.add_argument('--steps-per-round',type=int,default=80)
    p.add_argument('--final-steps',type=int,default=160)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--dev-streams',type=int,default=256)
    p.add_argument('--test-streams',type=int,default=512)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--l1',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=94001)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    device=torch.device(a.device)
    source=Path(a.source)
    out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    model,Generator,cfg=model_and_generator(source,device)
    model.eval()
    params=prunable(model)
    mask={name:torch.ones_like(p,dtype=torch.bool) for name,p in params.items()}
    dev=make_data(Generator(seed=a.seed+1),a.dev_streams,a.length,device)
    test=make_data(Generator(seed=a.seed+2),a.test_streams,a.length,device)
    rows=[]
    def record(label):
        row=dict(stage=label,mask=mask_stats(mask),
                 dev=evaluate(model,dev),test=evaluate(model,test))
        rows.append(row)
        torch.save(dict(model=model.state_dict(),mask=mask,config=vars(a)),
                   out/f'{label}.pt')
        print(label,'sparsity',round(row['mask']['sparsity'],3),
              'dev CE',round(row['dev']['overall_ce'],3),
              'noun',round(row['dev']['rule_ce'].get('noun_bigram',float('nan')),3),
              'bracket',round(row['dev']['rule_ce'].get('skip_trigram',float('nan')),3),
              flush=True)
    record('dense')
    train_gen=Generator(seed=a.seed)
    for target in [float(x) for x in a.target_sparsities.split(',')]:
        assert 0<target<1 and target>mask_stats(mask)['sparsity']
        train_phase(model,params,mask,train_gen,device,a.steps_per_round,
                    a.batch_size,a.length,a.lr,a.l1)
        global_prune(params,mask,target)
        record(f'pruned_{round(target*100)}')
    train_phase(model,params,mask,train_gen,device,a.final_steps,a.batch_size,
                a.length,a.lr,0.)
    record('final_no_l1')
    (out/'results.json').write_text(json.dumps(dict(config=vars(a),rows=rows,
        caveat='All checkpoints start from the same dense v3 model. This is '
               'attention-matrix sparsity; embedding and unembedding remain '
               'dense. Rule CE uses sampled targets, not an exact oracle KL.'),
        indent=2)+'\n')


if __name__=='__main__':main()
