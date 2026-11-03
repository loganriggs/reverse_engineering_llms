"""Weight-only vs data-aware local bond reductions on crossed skip rules.

All bases are fit without using task labels except for the explicit
``without_skip_bigram`` site-exclusion control. Model training includes every
rule. This compares fidelity at a single pre-W_O attention bond, not a
canonical decomposition of the complete function.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_pre=argparse.ArgumentParser(add_help=False)
_pre.add_argument('--source',default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
_source,_=_pre.parse_known_args()
sys.path.insert(0,_source.source)

from model import BilinearGPT, Config
from original_v3_crossed_skip import LanguageV3CrossedSkip
from train_original_v3_shared_bracket import make_batch
from eval_v3_crossed_skip_model import collect
from eval_original_v3_multirule_bond_bases import activation_grams, bases_from_grams


RULES=('noun_bigram','skip_bigram','skip_trigram','induction')
METHODS=('weight_only','all','without_skip_bigram',
         'random_without_skip_bigram','hybrid_1')


def edited_inputs(gen,edits,length,device):
    original=torch.zeros(len(edits),length-1,dtype=torch.long,device=device)
    changed=original.clone()
    for i,row in enumerate(edits):
        original[i,:len(row['x'])]=torch.tensor(row['x'],device=device)
        changed[i,:len(row['edited_x'])]=torch.tensor(row['edited_x'],device=device)
    q=torch.tensor([row['query'] for row in edits],device=device)
    new=torch.tensor([gen.token2id[row['new_target']] for row in edits],device=device)
    return original,changed,q,new


@torch.no_grad()
def model_outputs(model,x,edited,layer=None,basis=None,rank=None,capture=None):
    hooks=[]
    capture_hooks=[]
    if capture is not None:
        for index,block in enumerate(model.blocks):
            def store(module,inputs,index=index):
                capture[index]=inputs[0].detach()
            capture_hooks.append(block.attn.out.register_forward_pre_hook(store))
    if basis is not None:
        U=basis[:,:rank]
        def project(module,inputs):
            z=inputs[0]
            return ((z@U)@U.T,)
        hooks.append(model.blocks[layer].attn.out.register_forward_pre_hook(project))
    try:
        first=model(x)[0].float()
        for hook in capture_hooks:hook.remove()
        capture_hooks=[]
        return first,model(edited)[0].float()
    finally:
        for hook in hooks+capture_hooks:hook.remove()


def masks_from_labels(labels,device):
    masks={}
    for rule in RULES:
        masks[rule]=torch.tensor([[label==rule for label in row]
                                  for row in labels],device=device)
        assert masks[rule].any()
    return masks


@torch.no_grad()
def panel(model,gen,seed,nstreams,length,edits_per_pair,device):
    x,y,labels,_=make_batch(LanguageV3CrossedSkip(seed=seed),
                             nstreams,length,device)
    _,edits=collect(LanguageV3CrossedSkip(seed=seed+1),
                    nstreams,length,edits_per_pair)
    original,changed,q,new=edited_inputs(gen,edits,length,device)
    pair=torch.cat((original,changed))
    captured={}
    native_logits,native_edit_logits=model_outputs(model,x,pair,capture=captured)
    native_logp=F.log_softmax(native_logits,-1)
    native_edit_probs=native_edit_logits.softmax(-1)
    index=torch.arange(len(edits),device=device)
    native_response=(native_edit_probs[len(edits)+index,q,new]
                     -native_edit_probs[index,q,new])
    return dict(x=x,y=y,masks=masks_from_labels(labels,device),pair=pair,
                q=q,new=new,nedits=len(edits),captured=captured,
                native_logp=native_logp,
                native_response=native_response)


@torch.no_grad()
def score(model,data,layer,basis,rank):
    logits,edit_logits=model_outputs(model,data['x'],data['pair'],
                                     layer,basis,rank)
    logp=F.log_softmax(logits,-1)
    native=data['native_logp']
    kl=(native.exp()*(native-logp)).sum(-1)
    ce=-logp.gather(-1,data['y'][...,None]).squeeze(-1)
    native_ce=-native.gather(-1,data['y'][...,None]).squeeze(-1)
    U=basis[:,:rank]
    z=data['captured'][layer]
    reconstruction=(z-(z@U)@U.T).square().mean(-1)
    n=data['nedits']
    p=edit_logits.softmax(-1)
    index=torch.arange(n,device=logp.device)
    response=p[n+index,data['q'],data['new']]-p[index,data['q'],data['new']]
    result=dict(all_teacher_kl=float(kl.mean()),
                edit_response=float(response.mean()),
                native_edit_response=float(data['native_response'].mean()),
                edit_native_mae=float((response-data['native_response']).abs().mean()),
                edit_oracle_mae=float((response-1).abs().mean()),
                bond_reconstruction_mse=float(reconstruction.mean()))
    for rule,mask in data['masks'].items():
        result[rule]=dict(n=int(mask.sum()),teacher_kl=float(kl[mask].mean()),
                          ce=float(ce[mask].mean()),
                          native_ce=float(native_ce[mask].mean()),
                          bond_reconstruction_mse=float(reconstruction[mask].mean()))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--fit-streams',type=int,default=256)
    p.add_argument('--eval-streams',type=int,default=512)
    p.add_argument('--fit-seed',type=int,default=99501)
    p.add_argument('--eval-seeds',default='99511,99521')
    p.add_argument('--length',type=int,default=129)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--edits-per-pair',type=int,default=16)
    p.add_argument('--ranks',default='8,16,32,64')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    device=torch.device(args.device)
    payload=torch.load(args.checkpoint,map_location=device,weights_only=True)
    assert payload.get('generator')=='LanguageV3CrossedSkip'
    model=BilinearGPT(Config(**payload['config'])).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    gen=LanguageV3CrossedSkip(seed=args.fit_seed)
    grams,fit_counts=activation_grams(model,gen,args.fit_streams,
                                      args.batch_size,args.length,device,args.fit_seed)
    bases=bases_from_grams(model,grams)
    rows=[]
    for eval_seed in map(int,args.eval_seeds.split(',')):
        data=panel(model,gen,eval_seed,args.eval_streams,args.length,
                   args.edits_per_pair,device)
        for layer in range(len(model.blocks)):
            for method in METHODS:
                basis=bases[layer][method]
                for rank in map(int,args.ranks.split(',')):
                    metrics=score(model,data,layer,basis,rank)
                    if rank==model.config.n_embd:
                        assert metrics['all_teacher_kl']<1e-5
                        assert metrics['edit_native_mae']<1e-5
                    rows.append(dict(eval_seed=eval_seed,layer=layer,
                                     method=method,rank=rank,metrics=metrics))
                    print(eval_seed,layer,method,rank,
                          'skip KL',round(metrics['skip_bigram']['teacher_kl'],3),
                          'edit MAE',round(metrics['edit_native_mae'],3),flush=True)
    result=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                fit_counts=fit_counts,rows=rows,
                caveat='One pre-W_O bond is projected at a time. The model '
                       'was trained on all rules. Withholding skip_bigram '
                       'sites is a featurizer-fit site holdout, not a '
                       'model-training task holdout. Reconstruction, teacher '
                       'KL, sampled CE, and edit response are distinct. '
                       'No basis here proves a unique or human-legible '
                       'function-level canonicalization.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
