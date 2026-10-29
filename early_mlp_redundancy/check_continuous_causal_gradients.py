"""M112 parameter gradients through continuous/hard encoding and native causal loss."""
import json
import torch
from encoded_change_loss import EncodedExperiment
from balanced_response_loss import P,V,sha
from continuous_write_featurizer import ContinuousBlockFeaturizer
from fit_pca_encoder import sha


def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');e=EncodedExperiment();cal=json.loads((V/'encoded_change_calibration.json').read_text());checks=[]
 for entry in [x for x in e.protocol['models'] if x['kind']=='block']:
  for mode in ['hard','continuous']:
   f,mu,scale=e.featurizer(entry)
   if mode=='continuous':
    g=ContinuousBlockFeaturizer(entry['seed']).cuda();g.load_state_dict(f.state_dict());f=g
   key=e.protocol['schedules'][str(entry['seed'])]['task_updates'][0]['pair_id']+'|0';c=e.cache[key];d=e.cache[e.donor_key(key)];xr=(c['write'].cuda()-mu)/scale;xd=(d['write'].cuda()-mu)/scale
   def value():
    pr,sr,_=f(xr);pd,sd,_=f(xd);K,R,B,_,_=e.encoded_losses(pr*scale+mu,pd*scale+mu,key)
    signatures=[]
    for x in [xr,xd]:
     a=torch.nn.functional.linear(x,f.encoder,f.encoder_bias).reshape(*x.shape[:-1],32,4);r=a.norm(dim=-1);order=r.topk(9,-1).indices; selected=order[...,:8].sort(-1).values; signatures.append((selected if mode=='hard' else torch.cat([selected,order[...,8:9]],-1)).detach())
    return K+cal['encoded_weight']*B,signatures
   loss,sigs=value();params=list(f.parameters());grads=torch.autograd.grad(loss,params);assert all(torch.isfinite(g).all() for g in grads)
   gn=torch.sqrt(sum(g.double().square().sum() for g in grads));pn=torch.sqrt(sum(x.detach().double().square().sum() for x in params));directions=[(g.double()/gn*pn).to(g.dtype).detach() for g in grads];analytic=sum(float((g.double()*d.double()).sum()) for g,d in zip(grads,directions));originals=[x.detach().clone() for x in params];trials=[]
   with torch.no_grad():
    for eps in [1e-4,3e-5,1e-5]:
     vals=[];changes=[]
     for sign in [1,-1]:
      for param,base,direction in zip(params,originals,directions):param.copy_(base+sign*eps*direction)
      val,ss=value();vals.append(float(val));changes.append(sum(int((a!=b).sum()) for a,b in zip(ss,sigs)))
     numeric=(vals[0]-vals[1])/(2*eps);rel=abs(numeric-analytic)/max(abs(analytic),1e-30);stable=not any(changes)
     trials.append(dict(epsilon=eps,analytic=analytic,numeric=numeric,relative_error=rel,changed_branch_signature_entries=changes,stable=stable))
     if stable:assert rel<.05 or abs(numeric-analytic)<1e-6,(mode,entry['seed'],eps,rel)
    for param,base in zip(params,originals):param.copy_(base)
   assert any(x['stable'] for x in trials),'No verified smooth scale; do not fit.'
   assert all(torch.equal(x,y) for x,y in zip(params,originals));assert all(p.grad is None for p in e.model.parameters())
   checks.append(dict(mode=mode,seed=entry['seed'],initial_sha256=entry['sha256'],key=key,initial_loss=float(loss.detach()),trials=trials));print('CHECKED',mode,entry['seed'],trials,flush=True)
 out=dict(status='passed',source_sha256=sha(__file__),dependencies={name:sha(P/name) for name in ['encoded_change_loss.py','balanced_response_loss.py','continuous_write_featurizer.py','continuous_block_shrinkage.py']},training_protocol_sha256=sha(V/'balanced_response_training_protocol.json'),calibration_sha256=sha(V/'encoded_change_calibration.json'),checks=checks,forward_count=e.calls,scope='Four model starts, all parameters in one gradient-aligned direction each. Three fixed epsilons; selected-support/threshold-identity changes recorded; within-selected ordering ignored. Stable trials require<5% discrepancy. Exact restore, no optimizer updates. Not exhaustive derivative validation.')
 dest=V/'continuous_causal_gradient_checks.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('DONE',e.calls)
if __name__=='__main__':main()
