"""M53 fixed eligible-position native causal response prediction."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'shared_context_native_protocol.json';q=json.loads(pp.read_text());ip=V/'shared_context_inventory.json';mp=V/'coactivation_groups.json';rp=V/'shared_context_token_mean.json'
 for p,k in [(ip,'inventory_sha256'),(mp,'models_sha256'),(rp,'token_mean_result_sha256')]:assert sha(p)==q[k]
 result=json.loads(rp.read_text());sp=Path(result['mean_state_path']);assert sha(sp)==q['token_mean_state_sha256'];tables=torch.load(sp,weights_only=True);inv=json.loads(ip.read_text());u=inv['units'];entries=[x for x in json.loads(mp.read_text())['models'] if x['kind']=='shared_fixed'];cache=A/'emlp_m50_native_writes.pt';assert sha(cache)==result['evaluation_cache_sha256'];cached=torch.load(cache,weights_only=True);X=cached['writes'].cuda();g=torch.tensor([x['group'] for x in u],device='cuda');eligible=torch.tensor(inv['eligible'],device='cuda');donor=torch.tensor(inv['donors'],device='cuda');lookup={int(t):i for i,t in enumerate(tables['tokens'])};ti=torch.tensor([lookup.get(x['token'],-1) for x in u],device='cuda');perm=tables['permutation'].cuda();bank={};checks=[]
 for e in entries:
  path=Path(e['path']);assert sha(path)==e['sha256'];s=torch.load(path,weights_only=True);f=Featurizer('shared_fixed',e['seed']).cuda();f.load_state_dict(s['state']);scale=s['scale'].cuda();a=torch.nn.functional.linear((X-s['mean'].cuda())/scale,f.encoder,f.encoder_bias);scores=a.square()@f.membership.T;gates=torch.zeros_like(scores).scatter_(-1,scores.topk(4,dim=-1).indices,1);cnt=gates@f.membership;sel=gates[torch.arange(len(u),device='cuda'),g,None].bool()&f.membership[g].bool();D=torch.nn.functional.normalize(f.dictionary,dim=0);t=next(z for z in tables['models'] if z['seed']==e['seed'] and z['arm']==e['arm']);parts={}
  for comp,mask in [('exclusive',sel&(cnt==1)),('shared',sel&(cnt>1)),('union',sel)]:
   delta=((a*mask)@D.T*scale).double();norm=delta.norm(dim=-1);mean=t['means'][comp].cuda()[g];tm=torch.empty_like(mean);tab=t['token_tables'][comp]
   for group in range(32):
    ii=torch.where(g==group)[0];atom=tab['atom_indices'][group].cuda();coeff=tab['coefficients'][group].cuda();values=coeff[ti[ii].clamp_min(0)]@D[:,atom].double().T*scale.double();values[ti[ii]<0]=mean[ii][ti[ii]<0];tm[ii]=values
   armdata={'target':delta.float()}
   for name,base in [('donor',delta[donor]),('group_mean',mean),('token_mean',tm),('scrambled',delta[:,perm])]:
    bn=base.norm(dim=-1);p=base*torch.where(bn>0,norm/bn.clamp_min(1e-300),torch.zeros_like(norm))[:,None];assert torch.allclose(p.norm(dim=-1)[bn>0],norm[bn>0],rtol=1e-10,atol=1e-10);assert not bool(p[norm==0].count_nonzero());armdata[name]=p.float()
   for name in armdata:armdata[name][~eligible]=0
   parts[comp]=armdata
  assert torch.allclose(parts['exclusive']['target']+parts['shared']['target'],parts['union']['target'],rtol=1e-4,atol=1e-5)
  bank[e['seed'],e['arm']]=parts
 panel=json.loads((V/'training_source_context_audit.json').read_text());cohort=json.loads((V/'natural_token_context_cohort.json').read_text());train=set(json.loads((V/'diverse_supervision_split.json').read_text())['train_pair_ids']);inputs=[]
 for pair in panel['rows']:
  for ep in [0,1]:inputs.append(dict(key=f"task:{pair['id']}:{ep}",ids=pair['token_ids'][ep],domain='task',split='train' if pair['id'] in train else 'evaluation',family=pair['family']))
 for doc in range(32,64):inputs.append(dict(key=f'natural:{doc}',ids=cohort['token_ids'][doc],domain='natural',split='evaluation',family=None))
 mapping={c['key']:[] for c in inputs}
 for o in inv['occurrences']:mapping[o['key']].append((o['position'],o['unit']))
 for c in inputs:
  pairs=sorted(mapping[c['key']]);assert [p for p,i in pairs]==list(range(len(c['ids'])));mapping[c['key']]=torch.tensor([i for p,i in pairs],device='cuda')
 model,receipt=load_model();assert receipt.weights_sha256==cached['checkpoint_sha256'];rows=[];calls=0;completed=0;progress=A/'emlp_m53_progress.pt';op=V/'shared_context_native.json';assert not op.exists();sourcehash=sha(Path(__file__))
 if progress.exists():
  old=torch.load(progress,weights_only=True);assert old['protocol_sha256']==sha(pp) and old['source_sha256']==sourcehash;rows,calls,completed,checks=[old[k] for k in ['rows','calls','completed','checks']]
 for k,c in enumerate(inputs):
  if k<completed:continue
  ids=torch.tensor([c['ids']],device='cuda');base,cap=trace(model,ids);calls+=1;ix=mapping[c['key']];w=model.transformer.h[1].mlp.Down(cap[1]['products'])[0];assert torch.allclose(w,X[ix],rtol=1e-6,atol=1e-5);lp=base.double().log_softmax(-1);prob=lp.exp()
  for e in entries:
   for comp,arms in bank[e['seed'],e['arm']].items():
    delta=arms['target'][ix][None];targetnoop=not bool(delta.count_nonzero())
    if targetnoop:target=base
    else:target,_=trace(model,ids,replace=lambda j,z,w:w-delta if j==1 else w);calls+=1
    rt=target.double().log_softmax(-1)-lp;energy=float((prob*rt.square()).sum())
    for name in ['donor','group_mean','token_mean','scrambled','zero_response']:
     pred=None if name=='zero_response' else arms[name][ix][None];noop=pred is None or not bool(pred.count_nonzero())
     if noop:out=base
     else:out,_=trace(model,ids,replace=lambda j,z,w:w-pred if j==1 else w);calls+=1
     response=out.double().log_softmax(-1)-lp;err=float((prob*(response-rt).square()).sum());row={a:c[a] for a in ['key','domain','split','family']}|{a:e[a] for a in ['seed','arm','source_arm','grouping']}|dict(component=comp,predictor=name,target_noop=targetnoop,predictor_noop=noop,target_energy=energy,error_energy=err,relative_error=err/energy if energy else None,eligible_positions=int(eligible[ix].sum()),positions=len(c['ids']),target_delta_energy=float(delta.double().square().sum()));rows.append(row)
     if k in [0,160] and comp=='shared':
      def logp(t):
       a=t.cpu().double().numpy();a=a-a.max(-1,keepdims=True);return a-np.log(np.exp(a).sum(-1,keepdims=True))
      n=logp(base);tr=logp(target)-n;pr=logp(out)-n;en=float(np.sum(np.exp(n)*tr**2));er=float(np.sum(np.exp(n)*(pr-tr)**2));assert np.isclose(en,energy,rtol=1e-8,atol=1e-12);assert np.isclose(er,err,rtol=1e-8,atol=1e-12);checks.append(dict(key=c['key'],seed=e['seed'],arm=e['arm'],predictor=name,max_absolute_error=max(abs(en-energy),abs(er-err))))
  tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),source_sha256=sourcehash,rows=rows,calls=calls,completed=k+1,checks=checks),tmp);tmp.replace(progress)
  if (k+1)%8==0:print('INPUTS',k+1,'FORWARDS',calls,flush=True)
 assert len(rows)==192*8*3*5
 summaries=[]
 for e in entries:
  for dom,split in [('task','train'),('task','evaluation'),('natural','evaluation')]:
   for comp in ['exclusive','shared','union']:
    for name in ['donor','group_mean','token_mean','scrambled','zero_response']:
     rr=[r for r in rows if (r['seed'],r['arm'],r['domain'],r['split'],r['component'],r['predictor'])==(e['seed'],e['arm'],dom,split,comp,name)];en=sum(r['target_energy'] for r in rr);er=sum(r['error_energy'] for r in rr);summaries.append({a:e[a] for a in ['seed','arm','source_arm','grouping']}|dict(domain=dom,split=split,component=comp,predictor=name,inputs=len(rr),zero_energy_inputs=sum(r['target_energy']==0 for r in rr),wins_vs_zero=sum(r['error_energy']<r['target_energy'] for r in rr),target_energy=en,error_energy=er,relative_error=er/en if en else None))
 op.write_text(json.dumps(dict(status='completed',protocol_sha256=sha(pp),source_sha256=sourcehash,checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,summaries=summaries,numpy_checks=checks),indent=2)+'\n');print('DONE',calls,len(rows),flush=True)
if __name__=='__main__':main()
