"""M52: token-conditioned training means; sparse coverage retained."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent; V=P/'validation'; A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 op=V/'shared_context_token_mean.json';assert not op.exists()
 protocol=dict(training='Natural documents 0–31, all 4096 positions, including inactive gates as zero. No evaluation fitting.',baseline='Per-token/group/component training mean. Unseen token falls back to unconditional group mean. Seen but zero mean stays zero. Oracle recipient norm. No count threshold or smoothing search.',control='Recipient coordinates permuted by fixed CPU torch randperm seed 51001, same permutation for every candidate and unit; preserves norm.',evaluation='M50 eligible unique prefixes, unchanged donor mapping. Local squared vector error, zero predictor=1. Not native response fidelity.',selection='All eight frozen M38 models, all three components; no search.')
 pp=V/'shared_context_token_mean_protocol.json';assert not pp.exists();pp.write_text(json.dumps(protocol,indent=2)+'\n')
 invp=V/'shared_context_inventory.json';inv=json.loads(invp.read_text());u=inv['units'];g=torch.tensor([x['group'] for x in u],device='cuda');eligible=torch.tensor(inv['eligible'],device='cuda');donor=torch.tensor(inv['donors'],device='cuda')
 meta=json.loads((V/'natural_token_context.json').read_text());cp=Path(meta['cache']);assert sha(cp)==meta['cache_sha256'];data=torch.load(cp,weights_only=True);assert bool((data['documents'][:4096]<32).all());assert bool((data['documents'][4096:]>=32).all());train=data['Y'][1][:4096].cuda(); tokens,ti,tc=torch.unique(data['tokens'][:4096],sorted=True,return_inverse=True,return_counts=True); lookup={int(t):i for i,t in enumerate(tokens)}; ui=torch.tensor([lookup.get(z['token'],-1) for z in u],device='cuda'); counts=torch.tensor([int(tc[lookup[z['token']]]) if z['token'] in lookup else 0 for z in u],device='cuda'); ti=ti.cuda(); tc=tc.cuda(); coverage=[]
 for dom,split in [('task','train'),('task','evaluation'),('natural','evaluation')]:
  sel=eligible&torch.tensor([(z['domain'],z['split'])==(dom,split) for z in u],device='cuda');coverage.append(dict(domain=dom,split=split,eligible=int(sel.sum()),unseen=int((sel&(counts==0)).sum()),singleton=int((sel&(counts==1)).sum()),repeated=int((sel&(counts>=2)).sum())))
 cache=A/'emlp_m50_native_writes.pt';cached=torch.load(cache,weights_only=True);assert cached['inventory_sha256']==sha(invp);X=cached['writes'].cuda()
 mp=V/'coactivation_groups.json';models=[e for e in json.loads(mp.read_text())['models'] if e['kind']=='shared_fixed'];rows=[];checks=[];means=[];perm=torch.randperm(1152,generator=torch.Generator().manual_seed(51001)).cuda()
 for e in models:
  p=Path(e['path']);assert sha(p)==e['sha256'];st=torch.load(p,weights_only=True);f=Featurizer('shared_fixed',e['seed']).cuda();f.load_state_dict(st['state']);scale=st['scale'].cuda();D=torch.nn.functional.normalize(f.dictionary,dim=0)
  def encode(w):
   a=torch.nn.functional.linear((w-st['mean'].cuda())/scale,f.encoder,f.encoder_bias);scores=a.square()@f.membership.T;gates=torch.zeros_like(scores).scatter_(-1,scores.topk(4,dim=-1).indices,1);return a,gates,gates@f.membership
  at,gt,ct=encode(train);a,gate,count=encode(X);selected=gate[torch.arange(len(u),device='cuda'),g,None].bool()&f.membership[g].bool();cm={}; tables={};maxerr=0.
  for comp in ['exclusive','shared','union']:
   predmeans=[]; tokenpred=torch.empty(len(u),1152,device="cuda",dtype=torch.float64); group_tables=[]; group_indices=[]
   for group in range(32):
    sel=gt[:,group,None].bool()&f.membership[group].bool();mask=sel & (ct==1) if comp=='exclusive' else sel&(ct>1) if comp=='shared' else sel
    coeff=(at*mask).double().mean(0);mean=coeff@D.double().T*scale.double();predmeans.append(mean)
    # Independent NumPy mask/count and mean on all training positions for each group.
    aa=at.cpu().double().numpy();gg=gt.cpu().numpy();mm=f.membership.cpu().numpy();cc=gg@mm;ss=(gg[:,group,None]>0)&(mm[group]>0);mk=ss&(cc==1) if comp=='exclusive' else ss&(cc>1) if comp=='shared' else ss
    ref=((aa*mk).mean(0)@D.cpu().double().numpy().T)*float(scale);err=np.linalg.norm(ref-mean.cpu().numpy())/max(np.linalg.norm(ref),1);assert err<1e-10;maxerr=max(maxerr,float(err))
    atomidx=torch.where(f.membership[group]>0)[0]; codes=(at*mask)[:,atomidx].double(); sums=torch.zeros(len(tokens),len(atomidx),device='cuda',dtype=torch.float64).index_add_(0,ti,codes); coeffs=sums/tc[:,None]; group_tables.append(coeffs.cpu());group_indices.append(atomidx.cpu()); refs=np.zeros((len(tokens),len(atomidx)));np.add.at(refs,ti.cpu().numpy(),(aa*mk)[:,atomidx.cpu().numpy()]);refs/=tc.cpu().numpy()[:,None];assert np.allclose(refs,coeffs.cpu().numpy(),rtol=1e-12,atol=1e-12); chosen=torch.where(g==group)[0]; idx=ui[chosen]; pred=coeffs[idx.clamp_min(0)]@D[:,atomidx].double().T*scale.double();pred[idx<0]=mean;tokenpred[chosen]=pred
   tables[comp]=dict(coefficients=torch.stack(group_tables),atom_indices=torch.stack(group_indices)); cm[comp]=torch.stack(predmeans);mask=selected&(count==1) if comp=='exclusive' else selected&(count>1) if comp=='shared' else selected;delta=((a*mask)@D.T*scale).double();norm=delta.norm(dim=-1)
   for baseline,base in [('donor',delta[donor]),('group_mean',cm[comp][g]),('token_mean',tokenpred),('scrambled',delta[:,perm])]:
    bn=base.norm(dim=-1);pred=base*torch.where(bn>0,norm/bn.clamp_min(1e-300),torch.zeros_like(norm))[:,None];error=(pred-delta).square().sum(-1)
    for dom,split in [('task','train'),('task','evaluation'),('natural','evaluation')]:
     cohort=eligible&torch.tensor([(z['domain'],z['split'])==(dom,split) for z in u],device='cuda')
     for subset,condition in [('all',torch.ones_like(eligible)),('unseen',counts==0),('singleton',counts==1),('repeated',counts>=2)]:
      take=cohort&condition;energy=float(norm[take].square().sum());er=float(error[take].sum());rows.append({k:e[k] for k in ['seed','arm','source_arm','grouping']}|dict(component=comp,baseline=baseline,domain=dom,split=split,subset=subset,eligible_units=int(take.sum()),target_energy=energy,error_energy=er,relative_error=er/energy if energy else None,missing_prediction_nonzero_target=int((take&(norm>0)&(bn==0)).sum())))
  assert torch.allclose(cm['exclusive']+cm['shared'],cm['union'],rtol=1e-10,atol=1e-10)
  means.append({k:e[k] for k in ['seed','arm','grouping','sha256']}|dict(means={k:v.cpu() for k,v in cm.items()},token_tables=tables));checks.append(dict(seed=e['seed'],arm=e['arm'],numpy_groups_components=96,max_relative_error=maxerr));print('MODEL',e['seed'],e['arm'],flush=True)
 old=json.loads((V/'shared_context_directions.json').read_text())['rows']
 for r in rows:
  if r['baseline']=='donor' and r['subset']=='all':
   o=next(x for x in old if all(x[k]==r[k] for k in ['seed','arm','component','domain','split']));assert np.isclose(o['error_energy'],r['error_energy'],rtol=1e-12);assert np.isclose(o['target_energy'],r['target_energy'],rtol=1e-12)
 sp=A/'emlp_m52_token_group_means.pt';assert not sp.exists();torch.save(dict(models=means,tokens=tokens,training_counts=tc.cpu(),permutation=perm.cpu(),protocol_sha256=sha(pp)),sp)
 op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),protocol_sha256=sha(pp),inventory_sha256=sha(invp),models_sha256=sha(mp),training_cache_sha256=sha(cp),evaluation_cache_sha256=sha(cache),mean_state_path=str(sp),mean_state_sha256=sha(sp),rows=rows,coverage=coverage,checks=checks,token_table_numpy_checks=768,donor_replay_rows=72),indent=2)+'\n');print('DONE',len(rows),flush=True)
if __name__=='__main__':main()
