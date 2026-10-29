"""M50 local conditional direction transfer, retaining zero donors and targets."""
from pathlib import Path
import hashlib,json
import torch
import numpy as np
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);ip=V/'shared_context_inventory.json';inv=json.loads(ip.read_text());assert inv['status']=='passed';units=inv['units'];cache=A/'emlp_m50_native_writes.pt'
 if cache.exists():
  data=torch.load(cache,weights_only=True);assert data['inventory_sha256']==sha(ip)
 else:
  model,receipt=load_model();dev=next(model.parameters()).device;panel=json.loads((V/'training_source_context_audit.json').read_text());cohort=json.loads((V/'natural_token_context_cohort.json').read_text());inputs={f"task:{p['id']}:{ep}":p['token_ids'][ep] for p in panel['rows'] for ep in [0,1]};inputs.update({f'natural:{i}':cohort['token_ids'][i] for i in range(32,64)});writes=torch.empty(len(units),1152);calls=0
  for key,ids in inputs.items():
   _,cap=trace(model,torch.tensor([ids],device=dev));calls+=1;w=model.transformer.h[1].mlp.Down(cap[1]['products'])[0].cpu()
   for u in units:
    if u['source_key']==key:writes[u['id']]=w[u['position']]
  data=dict(inventory_sha256=sha(ip),checkpoint_sha256=receipt.weights_sha256,writes=writes,forward_count=calls);torch.save(data,cache);print('CAPTURED',calls,flush=True)
 mp=V/'coactivation_groups.json';models=[x for x in json.loads(mp.read_text())['models'] if x['kind']=='shared_fixed'];X=data['writes'].cuda();gidx=torch.tensor([u['group'] for u in units],device=X.device);donor=torch.tensor(inv['donors'],device=X.device);eligible=torch.tensor(inv['eligible'],device=X.device);rows=[];checks=[]
 for e in models:
  path=Path(e['path']);assert sha(path)==e['sha256'];st=torch.load(path,weights_only=True);f=Featurizer('shared_fixed',e['seed']).cuda();f.load_state_dict(st['state']);scale=st['scale'].cuda();a=torch.nn.functional.linear((X-st['mean'].cuda())/scale,f.encoder,f.encoder_bias);scores=a.square()@f.membership.T;gates=torch.zeros_like(scores).scatter_(-1,scores.topk(4,dim=-1).indices,1);count=gates@f.membership;selected=gates[torch.arange(len(units),device=X.device),gidx,None].bool()&f.membership[gidx].bool();D=torch.nn.functional.normalize(f.dictionary,dim=0)
  for component,mask in [('exclusive',selected&(count==1)),('shared',selected&(count>1)),('union',selected)]:
   delta=((a*mask)@D.T*scale).double();other=delta[donor];norm=delta.norm(dim=-1);dnorm=other.norm(dim=-1);factor=torch.where(dnorm>0,norm/dnorm.clamp_min(1e-300),torch.zeros_like(norm));pred=other*factor[:,None];error=(pred-delta).square().sum(-1);energy=norm.square();both=(norm>0)&(dnorm>0);cos=(delta*other).sum(-1)/(norm*dnorm).clamp_min(1e-300)
   for dom,part in [('task','train'),('task','evaluation'),('natural','evaluation')]:
    take=eligible&torch.tensor([(u['domain'],u['split'])==(dom,part) for u in units],device=X.device);n=int(take.sum());en=float(energy[take].sum());er=float(error[take].sum());rows.append({k:e[k] for k in ['seed','source_arm','grouping','arm']}|dict(component=component,domain=dom,split=part,eligible_units=n,nonzero_targets=int(((norm>0)&take).sum()),nonzero_both=int((both&take).sum()),missing_donor_nonzero_target=int(((norm>0)&(dnorm==0)&take).sum()),target_energy=en,error_energy=er,relative_direction_error=er/en if en else None,mean_nonzero_cosine=float(cos[take&both].mean()) if bool((take&both).any()) else None))
   # Independent NumPy reconstruction/normalization on fixed eligible rows, including zero cases.
   indices=torch.where(eligible)[0][::max(1,int(eligible.sum())//16)][:16];maxerr=0.
   for i in indices.tolist():
    j=int(donor[i]);aa=a[[i,j]].cpu().double().numpy();mm=mask[[i,j]].cpu().numpy();dd=D.cpu().double().numpy();dv=(aa*mm)@dd.T*float(scale);ref=delta[[i,j]].cpu().numpy();rel=np.linalg.norm(dv-ref)/max(np.linalg.norm(ref),1);assert rel<1e-5;norms=np.linalg.norm(ref,axis=1);prediction=ref[1]*(norms[0]/norms[1]) if norms[1]>0 else np.zeros(1152);val=np.sum((prediction-ref[0])**2);assert np.isclose(val,float(error[i]),rtol=1e-8,atol=1e-7);maxerr=max(maxerr,float(rel))
   checks.append({k:e[k] for k in ['seed','arm']}|dict(component=component,samples=len(indices),max_decode_relative_error=maxerr))
  print('MODEL',e['seed'],e['arm'],flush=True)
 result=dict(status='passed',inventory_sha256=sha(ip),models_sha256=sha(mp),source_sha256=sha(Path(__file__)),cache_path=str(cache),cache_sha256=sha(cache),forward_count=data['forward_count'],rows=rows,checks=checks,scope='Local direction transfer only; donor norm matched to recipient (oracle), zero donors preserved, all fixed eligible unique prefixes. Relative error1 is zero-vector prediction. No native response test or semantic reuse claim.')
 op=V/'shared_context_directions.json';assert not op.exists();op.write_text(json.dumps(result,indent=2)+'\n');print('DONE',len(rows),flush=True)
if __name__=='__main__':main()
