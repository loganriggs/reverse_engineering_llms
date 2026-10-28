"""H957 teacher-informed downstream metric oracle in frozen top-eight span."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import measure
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_fisher_span_oracle'
def trustsolve(G,b,radius):
 val,V=torch.linalg.eigh(G);val=val.clamp_min(0);ridge=1e-6*val.sum()/len(val);ridge=ridge.clamp_min(1e-30);z=V.T@b
 def at(t):return V@(z/(val+ridge+t))
 lo=torch.zeros_like(ridge);hi=val.max()+ridge
 for _ in range(80):
  if at(hi).norm()<=radius:break
  hi*=2
 for _ in range(80):
  mid=(lo+hi)/2
  if at(mid).norm()>radius:lo=mid
  else:hi=mid
 t=hi if at(lo*0).norm()>radius else lo*0
 coef=at(t);res=((G+(ridge+t)*torch.eye(len(b),dtype=b.dtype))@coef-b).norm()/b.norm().clamp_min(1e-30)
 return coef,float(res),float(coef.norm()/radius),float(ridge),float(t)
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'fits':ART/'bilin18_product_complexity_training_fits.pt','effects':ART/'bilin18_factorial_number_contexts_effects.pt','tasks':ART/'bilin18_factorial_number_contexts_captures.pt','prior':ART/'bilin18_factorial_number_contexts.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 cfg={'experiment':'H957','hashes':hashes,'design':'Frozen .1-penalty top8 span, both seeds, all168 factorial swaps and MLP/joint paths. Teacher-informed per-example Fisher weighted least-squares coefficients, centered at zero MLP change. Reference distribution weights center logit Jacobians and exact target logit difference. Ridge=1e-6 mean Fisher eigenvalue. Exact quadratic norm-constrained solution via eigensystem/bisection; radii1x and2x teacher MLP change norm, all retained. No model training.','finite_differences':'Central differences per orthonormal span vector at .01 and .005 teacher-effect norm. Half step used; full-step derivative discrepancy reported for every case, no filtering.','comparators':'Zero MLP change, exact teacher, actual top8, Euclidean projection, Fisher radius1 and radius2. All candidates evaluated in nonlinear live model; local quadratic fit is not a guarantee of globally best KL.','scope':'An oracle using evaluation teacher outputs, not a deployable or learned representation. Output KL measures distributions at the final token. Norm budgets limit arbitrary off-manifold compensation; do not establish on-manifold validity.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 c=torch.load(paths['effects'],weights_only=False);task=torch.load(paths['tasks'],weights_only=False);fits=torch.load(paths['fits'],weights_only=False);prior=json.loads(paths['prior'].read_text());ex=task['examples'];jobs=c['jobs'];E=c['true_effects'].double();bases={};actualeffects={};ortho=0.
 for seed in [0,1]:
  state=fits['models'][.1,seed];ix=fits['orders'][.1,seed]['agreement'][:8];O=state['out'][ix].double();W=O/O.norm(dim=1,keepdim=True);_,sv,V=torch.linalg.svd(W,full_matrices=False);assert sv[-1]>sv[0]*1e-10;B=V.T;bases[seed]=B;ortho=max(ortho,float((B.T@B-torch.eye(8)).abs().max()))
  u=F.rms_norm(c['inputs'],(1152,));v=F.rms_norm(c['inputs']+c['deltas'],(1152,));L=F.normalize(state['left'],dim=0)[:,ix];R=F.normalize(state['right'],dim=0)[:,ix];actualeffects[seed]=(((v@L)*(v@R)-(u@L)*(u@R)).double()@O).float()
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 count=0
 def run(ids,pos,addition):
  nonlocal count
  x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
  for l,block in enumerate(model.blocks):
   raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att;x=pre+block.mlp(F.rms_norm(pre,(1152,)))
   if l==4:x[torch.arange(len(ids),device='cuda'),pos]+=addition.float().cuda()
  count+=len(ids);return (30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30)).double().cpu()
 buckets=defaultdict(list)
 for n,j in enumerate(jobs):buckets[len(ex[j['index']]['ids'])].append(n)
 hist={(r['id'],r['recipient']):r for r in prior['records']};records=[];diag=[];coefficients=[];identity=0.;replay=0.;kkt=0.;normmax=0.
 for length,ns in sorted(buckets.items()):
  for st in range(0,len(ns),8):
   ii=ns[st:st+8];batch=[jobs[n] for n in ii];ids=torch.tensor([ex[j['index']]['ids'] for j in batch],device='cuda');pos=torch.tensor([ex[j['index']]['position'] for j in batch],device='cuda');et=E[ii];en=et.norm(dim=1);assert bool((en>0).all())
   for seed in [0,1]:
    B=bases[seed];euclidean=et@B@B.T
    for path in ['mlp','joint']:
     offset=c['deltas'][ii] if path=='joint' else torch.zeros_like(c['deltas'][ii]);zero=run(ids,pos,offset);reference=run(ids,pos,offset+et);lp=reference.log_softmax(1);p=lp.exp();target=reference-zero;target-=torch.sum(p*target,dim=1,keepdim=True)
     jac=[]
     for step in [.01,.005]:
      h=step*en;cols=[]
      for k in range(8):
       edit=h[:,None]*B[:,k];zp=run(ids,pos,offset+edit);zm=run(ids,pos,offset-edit);cols.append((zp-zm)/(2*h[:,None]))
      J=torch.stack(cols,dim=2);J-=torch.sum(p[:,:,None]*J,dim=1,keepdim=True);jac.append(J)
     J=jac[1];difference=((jac[0]-J).square()*p[:,:,None]).sum((1,2)).sqrt()/(J.square()*p[:,:,None]).sum((1,2)).sqrt().clamp_min(1e-30)
     G=torch.einsum('nvk,nv,nvl->nkl',J,p,J);b=torch.einsum('nvk,nv,nv->nk',J,p,target);outs={'actual8':run(ids,pos,offset+actualeffects[seed][ii]),'euclidean8':run(ids,pos,offset+euclidean)}
     for radius in [1.,2.]:
      co=[]
      for a,n in enumerate(ii):
       z,err,nr,reg,t=trustsolve(G[a],b[a],radius*en[a]);kkt=max(kkt,err);normmax=max(normmax,nr);co.append(z)
       predicted=.5*float(((target[a]-J[a]@z).square()*p[a]).sum());coefficients.append(dict(index=n,seed=seed,path=path,radius=radius,coefficients=z.tolist(),ridge=reg,lagrange=t,predicted_quadratic=predicted,norm_over_teacher=float(z.norm()/en[a])))
      co=torch.stack(co);outs[f'fisher{int(radius)}']=run(ids,pos,offset+co@B.T)
     for a,n in enumerate(ii):
      job=jobs[n];groups=ex[job['index']]['groups'];margin=lambda x:measure(x,groups)['margin'];native=task['native'][job['index']];base=margin(native);refeff=margin(reference[a])-base;zeroKL=float((p[a]*(lp[a]-zero[a].log_softmax(0))).sum())
      if path=='mlp':identity=max(identity,float((zero[a]-native).abs().max()))
      else:replay=max(replay,abs((1 if job['recipient']==1 else -1)*refeff-hist[job['id'],job['recipient']]['donor_directed_delta']))
      diag.append(dict(index=n,seed=seed,path=path,format=job['format'],finite_step_relative_error=float(difference[a]),zero_KL=zeroKL))
      for mode,out in outs.items():
       pred=margin(out[a])-base;kl=float((p[a]*(lp[a]-out[a].log_softmax(0))).sum());records.append(job|dict(seed=seed,path=path,mode=mode,reference_effect=refeff,predicted_effect=pred,effect_error=pred-refeff,KL=kl,zero_MLP_KL=zeroKL))
   print('DONE',length,st,count,flush=True)
 summary=[];grouped=defaultdict(list)
 for r in records:grouped[r['format'],r['seed'],r['path'],r['mode']].append(r)
 for key,rr in grouped.items():
  rms=lambda k:float(np.mean([r[k]**2 for r in rr])**.5);summary.append(dict(zip(['format','seed','path','mode'],key))|dict(relative_error=rms('effect_error')/rms('reference_effect'),reference_RMS=rms('reference_effect'),KL=float(np.mean([r['KL'] for r in rr])),KL_ratio=sum(r['KL'] for r in rr)/sum(r['zero_MLP_KL'] for r in rr)))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),orthogonality_max_abs=ortho,identity_logits_max_abs=identity,historical_joint_margin_max_abs=replay,kkt_relative_residual=kkt,max_norm_over_budget=normmax,finite_step_max_relative_error=max(r['finite_step_relative_error'] for r in diag),finite_step_median_relative_error=float(np.median([r['finite_step_relative_error'] for r in diag])),whole_model_forwards=count,expected_forwards=len(jobs)*2*2*38,no_model_training=True,teacher_informed_coefficients=True,test_evaluations=0,finite=all(np.isfinite(r['KL']) for r in records))
 assert audits['hashes_unchanged'] and audits['finite'] and count==audits['expected_forwards'] and identity<1e-4 and replay<1e-4 and kkt<1e-8 and normmax<1+1e-10
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,diagnostics=diag,coefficients=coefficients,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
