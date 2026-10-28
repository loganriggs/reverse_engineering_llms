import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_canonical_function_alignment'
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd=L.double(),R.double(),D.double();z=np.load(ART/'bilin18_l17_structured_crossrank_compiled.npz');b=torch.tensor(z['r64_basis'],device='cuda');Q=torch.tensor(z['r64_Q'],device='cuda');V0=torch.tensor(z['r64_write'],device='cuda');bl=Ld@b;br=Rd@b;M=torch.stack([-Dd@(br[:,i,None]*Ld+bl[:,i,None]*Rd) for i in range(4)]);h=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').double();eps=torch.finfo(torch.float32).eps;pa=np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][:1280]
 def native(p):
  x=F.rms_norm(p,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s.float(),(1152,))@W.T/30)).double().log_softmax(-1)
 @torch.no_grad()
 def panel(pa,donors):
  p=torch.tensor(pa,device='cuda');pd=p.double();y0=native(p);state=p+y0;w0=y0.double()-bias.double();hh=h[np.arange(len(p))%donors];d0=pd.square().mean(-1)+eps;xs=[];os=[];targets=[];us=[]
  for alpha in [-1,1]:
   aa=alpha*(hh@b);hp=aa@b.T;den=(pd-hp).square().mean(-1)+eps;cross=(torch.einsum('nd,kod->nko',pd,M)*aa[:,:,None]).sum(1)/den[:,None];other=torch.einsum('ni,nj,ijo->no',aa,aa,Q)/den[:,None]+(d0/den-1)[:,None]*w0;yy=native(p-alpha*hh.float());xs.append(cross);os.append(other);targets.append(p+yy);us.append((yy-y0).double())
  pan={'X':torch.cat(xs),'other':torch.cat(os),'state':state.repeat(2,1),'target':torch.cat(targets),'U':torch.cat(us),'n':len(p)};pan['truth']=torch.cat([lp(pan['target'][i:i+16]) for i in range(0,2*len(p),16)]);return pan
 torch.set_grad_enabled(False);test=panel(np.load(ART/'bilin18_l17_structured_metric_subspace_rows.npz')['pre'],64);prior=np.load(ART/'bilin18_l17_canonical_shared_core_rows.npz');bases={}
 for rep in range(4):
  for side in ['left','right']:
   n=f'{rep}_{side}';bases[n]={mode:torch.tensor(prior[n+'_'+mode+'_basis'],device='cuda') for mode in ['core','tail']}
 saved={};summary={};audit={'finite_omission_KL_replay_maxabs':0.};Wd=W.double();acc={str(i):[] for i in range(4)}
 def inner(x,y,weight=None):return (x*y).sum(-1) if weight is None else (weight*x*y).sum(-1)
 for start in range(0,1024,16):
  sl=slice(start,start+16);X=test['X'][sl];other=test['other'][sl];ss=test['state'][sl];lp0=lp(ss);prob0=lp0.exp();th=torch.tanh(F.rms_norm(ss,(1152,)).double()@Wd.T/30)
  def center(t):return t-(prob0*t).sum(-1,keepdim=True)
  def tangent(v):return center((native_drms(ss.double(),v)@Wd.T)*(1-th.square()))
  nativeT=tangent(test['U'][sl]);truth=test['truth'][sl];effect=(truth.exp()*(truth-lp0)).sum(-1);nativeF=inner(nativeT,nativeT,prob0)
  for rep in range(4):
   tails=[];ts=[];fs=[];kls=[]
   for side in ['left','right']:
    n=f'{rep}_{side}';bc=bases[n]['core'];bt=bases[n]['tail'];core=(X@bc.T)@bc;tail=(X@bt.T)@bt;tails.append(tail);ts.append(tangent(tail));lc=lp(ss+(core+other).float());lf=lp(ss+(core+tail+other).float());fs.append(center(lf-lc));kl=(lf.exp()*(lf-lc)).sum(-1);kls.append(kl);ref=prior[n+'_details'][sl,3];err=float(abs(kl-torch.tensor(ref,device='cuda')).max());audit['finite_omission_KL_replay_maxabs']=max(audit['finite_omission_KL_replay_maxabs'],err);assert np.allclose(kl.cpu().numpy(),ref,rtol=1e-4,atol=1e-8)
   vals=[]
   for pair,weight in [(tails,None),(ts,prob0),(fs,prob0)]:
    aa,bb=pair;vals.extend([inner(aa,aa,weight),inner(bb,bb,weight),inner(aa,bb,weight)])
   vals.extend(kls+[effect,nativeF]);acc[str(rep)].append(torch.stack(vals,-1).cpu().numpy())
 for rep,rs in acc.items():
  rr=np.concatenate(rs);saved[rep+'_inner_products']=rr;summary[rep]={}
  for sign,sl in [('negative',slice(0,512)),('positive',slice(512,1024))]:
   sums=rr[sl].sum(0);s={}
   for label,i in [('raw',0),('Fisher',3),('finite_logratio',6)]:
    aa,bb,ab=sums[i:i+3];cos=ab/(aa*bb)**.5;dis=aa+bb-2*ab;assert abs(cos)<=1+1e-9 and dis>=-1e-9;s[label]={'pooled_cosine':float(cos),'disagreement_over_mean_energy':float(dis/(.5*(aa+bb))),'left_energy':float(aa),'right_energy':float(bb)}
   s['Fisher']['disagreement_over_native_Fisher']=float((sums[3]+sums[4]-2*sums[5])/sums[12]);s['finite_left_KL_over_native_effect']=float(sums[9]/sums[11]);s['finite_right_KL_over_native_effect']=float(sums[10]/sums[11]);summary[rep][sign]=s
  print(rep,summary[rep],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H494','config':cfg,'summary':summary,'audits':audit,'columns':['raw_left_energy','raw_right_energy','raw_inner','Fisher_left_energy','Fisher_right_energy','Fisher_inner','finite_logratio_left_energy','finite_logratio_right_energy','finite_logratio_inner','left_full_to_core_KL','right_full_to_core_KL','native_effect_KL','native_Fisher_energy'],'unique_completed_documents':18880},indent=2)+'\n')
if __name__=='__main__':main()
