import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_canonical_stability'
def assignment(cost):
 n=len(cost);u=np.zeros(n+1);v=np.zeros(n+1);p=np.zeros(n+1,dtype=int);way=np.zeros(n+1,dtype=int)
 for i in range(1,n+1):
  p[0]=i;j0=0;minv=np.full(n+1,np.inf);used=np.zeros(n+1,dtype=bool)
  while True:
   used[j0]=True;i0=p[j0];delta=np.inf;j1=0
   for j in range(1,n+1):
    if not used[j]:
     cur=cost[i0-1,j-1]-u[i0]-v[j]
     if cur<minv[j]:minv[j]=cur;way[j]=j0
     if minv[j]<delta:delta=minv[j];j1=j
   for j in range(n+1):
    if used[j]:u[p[j]]+=delta;v[j]-=delta
    else:minv[j]-=delta
   j0=j1
   if p[j0]==0:break
  while True:
   j1=way[j0];p[j0]=p[j1];j0=j1
   if j0==0:break
 result=np.empty(n,dtype=int)
 for j in range(1,n+1):result[p[j]-1]=j-1
 assert len(set(result))==n
 return result
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
 torch.set_grad_enabled(False);train=panel(pa[:1024],32);test=panel(np.load(ART/'bilin18_l17_structured_metric_subspace_rows.npz')['pre'],64);rng=np.random.default_rng(cfg['split_seed']);models={'full':V0};saved={};splits={};geometry={}
 # Small brute-force audit of the assignment implementation, independent of real alignments.
 import itertools
 cc=rng.normal(size=(5,5));match=assignment(cc);assert abs(cc[np.arange(5),match].sum()-min(sum(cc[i,p[i]] for i in range(5)) for p in itertools.permutations(range(5))))<1e-10
 for rep in range(cfg['split_repetitions']):
  left=[];right=[]
  for donor in range(32):
   ids=rng.permutation(np.arange(donor,1024,32));left.extend(ids[:16]);right.extend(ids[16:])
  left=np.array(sorted(left));right=np.array(sorted(right));assert not set(left)&set(right) and len(set(left)|set(right))==1024;splits[str(rep)]={'left':left.tolist(),'right':right.tolist()}
  for side,ids in [('left',left),('right',right)]:
   ix=np.r_[ids,ids+1024];_,sv,Vh=torch.linalg.svd(train['X'][ix],full_matrices=False);name=f'{rep}_{side}';models[name]=Vh[:64];saved[name+'_basis']=Vh[:64].cpu().numpy();saved[name+'_singular_values']=sv.cpu().numpy()
  A=models[f'{rep}_left'];B=models[f'{rep}_right'];over=(A@B.T).cpu().numpy();cos=torch.linalg.svdvals(A@B.T).cpu().numpy().clip(0,1);perm=assignment(-abs(over));aligned=abs(over[np.arange(64),perm]);geom={'projector_overlap':float((cos**2).mean()),'principal_angle_degrees_quantiles':np.quantile(np.degrees(np.arccos(cos)),[0,.1,.5,.9,1]).tolist(),'matched_axis_cosine_quantiles':np.quantile(aligned,[0,.1,.5,.9,1]).tolist(),'first8_matched_axis_cosines':aligned[:8].tolist(),'axis_permutation':perm.tolist(),'axis_signs':np.sign(over[np.arange(64),perm]).tolist()};geometry[str(rep)]=geom
  for side in ['left','right']:
   ss=saved[f'{rep}_{side}_singular_values'];geom[side+'_relative_spectral_gaps_first8']=((ss[:8]-ss[1:9])/ss[:8]).tolist();geom[side+'_rank64_relative_gap']=float((ss[63]-ss[64])/ss[63]);geom[side+'_raw_training_energy_covered64']=float((ss[:64]**2).sum()/(ss**2).sum())
  print('split',rep,geom,flush=True)
 errors={n:[] for n in models};pairs={str(i):[] for i in range(4)};effects=[]
 for start in range(0,1024,16):
  sl=slice(start,start+16);truth=test['truth'][sl];pr=truth.exp();base=lp(test['state'][sl]);effects.append(torch.stack([(pr*(truth-base)).sum(-1),test['U'][sl].square().sum(-1),test['X'][sl].square().sum(-1)],-1).cpu().numpy());preds={};lps={}
  for n,V in models.items():
   d=(test['X'][sl]@V.T)@V+test['other'][sl];q=lp(test['state'][sl]+d.float());errors[n].append(torch.stack([(pr*(truth-q)).sum(-1),(d-test['U'][sl]).square().sum(-1)],-1).cpu().numpy());preds[n]=d;lps[n]=q
  for rep in range(4):
   na=f'{rep}_left';nb=f'{rep}_right';qa=lps[na];qb=lps[nb];sym=.5*((qa.exp()*(qa-qb)).sum(-1)+(qb.exp()*(qb-qa)).sum(-1));raw=(preds[na]-preds[nb]).square().sum(-1);pairs[str(rep)].append(torch.stack([sym,raw],-1).cpu().numpy())
 ef=np.concatenate(effects);saved['effects']=ef;metrics={};pair_metrics={}
 for n,rs in errors.items():
  er=np.concatenate(rs);saved[n+'_errors']=er;metrics[n]={}
  for sign,sl in [('negative',slice(0,512)),('positive',slice(512,1024))]:metrics[n][sign]={'full_KL':float(1-er[sl,0].sum()/ef[sl,0].sum()),'full_raw':float(1-er[sl,1].sum()/ef[sl,1].sum())}
 for rep,rs in pairs.items():
  rr=np.concatenate(rs);saved[rep+'_disagreement']=rr;pair_metrics[rep]={}
  for sign,sl in [('negative',slice(0,512)),('positive',slice(512,1024))]:pair_metrics[rep][sign]={'symmetric_KL_over_native_effect':float(rr[sl,0].sum()/ef[sl,0].sum()),'raw_disagreement_over_native_response':float(rr[sl,1].sum()/ef[sl,1].sum()),'raw_disagreement_over_cross_energy':float(rr[sl,1].sum()/ef[sl,2].sum())}
 prior=np.load(ART/'bilin18_l17_structured_metric_subspace_rows.npz');assert np.allclose(saved['full_errors'],prior['confirmation_initial'],rtol=3e-5,atol=1e-8);assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H492','config':cfg,'splits':splits,'geometry':geometry,'metrics':metrics,'pair_metrics':pair_metrics,'audits':{'assignment_bruteforce':True,'full_replay':True,'finite':True},'unique_completed_documents':18880},indent=2)+'\n');print('metrics',metrics,flush=True);print('pair metrics',pair_metrics,flush=True)
if __name__=='__main__':main()
