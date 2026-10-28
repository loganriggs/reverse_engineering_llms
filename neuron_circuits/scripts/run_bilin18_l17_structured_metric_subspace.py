import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_structured_metric_subspace'
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
 train=panel(pa[:1024],32);val=panel(pa[1024:],64)
 def loss(pan,V,backward=False):
  total=0.;N=len(pan['X'])
  for start in range(0,N,16):
   sl=slice(start,start+16);delta=((pan['X'][sl]@V.T)@V+pan['other'][sl]).float();q=lp(pan['state'][sl]+delta);truth=pan['truth'][sl];v=(truth.exp()*(truth-q)).sum()/N
   if backward:v.backward()
   total+=float(v.detach())
  return total
 with torch.no_grad():tl=loss(train,V0);vl=loss(val,V0)
 V=V0.clone();best=V.clone();bestval=vl;beststep=0;hist=[{'step':0,'train_KL':tl,'validation_KL':vl}];audit={'orthogonality_max':0.}
 for step in range(1,cfg['iterations']+1):
  V=V.detach().requires_grad_();loss(train,V,True);g=V.grad.detach()
  with torch.no_grad():
   g=g-(g@V.T)@V;X=train['X'];direction=-g*((X@V.T)@V).norm()/((X@g.T)@V+(X@V.T)@g).norm();candidates=[(tl,0.,V.detach())]
   for frac in cfg['fractions']:
    vv=torch.linalg.qr((V+frac*direction).T).Q.T;err=float((vv@vv.T-torch.eye(64,device='cuda')).abs().max());audit['orthogonality_max']=max(audit['orthogonality_max'],err);assert err<1e-10;candidates.append((loss(train,vv),frac,vv))
   tl,frac,V=min(candidates,key=lambda t:t[0]);V=V.clone();vl=loss(val,V);hist.append({'step':step,'train_KL':tl,'validation_KL':vl,'fraction':frac,'candidates':[[float(v[1]),v[0]] for v in candidates]})
   if vl<bestval:bestval=vl;beststep=step;best=V.clone()
   print('step',step,'train',tl,'validation',vl,'selected',beststep,flush=True)
 torch.set_grad_enabled(False);np.savez_compressed(STEM.with_name(STEM.name+'_compiled.npz'),basis=b.cpu().numpy(),Q=Q.cpu().numpy(),initial_write=V0.cpu().numpy(),selected_write=best.cpu().numpy(),initial_read=(M.transpose(1,2)@V0.T).cpu().numpy(),selected_read=(M.transpose(1,2)@best.T).cpu().numpy());STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'selected_step':beststep,'history':hist},indent=2)+'\n');print('SELECTION SAVED BEFORE CONFIRMATION',flush=True)
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);test=panel(pre.cpu().numpy(),64);rows={};metrics={}
 for label,pan in [('validation',val),('confirmation',test)]:
  eff=[];errs={n:[] for n in ['initial','selected']}
  for start in range(0,len(pan['X']),16):
   sl=slice(start,start+16);truth=pan['truth'][sl];pr=truth.exp();base=lp(pan['state'][sl]);eff.append(torch.stack([(pr*(truth-base)).sum(-1),pan['U'][sl].square().sum(-1)],-1).cpu().numpy())
   for n,vv in [('initial',V0),('selected',best)]:
    delta=(pan['X'][sl]@vv.T)@vv+pan['other'][sl];q=lp(pan['state'][sl]+delta.float());errs[n].append(torch.stack([(pr*(truth-q)).sum(-1),(delta-pan['U'][sl]).square().sum(-1)],-1).cpu().numpy())
  ef=np.concatenate(eff);rows[label+'_effects']=ef;metrics[label]={}
  for n,rr in errs.items():
   rr=np.concatenate(rr);rows[label+'_'+n]=rr;metrics[label][n]={}
   for sign,sl in [('negative',slice(0,pan['n'])),('positive',slice(pan['n'],None))]:metrics[label][n][sign]={'full_KL':float(1-rr[sl,0].sum()/ef[sl,0].sum()),'full_raw':float(1-rr[sl,1].sum()/ef[sl,1].sum())}
  print(label,metrics[label],flush=True)
 assert all(np.isfinite(v).all() for v in rows.values());assert all(hist[i]['train_KL']<=hist[i-1]['train_KL']+1e-12 for i in range(1,len(hist)));np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,pre=pre.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H491','config':cfg,'selected_step':beststep,'history':hist,'metrics':metrics,'audits':audit,'unique_completed_documents':18880},indent=2)+'\n')
if __name__=='__main__':main()
