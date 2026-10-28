import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_canonical_finite_refinement'
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Wd=W.double();docs=np.load(ROOT/cfg['corpus']);pa=np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][:1280];h0=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float();z=np.load(ART/'bilin18_l17_canonical_width_budget_rank128_models.npz');z={k:torch.tensor(z[k],device='cuda').double() for k in z.files};learned=np.load(ART/'bilin18_l17_canonical_remainder_learning_models.npz');sel=json.loads((ART/'bilin18_l17_canonical_remainder_learning_selection.json').read_text())['selected']['pca'];B=torch.tensor(learned['basis'][:20],device='cuda');C0=torch.tensor(learned[sel+'_coefficients'][:,:20],device='cuda');scale=z['scale'];z['kernel_encoder']=torch.tensor(learned['expanded_kernel_encoder'],device='cuda');audit={}
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 def feat(x):
  k=(x.double()@z['support'].T/1152).square();return (k-z['kernel_mean'][None]-k.mean(1,keepdim=True)+z['kernel_grand'])@z['kernel_encoder']
 def lp(s):return (30*torch.tanh(F.rms_norm(s.float(),(1152,))@W.T/30)).double().log_softmax(-1)
 def panel(pa,donors,metric=False):
  p=torch.tensor(pa,device='cuda');h=h0[np.arange(len(p))%donors];xx=F.rms_norm(p,(1152,));y0=native(xx);state=p+y0;z0=feat(xx);Xs=[];Us=[];Ys=[]
  for alpha in [-1,1]:
   xx=F.rms_norm(p-alpha*h,(1152,));yy=native(xx);Xs.append(feat(xx)-z0);Us.append((yy-y0).double()/scale);Ys.append(p+yy)
  out={'X':torch.cat(Xs),'U':torch.cat(Us),'target':torch.cat(Ys),'state':state.repeat(2,1),'n':len(p)}
  if metric:
   gs=[];bs=[]
   for start in range(0,2*len(p),16):
    ss=out['state'][start:start+16];u=out['U'][start:start+16];th=torch.tanh(F.rms_norm(ss,(1152,)).double()@Wd.T/30);prob=(30*th).softmax(-1)
    def tangent(v):
     t=(native_drms(ss.double(),v)@Wd.T)*(1-th.square());return t-(prob*t).sum(-1,keepdim=True)
    tb=torch.stack([tangent(b.expand_as(u)) for b in B],1);tu=tangent(u);gs.append(torch.einsum('biv,bv,bjv->bij',tb,prob,tb));bs.append(torch.einsum('biv,bv,bv->bi',tb,prob,tu))
   out['G']=torch.cat(gs);out['rhs']=torch.cat(bs)
  return out
 with torch.no_grad():
  train=panel(pa[:1024],32);val=panel(pa[1024:],64)
  for pan in [train,val]:pan['truth']=torch.cat([lp(pan['target'][k:k+16]) for k in range(0,len(pan['X']),16)])
 def loss(pan,C,backward=False):
  total=0.;N=len(pan['X'])
  for start in range(0,N,16):
   sl=slice(start,start+16);delta=(scale*((pan['X'][sl]@C)@B)).float();q=lp(pan['state'][sl]+delta);truth=pan['truth'][sl];v=(truth.exp()*(truth-q)).sum()/N
   if backward:v.backward()
   total+=float(v.detach())
  return total
 with torch.no_grad():tl=loss(train,C0);vl=loss(val,C0)
 C=C0.clone();best=C.clone();bestval=vl;beststep=0;history=[{'step':0,'train_KL':tl,'validation_KL':vl}]
 for step in range(1,cfg['iterations']+1):
  C=C.detach().requires_grad_();loss(train,C,True);g=C.grad.detach()
  with torch.no_grad():
   size=(train['X']@C).norm();den=(train['X']@g).norm();assert float(den)>0;direction=-g*size/den;candidates=[(tl,0.,C.detach())]
   for fraction in cfg['fractions']:
    cc=C+fraction*direction;candidates.append((loss(train,cc),fraction,cc))
   tl,fraction,C=min(candidates,key=lambda t:t[0]);C=C.clone();vl=loss(val,C);history.append({'step':step,'train_KL':tl,'validation_KL':vl,'fraction':fraction,'candidate_train_losses':[[float(x[1]),x[0]] for x in candidates]})
   if vl<bestval:bestval=vl;beststep=step;best=C.clone()
   print('step',step,'train',tl,'validation',vl,'selected',beststep,flush=True)
 torch.set_grad_enabled(False);np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),initial=C0.cpu().numpy(),selected=best.cpu().numpy(),last=C.cpu().numpy(),basis=B.cpu().numpy(),encoder=z['kernel_encoder'].cpu().numpy());STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'selected_step':beststep,'history':history},indent=2)+'\n');print('SELECTION SAVED BEFORE CONFIRMATION',flush=True)
 pt,_,_=collect_sources(model,docs,cfg);test=panel(pt.cpu().numpy(),64);models={'initial':C0,'selected':best};rows={};metrics={}
 for label,pan in [('train',train),('validation',val),('confirmation',test)]:
  errors={n:[] for n in models};effects=[]
  for start in range(0,len(pan['X']),16):
   sl=slice(start,start+16);truth=lp(pan['target'][sl]);pr=truth.exp();base=lp(pan['state'][sl]);u=pan['U'][sl]*scale;effects.append(torch.stack([(pr*(truth-base)).sum(-1),u.square().sum(-1)],-1).cpu().numpy())
   for n,cc in models.items():
    delta=(scale*((pan['X'][sl]@cc)@B)).float();q=lp(pan['state'][sl]+delta);errors[n].append(torch.stack([(pr*(truth-q)).sum(-1),(delta.double()-u).square().sum(-1)],-1).cpu().numpy())
  ef=np.concatenate(effects);rows[label+'_effects']=ef;metrics[label]={}
  for n,ers in errors.items():
   er=np.concatenate(ers);rows[label+'_'+n]=er;metrics[label][n]={}
   for sign,sl in [('negative',slice(0,pan['n'])),('positive',slice(pan['n'],None))]:metrics[label][n][sign]={'KL_recovery':float(1-er[sl,0].sum()/ef[sl,0].sum()),'raw_recovery':float(1-er[sl,1].sum()/ef[sl,1].sum())}
  print(label,metrics[label],flush=True)
 assert all(np.isfinite(v).all() for v in rows.values());assert all(history[i]['train_KL']<=history[i-1]['train_KL']+1e-12 for i in range(1,len(history)));np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,pre=pt.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H484','config':cfg,'selected_step':beststep,'history':history,'metrics':metrics,'audits':{'finite':True,'training_monotone':True},'unique_completed_documents':16832},indent=2)+'\n')
if __name__=='__main__':main()
