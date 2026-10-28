import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_canonical_response_kl'
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 pp=np.empty((512,1152),np.float32)
 for p in ['compact_operation_confirmation','unprojected_finite_confirmation','covariance_source_confirmation','covariance_fresh_confirmation']:
  p='bilin18_l17_'+p;ids=json.loads((ART/f'{p}_selection.json').read_text())['sites'];pp[ids]=np.load(ART/f'{p}_rows.npz')['pre']
 split=json.loads((ART/'bilin18_l17_data_canonical_pilot_selection.json').read_text());tr=split['train'];te=split['test'];p=torch.tensor(pp,device='cuda');h=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float();trainh=h[np.arange(256)%32];testh=h[np.arange(128)%64];x0=F.rms_norm(p,(1152,));xa=torch.cat([x0[tr],F.rms_norm(p[tr]+trainh,(1152,)),F.rms_norm(p[tr]-trainh,(1152,))]);model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach()
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 frozen=np.load(ART/'bilin18_l17_canonical_measure_factorial_models.npz');prior=np.load(ART/'bilin18_l17_canonical_contrast_fit_models.npz');support=torch.tensor(frozen['augmented_support'],device='cuda');km=torch.tensor(frozen['augmented_kernel_mean'],device='cuda');grand=float(frozen['augmented_kernel_grand']);enc=torch.tensor(frozen['augmented_kernel_encoder'],device='cuda');scale=torch.tensor(float(prior['scale']),device='cuda')
 def feat(xx):
  k=(xx.double()@support.T/1152).square();return ((k-km[None]-k.mean(1,keepdim=True)+grand)@enc).float()
 def lp(state):return (30*torch.tanh(F.rms_norm(state,(1152,))@W.T/30)).double().log_softmax(-1)
 with torch.no_grad():
  ztrain=feat(x0[tr]);zm=ztrain.mean(0);ym=native(x0[tr]).mean(0)/scale;panels={}
  for splitname,ids,donorcount in [('train',tr,32),('validation',split['validation'],64)]:
   pp0=p[ids];xx=x0[ids];y0=native(xx);state=pp0+y0;hh=h[np.arange(len(ids))%donorcount];zz=feat(xx);changes=[];truth=[]
   for alpha in [-1,1]:
    x=F.rms_norm(pp0-alpha*hh,(1152,));changes.append(feat(x)-zz);yy=native(x);truth.append(torch.cat([lp(pp0[k:k+16]+yy[k:k+16]) for k in range(0,len(ids),16)]))
   panels[splitname]={'dz':torch.cat(changes).detach(),'state':state.repeat(2,1).detach(),'truth':torch.cat(truth).detach()}
 def balanced(C):
  u,sv,vh=torch.linalg.svd(C.double(),full_matrices=False);return (u[:,:16]*sv[:16].sqrt()).float(),(sv[:16].sqrt()[:,None]*vh[:16]).float()
 A,B=balanced(torch.tensor(prior['joint16_map'],device='cuda'));initial=(A@B).detach().clone()
 def loss(panel,aa,bb,backward=False):
  total=0.;N=len(panel['dz'])
  for start in range(0,N,16):
   sl=slice(start,start+16);pred=scale*((panel['dz'][sl]@aa)@bb);q=lp(panel['state'][sl]+pred);truth=panel['truth'][sl];v=(truth.exp()*(truth-q)).sum(-1).mean()*len(truth)/N
   if backward:v.backward()
   total+=float(v.detach())
  return total
 with torch.no_grad():trainloss=loss(panels['train'],A,B);val=loss(panels['validation'],A,B)
 best=(A.clone(),B.clone());bestval=val;beststep=0;history=[{'step':0,'training_KL':trainloss,'validation_KL':val}]
 for step in range(1,cfg['iterations']+1):
  A=A.detach().requires_grad_();B=B.detach().requires_grad_();loss(panels['train'],A,B,True);ga=A.grad.detach();gb=B.grad.detach()
  with torch.no_grad():
   dc=ga@B+A@gb;current=panels['train']['dz']@(A@B);deriv=panels['train']['dz']@dc;factor=current.norm()/deriv.norm();da=-factor*ga;db=-factor*gb;candidates=[(trainloss,0.,A.detach(),B.detach())]
   for frac in cfg['step_fractions']:
    aa=A+frac*da;bb=B+frac*db;candidates.append((loss(panels['train'],aa,bb),frac,aa,bb))
   chosen=min(candidates,key=lambda t:t[0]);A,B=balanced(chosen[2]@chosen[3]);trainloss=loss(panels['train'],A,B);val=loss(panels['validation'],A,B);history.append({'step':step,'selected_fraction':chosen[1],'training_KL':trainloss,'validation_KL':val,'candidate_training_losses':[[float(x[1]),x[0]] for x in candidates]})
   if val<bestval:bestval=val;beststep=step;best=(A.clone(),B.clone())
   print('step',step,'fraction',chosen[1],'train',trainloss,'validation',val,'selected',beststep,flush=True)
 torch.set_grad_enabled(False);C=best[0]@best[1];models={};save={'support':support.cpu().numpy(),'kernel_mean':km.cpu().numpy(),'kernel_grand':grand,'kernel_encoder':enc.cpu().numpy(),'scale':float(scale)}
 for n in ['joint16','response16','kl16']:
  cc=C if n=='kl16' else torch.tensor(prior[n+'_map'],device='cuda').float();b=ym-zm@cc if n=='kl16' else torch.tensor(prior[n+'_intercept'],device='cuda').float();models[n]=(support,km,grand,enc,cc,b);save[n+'_map']=cc.cpu().numpy();save[n+'_intercept']=b.cpu().numpy()
 np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**save);print('selected map frozen before test; step',beststep,flush=True)
 def predict(x,m):
  support,km,grand,enc,C,b=m;k=(x.double()@support.T/1152).square();z=((k-km[None]-k.mean(1,keepdim=True)+grand)@enc).float();return scale*(z@C+b)
 def lp(state):return (30*torch.tanh(F.rms_norm(state,(1152,))@W.T/30)).double().log_softmax(-1)
 labels=torch.tensor(np.load(ART/'bilin18_compact_operation_confirmation_documents.npy')[te,64].astype(np.int64),device='cuda');pt=p[te];y0=native(x0[te]);pred0={n:predict(x0[te],m) for n,m in models.items()};rows={};metrics={}
 for mode,alpha in [('intact',0),('negative',-1),('positive',1)]:
  xx=F.rms_norm(pt-alpha*testh,(1152,));yy=native(xx);pred={n:predict(xx,m) for n,m in models.items()};acc={n:[] for n in models};de={n:[] for n in models};effects=[]
  for start in range(0,128,16):
   sl=slice(start,start+16);truth=lp(pt[sl]+yy[sl]);prob=truth.exp();base=lp(pt[sl]+y0[sl]);labs=labels[sl];ar=torch.arange(len(labs),device='cuda');ce=-truth[ar,labs];effects.append(torch.stack([(prob*(truth-base)).sum(-1),(yy[sl]-y0[sl]).double().square().sum(-1)],-1).cpu().numpy())
   for n in models:
    q=lp(pt[sl]+pred[n][sl]);acc[n].append(torch.stack([((pred[n][sl]-yy[sl])/scale).square().mean(-1),(prob*(truth-q)).sum(-1),-q[ar,labs]-ce,(q.argmax(-1)==truth.argmax(-1)).float()],-1).cpu().numpy());delta=pred[n][sl]-pred0[n][sl];qd=lp(pt[sl]+y0[sl]+delta);de[n].append(torch.stack([(prob*(truth-qd)).sum(-1),(delta-(yy[sl]-y0[sl])).double().square().sum(-1)],-1).cpu().numpy())
  rows[mode+'_effects']=np.concatenate(effects)
  for n in models:
   rows[mode+'_'+n]=np.concatenate(acc[n]);rows[mode+'_'+n+'_delta']=np.concatenate(de[n]);metrics[mode+'_'+n]=dict(zip(['write_MSE','KL','CE_delta','top1_agreement'],rows[mode+'_'+n].mean(0).tolist()));print(mode,n,metrics[mode+'_'+n],flush=True)
 old=np.load(ART/'bilin18_l17_canonical_contrast_fit_rows.npz');replay=max(float(abs(rows[mode+'_joint16']-old[mode+'_joint16']).max()) for mode in ['intact','negative','positive']);assert replay<1e-4;assert all(np.isfinite(v).all() for v in rows.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);j={'experiment':'H471','config':cfg,'metrics':metrics,'training_history':history,'selected_step':beststep,'baseline_replay_max_abs':replay,'numerical_gates':{'baseline_replay':replay<1e-4,'finite':True},'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
