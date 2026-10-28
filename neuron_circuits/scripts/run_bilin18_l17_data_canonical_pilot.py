"""H466: empirical bilinear feature quotient and matched objective comparison."""
import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_data_canonical_pilot'
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());torch.manual_seed(cfg['seed'])
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 pre=np.empty((512,1152),dtype=np.float32);cached=np.empty_like(pre);seen=[]
 for p in ['compact_operation_confirmation','unprojected_finite_confirmation','covariance_source_confirmation','covariance_fresh_confirmation']:
  p='bilin18_l17_'+p;ids=json.loads((ART/f'{p}_selection.json').read_text())['sites'];z=np.load(ART/f'{p}_rows.npz');assert not set(ids)&set(seen);seen+=ids;pre[ids]=z['pre'];cached[ids]=z['state']
 assert len(set(seen))==512;order=np.random.default_rng(cfg['seed']).permutation(512);tr,va,te=order[:256],order[256:384],order[384:];STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist(),'scope':'historically used corpus, new disjoint fit/evaluation split'},indent=2)+'\n')
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();p=torch.tensor(pre,device='cuda');x=F.rms_norm(p,(1152,));y=((x@L.T)*(x@R.T))@D.T+bias;state=p+y;cache_err=float((state-torch.tensor(cached,device='cuda')).abs().max());cache_relative=float((state-torch.tensor(cached,device='cuda')).norm()/state.norm());xd=p.double()/(p.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();reference=p.double()+((xd@L.double().T)*(xd@R.double().T))@D.double().T+bias.double();native_double_error=float((state.double()-reference).norm()/reference.norm());cached_double_error=float((torch.tensor(cached,device='cuda').double()-reference).norm()/reference.norm());assert max(cache_relative,native_double_error,cached_double_error)<2e-6;print('cache/double relative audits',cache_relative,native_double_error,cached_double_error,flush=True)
 labels=torch.tensor(np.load(ART/'bilin18_compact_operation_confirmation_documents.npy')[:,64].astype(np.int64),device='cuda');tx=x[tr].double();K=(tx@tx.T/1152).square();km=K.mean(0);grand=K.mean();kc=K-km[None]-km[:,None]+grand;ev,U=torch.linalg.eigh(kc);ev=ev.flip(0)[:64];U=U.flip(1)[:,:64];assert float(ev[-1])>1e-10;encoder=U/ev[None]*256**.5
 def features(xx):
  k=(xx.double()@tx.T/1152).square();return ((k-km[None]-k.mean(1,keepdim=True)+grand)@encoder).float()
 Z=features(x).detach();white_err=float((Z[tr].double().T@Z[tr].double()/256-torch.eye(64,device='cuda')).abs().max());assert white_err<1e-5
 mean=y[tr].mean(0);scale=(y[tr]-mean).square().mean().sqrt();target=((y-mean)/scale).detach();C=Z[tr].double().T@target[tr].double()/256;u,sv,vh=torch.linalg.svd(C,full_matrices=False);A0=(u[:,:16]*sv[:16].sqrt()).float();B0=(sv[:16].sqrt()[:,None]*vh[:16]).float()
 def logprob(ss):return (30*torch.tanh(F.rms_norm(ss,(1152,))@W.T/30)).double().log_softmax(-1)
 with torch.no_grad():
  teacher=torch.cat([logprob(state[i:i+32]) for i in range(0,512,32)]);tp=teacher.exp()
 def loss(kind,idx,A,B):
  pred=mean+scale*((Z[idx]@A)@B)
  if kind=='write':return ((pred-y[idx])/scale).square().mean()
  lp=logprob(p[idx]+pred)
  if kind=='kl':return (tp[idx]*(teacher[idx]-lp)).sum(-1).mean()
  return -lp[torch.arange(len(idx),device='cuda'),labels[idx]].mean()
 fitted={'write':(A0.clone(),B0.clone())};history={}
 for kind in ['kl','ce']:
  A=torch.nn.Parameter(A0.clone());B=torch.nn.Parameter(B0.clone());opt=torch.optim.Adam([A,B],lr=cfg['learning_rate']);best=(A.detach().clone(),B.detach().clone());bestval=float(loss(kind,va,A,B));beststep=0;hist=[{'step':0,'validation':bestval}];rng=np.random.default_rng(cfg['seed']+(['kl','ce'].index(kind)+1));
  for step in range(1,cfg['steps']+1):
   idx=rng.choice(tr,cfg['batch_size'],replace=False);opt.zero_grad();v=loss(kind,idx,A,B);assert torch.isfinite(v);v.backward();opt.step()
   if step%cfg['validation_every']==0:
    with torch.no_grad():value=float(loss(kind,va,A,B))
    hist.append({'step':step,'validation':value,'training_batch_loss':float(v.detach())})
    if value<bestval:bestval=value;beststep=step;best=(A.detach().clone(),B.detach().clone())
    print(kind,step,'val',value,'beststep',beststep,flush=True)
  fitted[kind]=best;history[kind]={'selected_step':beststep,'history':hist}
 # SVD fixes a representative factor gauge of each effective map; no behavior change.
 maps={};gauge={};singular={};save={'training_normalized_inputs':tx.cpu().numpy(),'kernel_mean':km.cpu().numpy(),'kernel_grand_mean':float(grand),'kernel_encoder':encoder.cpu().numpy(),'write_mean':mean.cpu().numpy(),'write_scale':float(scale),'feature_eigenvalues':ev.cpu().numpy()}
 for name,(aa,bb) in fitted.items():
  c=aa.double()@bb.double();uu,ss,vv=torch.linalg.svd(c,full_matrices=False);recon=(uu[:,:16]*ss[:16])@vv[:16];gauge[name]=float((recon-c).norm()/c.norm());assert gauge[name]<1e-10;maps[name]=(aa@bb).detach();singular[name]=ss[:16].cpu().numpy().tolist();save[name+'_encoder']=uu[:,:16].cpu().numpy();save[name+'_singular_values']=ss[:16].cpu().numpy();save[name+'_decoder']=vv[:16].cpu().numpy();save[name+'_effective_map']=maps[name].cpu().numpy()
 np.savez_compressed(STEM.with_name(STEM.name+'_canonical_maps.npz'),**save);print('fit and canonical maps frozen before test evaluation',flush=True)
 # Batch all evaluation modes: direct residual stays fixed for interventions.
 sources=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float();source_idx=np.arange(128)%64;rows={};metrics={};rawbaseline={}
 with torch.no_grad():
  for mode,alpha in [('intact',0),('negative',-1),('positive',1)]:
   pp=p[te];xx=F.rms_norm(pp-alpha*sources[source_idx],(1152,));yy=((xx@L.T)*(xx@R.T))@D.T+bias;zz=features(xx);acc={name:[] for name in fitted};native_ce=[]
   for start in range(0,128,16):
    sl=slice(start,start+16);lp=logprob(pp[sl]+yy[sl]);prob=lp.exp();labs=labels[te][sl];ce=-lp[torch.arange(len(labs),device='cuda'),labs];native_ce.append(ce.cpu().numpy())
    for name,c in maps.items():
     pred=mean+scale*(zz[sl]@c);lq=logprob(pp[sl]+pred);ceq=-lq[torch.arange(len(labs),device='cuda'),labs];vals=torch.stack([((pred-yy[sl])/scale).square().mean(-1),(prob*(lp-lq)).sum(-1),ceq-ce,(lp.argmax(-1)==lq.argmax(-1)).float()],-1);acc[name].append(vals.cpu().numpy())
   rows[mode+'_native_ce']=np.concatenate(native_ce)
   for name in fitted:
    v=np.concatenate(acc[name]);rows[mode+'_'+name]=v;metrics[mode+'_'+name]=dict(zip(['normalized_write_mse','KL','CE_delta','top1_agreement'],v.mean(0).tolist()));print(mode,name,metrics[mode+'_'+name],flush=True)
  # Full training and validation objective matrix to expose optimization and generalization.
  objective_matrix={}
  for split,idx in [('train',tr),('validation',va),('test',te)]:
   for name,(aa,bb) in fitted.items():objective_matrix[split+'_'+name]={obj:float(loss(obj,idx,aa,bb)) for obj in ['write','kl','ce']}
 assert all(np.isfinite(v).all() for v in rows.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);j={'experiment':'H466','config':cfg,'metrics':metrics,'objective_matrix':objective_matrix,'training':history,'canonical_singular_values':singular,'numerical_gates':{'whitening':white_err<1e-5,'gauge_reconstruction':max(gauge.values())<1e-10,'cache_replay':max(cache_relative,native_double_error,cached_double_error)<2e-6},'whitening_max_abs':white_err,'gauge_relative_errors':gauge,'cached_state_max_abs':cache_err,'cache_relative':cache_relative,'native_double_relative':native_double_error,'cached_double_relative':cached_double_error,'scope':'exploratory canonical representative in fixed64-dimensional empirical bilinear feature family; histories all reused','unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
