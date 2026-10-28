"""H466: empirical bilinear feature quotient and matched objective comparison."""
import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_canonical_gradient_diagnosis'
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
 init=np.load(ART/'bilin18_l17_data_canonical_pilot_canonical_maps.npz');replay=float((A0@B0-torch.tensor(init['write_effective_map'],device='cuda')).norm()/(A0@B0).norm());assert replay<1e-6
 gradients={};base={}
 for split,idx in [('train',tr),('validation',va)]:
  base[split]={}
  with torch.no_grad():
   for obj in ['write','kl','ce']:base[split][obj]=float(loss(obj,idx,A0,B0))
  for obj in ['kl','ce']:
   aa=A0.detach().clone().requires_grad_();bb=B0.detach().clone().requires_grad_();ga=torch.zeros_like(aa);gb=torch.zeros_like(bb)
   for start in range(0,len(idx),32):
    sl=idx[start:start+32];g=torch.autograd.grad(loss(obj,sl,aa,bb),[aa,bb]);ga+=g[0]*len(sl)/len(idx);gb+=g[1]*len(sl)/len(idx)
   gradients[split+'_'+obj]=(ga.detach(),gb.detach())
 summaries={};curves={};saved={};cn=(A0@B0).norm()
 for obj in ['kl','ce']:
  ga,gb=gradients['train_'+obj];jac=ga@B0+A0@gb;factor=cn/jac.norm();da=-factor*ga;db=-factor*gb;gv=gradients['validation_'+obj];dot=float((ga*gv[0]).sum()+(gb*gv[1]).sum());normt=float((ga.square().sum()+gb.square().sum()).sqrt());normv=float((gv[0].square().sum()+gv[1].square().sum()).sqrt());summaries[obj]={'train_validation_gradient_cosine':dot/(normt*normv),'first_order_relative_map_step':float((da@B0+A0@db).norm()/cn),'directional_derivatives':{},'finite_difference':{}};curves[obj]=[];saved[obj+'_delta_A']=da.cpu().numpy();saved[obj+'_delta_B']=db.cpu().numpy()
  for split in ['train','validation']:
   summaries[obj]['directional_derivatives'][split]={}
   for target in ['kl','ce']:
    gg=gradients[split+'_'+target];summaries[obj]['directional_derivatives'][split][target]=float((gg[0]*da).sum()+(gg[1]*db).sum())
  with torch.no_grad():
   for alpha in cfg['fractions']:
    aa=A0+alpha*da;bb=B0+alpha*db;row={'fraction':alpha,'actual_relative_map_change':float((aa@bb-A0@B0).norm()/cn)}
    for split,idx in [('train',tr),('validation',va)]:row[split]={target:float(loss(target,idx,aa,bb)) for target in ['write','kl','ce']}
    curves[obj].append(row);print(obj,alpha,row,flush=True)
   eps=.001
   for split,idx in [('train',tr),('validation',va)]:
    plus=float(loss(obj,idx,A0+eps*da,B0+eps*db));minus=float(loss(obj,idx,A0-eps*da,B0-eps*db));summaries[obj]['finite_difference'][split]={'epsilon':eps,'symmetric_derivative':(plus-minus)/(2*eps),'analytic_derivative':summaries[obj]['directional_derivatives'][split][obj]}
 saved.update(A0=A0.cpu().numpy(),B0=B0.cpu().numpy());np.savez_compressed(STEM.with_name(STEM.name+'_directions.npz'),**saved);j={'experiment':'H467','config':cfg,'baseline':base,'gradients':summaries,'curves':curves,'initial_map_replay_relative':replay,'numerical_gates':{'initial_map_replay':replay<1e-6,'first_order_step_normalized':all(abs(v['first_order_relative_map_step']-1)<1e-5 for v in summaries.values())},'scope':'train and validation only; no test metrics inspected','unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],summaries,flush=True)
if __name__=='__main__':main()
