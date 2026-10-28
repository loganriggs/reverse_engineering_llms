"""H342 exact mean-linear, context-linear and edit-quadratic input terms."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import B,forward_normalized
from run_bilin18_l17_residual_term_accounting import mean_linear
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_head6_native_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias
 features=[]
 for off in range(0,512,32):
  lp=(30*torch.tanh(model.lm_head(F.rms_norm(intact[off:off+32],(1152,))).double()/30)).log_softmax(-1);prob=lp.exp();features.append(torch.stack([-(prob*lp).sum(1),torch.logit(prob.max(1).values.clamp(1e-10,1-1e-10))],1))
 features=torch.cat(features).reshape(128,4,2).cpu().numpy();f=(features-features.mean(0))/features.std(0);distance=((f[:,None]-f[None,:])**2).sum(-1);distance[np.arange(128),np.arange(128),:]=np.inf;neighbors=np.argsort(distance,axis=1)[:,:6,:];responses=[];geometry=0.
 variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
 for name,seed in variants:
  ref=heads[:,6]
  if seed is None:pay=ref
  elif name.startswith('moment'):
   gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,128,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[6],roots[6])+means[6];pay=angle_matched_payload(pre,ref,sources[6],direction.reshape(512,128).float())
  else:
   site=(((torch.arange(128,device='cuda')+seed)%128)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[6],heads[site,6]@sources[6])
  geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()),float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1))).max()));z=F.rms_norm(pre-pay,(1152,));responses.append(B(z,z,L,R,D)-native0)
 responses=torch.stack(responses).double();den=responses.square().sum(-1);indices=np.concatenate([np.arange(512)[None],np.stack([(neighbors[:,i,:]*4+np.arange(4)[None,:]).reshape(-1) for i in range(6)])]);energy=np.empty((7,13,512));W=model.lm_head.weight.double()
 for k,idx in enumerate(indices):
  for off in range(0,512,8):
   sl=slice(off,off+8);ids=torch.tensor(idx[sl],device='cuda');state=intact[ids];sd=state.double();rms=(sd.square().mean(1)+torch.finfo(torch.float32).eps).sqrt();d=responses[:,sl].permute(1,0,2);dn=(d-sd[:,None,:]*(d*sd[:,None,:]).mean(-1,keepdim=True)/rms[:,None,None]**2)/rms[:,None,None];rawlog=model.lm_head(F.rms_norm(state,(1152,))).double();prob=(30*torch.tanh(rawlog/30)).softmax(-1);dl=(dn@W.T)*(1-torch.tanh(rawlog/30).square())[:,None,:];variance=(dl.square()*prob[:,None,:]).sum(-1)-(dl*prob[:,None,:]).sum(-1).square();energy[k,:,sl]=variance.T.cpu().numpy()
  print('metric',k,'complete',flush=True)
 ray=(energy/den.cpu().numpy()[None]).reshape(7,13,1,128,4);assert ray.min()>0 and geometry<1e-4;np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),rayleigh=ray,responses=responses.cpu().numpy(),neighbors=neighbors,features=features,indices=indices)
 analyze(cfg,ray,geometry)

def analyze(cfg,ray,geometry):
 raw=ray.copy()
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));ranks=rng.integers(0,6,(4000,6));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};summary={};boots={};audit=0.
 def statistic(v,start,kind):
  A=v[:,0];C=v[:,start:start+6].mean(1)
  return np.exp(np.log(C/A).mean()) if kind=='invariant' else C.mean()/A.mean()
 for path,label in enumerate(['native_mlp']):
  summary[label]={}
  for kind,data in [('invariant',ray),('raw_pooled',raw)]:
   summary[label][kind]={}
   for n,start in [('moment',1),('donor',7)]:
    v=data[:,:,path];own=statistic(v[:1],start,kind);matched=statistic(v[1:],start,kind);values=np.array([own,matched,own/matched]);b=[]
    if kind=='invariant':audit=max(audit,abs(own-statistic(raw[:1,:,path],start,kind)),abs(matched-statistic(raw[1:,:,path],start,kind)))
    for ix,rr,mr in zip(docs,draws[n],ranks):
     A=v[:,0][:,ix];C=v[:,start+rr][:,:,ix].mean(1)
     if kind=='invariant':o=np.exp(np.log(C[:1]/A[:1]).mean());m=np.exp(np.log(C[1+mr]/A[1+mr]).mean())
     else:o=C[:1].mean()/A[:1].mean();m=C[1+mr].mean()/A[1+mr].mean()
     b.append([o,m,o/m])
    b=np.array(b);ci=np.quantile(b,[.025,.975],axis=0);summary[label][kind][n]={k:{'estimate':float(values[i]),'ci95':ci[:,i].tolist()} for i,k in enumerate(['own','matched','own_over_matched'])};boots[label+'_'+kind+'_'+n]=b
 assert audit<1e-10
 gates={f'native_{kind}_pairing':all(summary['native_mlp'][key][n]['own_over_matched']['ci95'][0]>1.2 for n in ['moment','donor']) for kind,key in [('invariant','invariant'),('raw','raw_pooled')]}
 np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**boots);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'gates':gates,'normalization_invariance_error':audit,'geometry_error':geometry},indent=2)+'\n');print(summary,gates,'audit',audit)
if __name__=='__main__':main()
