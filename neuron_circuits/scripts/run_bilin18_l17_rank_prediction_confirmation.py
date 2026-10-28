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
STEM=ART/'bilin18_l17_rank_prediction_confirmation'
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
 frozen=np.load(ART/'bilin18_l17_unembedding_rank_prediction_rows.npz');lam=torch.tensor(frozen['eigenvalues'],device='cuda');V=torch.tensor(frozen['eigenvectors'],device='cuda');Q=torch.tensor(frozen['random_basis'],device='cuda');predictions={k:[] for k in ['2','16','1152','random16']};allray=[]
 for h in range(9):
  responses=[];geometry=0.
  variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
  for name,seed in variants:
   ref=heads[:,h]
   if seed is None:pay=ref
   elif name.startswith('moment'):
    gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,128,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[h],roots[h])+means[h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(512,128).float())
   else:
    site=(((torch.arange(128,device='cuda')+seed)%128)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[h],heads[site,h]@sources[h])
   geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()),float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1))).max()));z=F.rms_norm(pre-pay,(1152,));responses.append(B(z,z,L,R,D)-native0)
  responses=torch.stack(responses).double();den=responses.square().sum(-1);indices=np.concatenate([np.arange(512)[None],np.stack([(neighbors[:,i,:]*4+np.arange(4)[None,:]).reshape(-1) for i in range(6)])]);energy=np.empty((7,13,512));W=model.lm_head.weight.double()
  for k,idx in enumerate(indices):
   for off in range(0,512,8):
    sl=slice(off,off+8);ids=torch.tensor(idx[sl],device='cuda');state=intact[ids];sd=state.double();rms=(sd.square().mean(1)+torch.finfo(torch.float32).eps).sqrt();d=responses[:,sl].permute(1,0,2);dn=(d-sd[:,None,:]*(d*sd[:,None,:]).mean(-1,keepdim=True)/rms[:,None,None]**2)/rms[:,None,None];rawlog=model.lm_head(F.rms_norm(state,(1152,))).double();prob=(30*torch.tanh(rawlog/30)).softmax(-1);dl=(dn@W.T)*(1-torch.tanh(rawlog/30).square())[:,None,:];variance=(dl.square()*prob[:,None,:]).sum(-1)-(dl*prob[:,None,:]).sum(-1).square();energy[k,:,sl]=variance.T.cpu().numpy()
   print('metric',k,'complete',flush=True)
  ray=(energy/den.cpu().numpy()[None]).reshape(7,13,1,128,4);assert ray.min()>0 and geometry<1e-4;np.savez_compressed(STEM.with_name(STEM.name+f'_h{h}_rows.npz'),rayleigh=ray,responses=responses.cpu().numpy(),neighbors=neighbors,features=features,indices=indices)
  u=F.normalize(intact.double(),dim=1)[torch.tensor(indices,device='cuda')];d=F.normalize(responses,dim=-1);cos=torch.einsum('bsd,ksd->kbs',d,u);uv=u@V;dv=d@V;ee=((dv[None]-cos[...,None]*uv[:,None]).square()*lam).cumsum(-1)
  for rank in [2,16,1152]:predictions[str(rank)].append(ee[...,rank-1].cpu().numpy().reshape(7,13,128,4))
  randomenergy=((d@Q)[None]-cos[...,None]*(u@Q)[:,None]).square()@lam[:16];predictions['random16'].append(randomenergy.cpu().numpy().reshape(7,13,128,4));del ee
  allray.append(ray[:,:,0]);print('head',h,'complete',flush=True)
 analyze(cfg,np.stack(allray),{k:np.stack(v) for k,v in predictions.items()})

def analyze(cfg,ray,predictions):
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));ranks=rng.integers(0,6,(4000,6));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};est=[];boot=[]
 for n,start in [('moment',1),('donor',7)]:
  A=ray[:,:,0];C=ray[:,:,start:start+6].mean(2);logratio=np.log(C/A);v=logratio[:,0].mean((1,2))-logratio[:,1:].mean((1,2,3));est.append(v);b=[]
  for ix,rr,mr in zip(docs,draws[n],ranks):
   aa=A[:,:,ix];cc=ray[:,:,start+rr][:,:,:,ix].mean(2);lr=np.log(cc/aa);b.append(lr[:,0].mean((1,2))-lr[:,1+mr].mean((1,2,3)))
  boot.append(np.stack(b))
 est=np.stack(est);boot=np.stack(boot,1);radius=float(np.quantile(np.max(np.abs(boot-est[None]),axis=(1,2)),.95));summary={n:{'own_over_matched':np.exp(est[i]).tolist(),'joint18_ci95':np.stack([np.exp(est[i]-radius),np.exp(est[i]+radius)],-1).tolist()} for i,n in enumerate(['moment','donor'])};lower=np.exp(est-radius);confirmed=np.flatnonzero((lower>1).all(0)).tolist();strong=np.flatnonzero((lower>1.2).all(0)).tolist();gates={'broad_pairing':len(confirmed)>=5,'broad_strong_pairing':len(strong)>=5}
 prediction_summary={}
 for key,data in predictions.items():
  prediction_summary[key]={}
  for i,(n,start) in enumerate([('moment',1),('donor',7)]):
   lr=np.log(data[:,:,start:start+6].mean(2)/data[:,:,0]);v=lr[:,0].mean((1,2))-lr[:,1:].mean((1,2,3));target=est[i];prediction_summary[key][n]={'actual_log_r2':float(1-np.sum((v-target)**2)/np.sum(target**2)),'same_sign_heads':int((np.sign(v)==np.sign(target)).sum()),'log_correlation':float(np.corrcoef(v,target)[0,1]),'predicted_pairing':np.exp(v).tolist()}
 gates={'rank2_r2_half':all(prediction_summary['2'][n]['actual_log_r2']>.5 for n in ['moment','donor']),'rank2_seven_signs':all(prediction_summary['2'][n]['same_sign_heads']>=7 for n in ['moment','donor']),'rank16_r2_half':all(prediction_summary['16'][n]['actual_log_r2']>.5 for n in ['moment','donor']),'rank16_orientation_gain20':all(prediction_summary['16'][n]['actual_log_r2']-prediction_summary['random16'][n]['actual_log_r2']>.2 for n in ['moment','donor'])}
 np.savez_compressed(STEM.with_name(STEM.name+'_predictions.npz'),**predictions)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),rayleigh=ray,bootstrap_log_pairing=boot);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'prediction_summary':prediction_summary,'joint_log_radius':radius,'heads_both_lower_above1':confirmed,'heads_both_lower_above1_2':strong,'gates':gates},indent=2)+'\n');print('FINISHED',prediction_summary,gates,flush=True)
if __name__=='__main__':main()
