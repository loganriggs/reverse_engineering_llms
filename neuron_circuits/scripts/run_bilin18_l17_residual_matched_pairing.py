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
STEM=ART/'bilin18_l17_residual_matched_pairing'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias
 cached=np.load(ART/'bilin18_l17_head6_native_confirmation_rows.npz');features=cached['features'];oldneighbors=cached['neighbors'];f=(features-features.mean(0))/features.std(0);distance=((f[:,None]-f[None,:])**2).sum(-1);distance[np.arange(128),np.arange(128),:]=np.inf;candidates=np.argsort(distance,axis=1)[:,:24,:];unit=F.normalize(intact.double(),dim=1).reshape(128,4,1152).cpu().numpy();cos=np.einsum('ipd,jpd->ijp',unit,unit);candidate_cos=np.take_along_axis(cos,candidates,axis=1);order=np.argsort(-candidate_cos,axis=1)[:,:6,:];neighbors=np.take_along_axis(candidates,order,axis=1);assert not (neighbors==np.arange(128)[:,None,None]).any()
 audit={}
 for label,nn in [('confidence',oldneighbors),('residual',neighbors)]:audit[label]={'mean_confidence_distance':float(np.sqrt(np.take_along_axis(distance,nn,axis=1)).mean()),'mean_residual_cosine':float(np.take_along_axis(cos,nn,axis=1).mean())}
 allray=[];allold=[];indices=np.stack([(neighbors[:,i,:]*4+np.arange(4)[None,:]).reshape(-1) for i in range(6)]);W=model.lm_head.weight.double()
 for h in [0,1,6]:
  source=cached if h==6 else np.load(ART/f'bilin18_l17_allhead_native_pairing_h{h}_rows.npz');responses=torch.tensor(source['responses'],device='cuda');den=responses.square().sum(-1);oldray=source['rayleigh'][:,:,0];energy=np.empty((6,13,512))
  for k,idx in enumerate(indices):
   for off in range(0,512,8):
    sl=slice(off,off+8);ids=torch.tensor(idx[sl],device='cuda');state=intact[ids];sd=state.double();rms=(sd.square().mean(1)+torch.finfo(torch.float32).eps).sqrt();d=responses[:,sl].permute(1,0,2);dn=(d-sd[:,None,:]*(d*sd[:,None,:]).mean(-1,keepdim=True)/rms[:,None,None]**2)/rms[:,None,None];rawlog=model.lm_head(F.rms_norm(state,(1152,))).double();prob=(30*torch.tanh(rawlog/30)).softmax(-1);dl=(dn@W.T)*(1-torch.tanh(rawlog/30).square())[:,None,:];variance=(dl.square()*prob[:,None,:]).sum(-1)-(dl*prob[:,None,:]).sum(-1).square();energy[k,:,sl]=variance.T.cpu().numpy()
  newray=(energy/den.cpu().numpy()[None]).reshape(6,13,128,4);allray.append(np.concatenate([oldray[:1],newray]));allold.append(oldray);print('head',h,'complete',flush=True)
 ray=np.stack(allray);oldray=np.stack(allold);np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),rayleigh=ray,old_rayleigh=oldray,neighbors=neighbors,indices=indices,features=features)
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));ranks=rng.integers(0,6,(4000,6));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};est=[];boot=[]
 for data in [oldray,ray]:
  ee=[];bb=[]
  for n,start in [('moment',1),('donor',7)]:
   A=data[:,:,0];C=data[:,:,start:start+6].mean(2);lr=np.log(C/A);ee.append(lr[:,0].mean((1,2))-lr[:,1:].mean((1,2,3)));b=[]
   for ix,rr,mr in zip(docs,draws[n],ranks):
    aa=A[:,:,ix];cc=data[:,:,start+rr][:,:,:,ix].mean(2);l=np.log(cc/aa);b.append(l[:,0].mean((1,2))-l[:,1+mr].mean((1,2,3)))
   bb.append(b)
  est.append(ee);boot.append(np.stack(bb,1))
 est=np.array(est);boot=np.stack(boot,1);radius=float(np.quantile(np.abs(boot[:,1]-est[1]).max((1,2)),.95));summary={n:{'heads':[0,1,6],'confidence_pairing':np.exp(est[0,i]).tolist(),'residual_pairing':np.exp(est[1,i]).tolist(),'residual_joint6_ci95':np.stack([np.exp(est[1,i]-radius),np.exp(est[1,i]+radius)],-1).tolist()} for i,n in enumerate(['moment','donor'])};delta=est[1,:,2]-est[0,:,2];bd=boot[:,1,:,2]-boot[:,0,:,2];rr=float(np.quantile(np.abs(bd-delta).max(1),.95));contrast={n:{'residual_over_confidence_head6':float(np.exp(delta[i])),'joint2_ci95':np.exp([delta[i]-rr,delta[i]+rr]).tolist()} for i,n in enumerate(['moment','donor'])};gates={'head6_persists':all(summary[n]['residual_joint6_ci95'][2][0]>1.2 for n in summary),'head6_attenuation10':all(contrast[n]['joint2_ci95'][1]<.9 for n in summary),'cosine_improvement05':audit['residual']['mean_residual_cosine']-audit['confidence']['mean_residual_cosine']>.05}
 np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),log_pairing=boot);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrast':contrast,'matching_audit':audit,'gates':gates},indent=2)+'\n');print('FINISHED',summary,contrast,audit,gates,flush=True)
if __name__=='__main__':main()
