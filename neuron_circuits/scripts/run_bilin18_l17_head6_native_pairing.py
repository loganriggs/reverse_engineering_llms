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
STEM=ART/'bilin18_l17_head6_native_pairing'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;saved=np.load(ART/'bilin18_l17_head6_sensitivity_confirmation_rows.npz');compiled0=forward_normalized(x,p);rows={};replay=0.;geometry=0.
 responses=[];names=[]
 variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
 for name,seed in variants:
  ref=heads[:,6]
  if seed is None:pay=ref
  elif name.startswith('moment'):
   gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,128,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[6],roots[6])+means[6];pay=angle_matched_payload(pre,ref,sources[6],direction.reshape(512,128).float())
  else:
   site=(((torch.arange(128,device='cuda')+seed)%128)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[6],heads[site,6]@sources[6])
  geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()));z=F.rms_norm(pre-pay,(1152,));compiled=forward_normalized(z,p)-compiled0;cached=torch.tensor(saved[name+'_coefficients'],device='cuda')@p['outputs'].double();replay=max(replay,float((compiled.double()-cached).norm()/cached.norm()));mlp=B(z,z,L,R,D)-native0;full=-pay+mlp
  responses.append(torch.stack([compiled,mlp],0));names.append(name)
 responses=torch.stack(responses,0).double();den=responses.square().sum(-1);assert replay<1e-4
 matched=np.load(ART/'bilin18_l17_head6_confidence_matching_rows.npz');neighbors=matched['neighbors'];O=p['outputs'].double().cpu().numpy();G=O@O.T;trace=np.einsum('bij,ji->b',saved['output_metric'],np.linalg.inv(G));assert trace.min()>0
 indices=np.concatenate([np.arange(512)[None],np.stack([(neighbors[:,i,:]*4+np.arange(4)[None,:]).reshape(-1) for i in range(6)])]);energy=np.empty((7,13,2,512));W=model.lm_head.weight.double()
 for k,idx in enumerate(indices):
  for off in range(0,512,8):
   sl=slice(off,off+8);ids=torch.tensor(idx[sl],device='cuda');state=intact[ids];sd=state.double();rms=(sd.square().mean(1)+torch.finfo(torch.float32).eps).sqrt();d=responses[:,:,sl].permute(2,0,1,3).reshape(len(state),26,1152);dn=(d-sd[:,None,:]*(d*sd[:,None,:]).mean(-1,keepdim=True)/rms[:,None,None]**2)/rms[:,None,None];raw=model.lm_head(F.rms_norm(state,(1152,))).double();prob=(30*torch.tanh(raw/30)).softmax(-1);dl=(dn@W.T)*(1-torch.tanh(raw/30).square())[:,None,:];variance=(dl.square()*prob[:,None,:]).sum(-1)-(dl*prob[:,None,:]).sum(-1).square();energy[k,:,:,sl]=variance.reshape(len(state),13,2).permute(1,2,0).cpu().numpy()/trace[idx[sl]][None,None,:]
  print('metric',k,'complete',flush=True)
 ray=energy/den.cpu().numpy()[None];oldown=np.stack([matched[n+'_own'][0] for n in names]).reshape(13,512);oldmatch=np.stack([matched[n+'_matched'] for n in names]).reshape(13,6,512).transpose(1,0,2);audit=max(float(np.linalg.norm(ray[0,:,0]-oldown)/np.linalg.norm(oldown)),float(np.linalg.norm(ray[1:,:,0]-oldmatch)/np.linalg.norm(oldmatch)));assert audit<1e-4
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};ranks=rng.integers(0,6,(4000,6));summary={};boots={};contrast={}
 for path in range(2):
  label=['compiled','native_mlp'][path];summary[label]={};contrast[label]={}
  for mode,arr in [('own',ray[:1,:,path]),('matched',ray[1:,:,path])]:
   arr=arr.reshape(-1,13,128,4);A=arr[:,0];summary[label][mode]={}
   for n,start in [('moment',1),('donor',7)]:
    C=arr[:,start:start+6];value=C.mean()/A.mean();b=[]
    for ix,rr,mr in zip(docs,draws[n],ranks):
     mm=mr if mode=='matched' else np.array([0]);b.append(C[mm][:,rr][:,:,ix].mean()/A[mm][:,ix].mean())
    b=np.array(b);boots[label+'_'+mode+'_'+n]=b;summary[label][mode][n]={'ratio':float(value),'ci95':np.quantile(b,[.025,.975]).tolist()}
  for n in ['moment','donor']:
   b=boots[label+'_own_'+n]/boots[label+'_matched_'+n];contrast[label][n]={'own_over_matched':summary[label]['own'][n]['ratio']/summary[label]['matched'][n]['ratio'],'ci95':np.quantile(b,[.025,.975]).tolist()}
 gates={'native_pairing':all(contrast['native_mlp'][n]['ci95'][0]>1.2 for n in ['moment','donor']),'native_own_gap':all(summary['native_mlp']['own'][n]['ci95'][0]>1.2 for n in ['moment','donor'])}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),rayleigh=ray,responses=responses.cpu().numpy(),indices=indices,reference_trace=trace);np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**boots);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrast':contrast,'gates':gates,'compiled_metric_replay':audit,'response_replay':replay},indent=2)+'\n');print('FINISHED',summary,contrast,gates,flush=True)
if __name__=='__main__':main()
