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
STEM=ART/'bilin18_l17_head6_sensitivity_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;output=p['outputs'].double();metrics=[]
 for off in range(0,512,cfg['logit_batch']):
  sl=slice(off,off+cfg['logit_batch']);state=intact[sl];sd=state.double();rms=(sd.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();delta=output[None].expand(len(state),-1,-1);dn=(delta-sd[:,None,:]*(delta*sd[:,None,:]).mean(2,keepdim=True)/rms[:,None,:].square())/rms[:,None,:];rawlog=model.lm_head(F.rms_norm(state,(1152,))).double();prob=(30*torch.tanh(rawlog/30)).softmax(-1);dl=torch.einsum('bkd,vd->bkv',dn,model.lm_head.weight.double())*(1-torch.tanh(rawlog/30).square())[:,None,:];dl=dl-(dl*prob[:,None,:]).sum(2,keepdim=True);metrics.append(torch.einsum('bkv,blv,bv->bkl',dl,dl,prob))
 metric=torch.cat(metrics)
 pd={k:v.double() for k,v in p.items()};mread=mu.double()@pd['reads']-pd['offset'];mm=torch.einsum('k,okr->or',mread,pd['eigenvectors']);cx=torch.einsum('nk,okr->nor',(x.double()-mu.double())@pd['reads'],pd['eigenvectors']);selected=torch.tensor(cfg['selected_modes'],device='cuda');rows={};term_replay=0.
 # Full ambient normalized-input Fisher Jacobian; independent of discovery mean.
 read=x.double()@pd['reads']-pd['offset'];xm=torch.einsum('nk,okr->nor',read,pd['eigenvectors'])
 jr=2*torch.einsum('nor,or,okr->nok',xm,pd['eigenvalues'],pd['eigenvectors'])+pd['linear'][None]
 J=jr@pd['reads'].T
 ev,U=torch.linalg.eigh(metric);sqrtM=(U*ev.clamp_min(0).sqrt()[:,None,:])@U.transpose(1,2)
 W=sqrtM@J;lam,Q=torch.linalg.eigh(W@W.transpose(1,2));lam=lam.flip(-1).clamp_min(0);Q=Q.flip(-1)
 valid=lam>lam[:,:1]*1e-10;V=(W.transpose(1,2)@Q)/lam.clamp_min(1e-30).sqrt()[:,None,:];V=V*valid[:,None,:]
 spectral={'eigenvalues':lam.cpu().numpy(),'rank':valid.sum(1).cpu().numpy()};sensitivity_replay=0.
 def coefficients(z):
  read=z@p['reads']-p['offset'];modes=torch.einsum('nk,okr->nor',read,p['eigenvectors']);return (modes.square()*p['eigenvalues'][None]).sum(-1)+read@p['linear'].T
 a0=coefficients(x);compiled0=forward_normalized(x,p);rows={};replay=0.;algebra=0.;geometry=0.;variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
 for name,seed in variants:
  ref=heads[:,6]
  if seed is None:pay=ref
  elif name.startswith('moment'):
   gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,128,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[6],roots[6])+means[6];pay=angle_matched_payload(pre,ref,sources[6],direction.reshape(512,128).float())
  else:
   site=(((torch.arange(128,device='cuda')+seed)%128)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[6],heads[site,6]@sources[6])
  geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()));z=F.rms_norm(pre-pay,(1152,));dc=(coefficients(z)-a0).double();response=dc@output;original=(forward_normalized(z,p)-compiled0).double();algebra=max(algebra,float((response-original).norm()/original.norm()));gram=dc[:,:,None]*metric*dc[:,None,:];doc=gram.reshape(128,4,16,16).sum(1).cpu().numpy();rows[name+'_gram']=doc;rows[name+'_coefficients']=dc.cpu().numpy();dv=(z.double()-x.double())@pd['reads'];dm=torch.einsum('nk,okr->nor',dv,pd['eigenvectors']);term0=2*(mm[None]*dm*pd['eigenvalues'][None]).sum(-1)+dv@pd['linear'].T;term1=2*(cx*dm*pd['eigenvalues'][None]).sum(-1);term2=(dm.square()*pd['eigenvalues'][None]).sum(-1);parts=torch.stack([term0,term1,term2],1);term_replay=max(term_replay,float((parts.sum(1)-dc).norm()/dc.norm()));rows[name+'_input_terms']=parts.cpu().numpy()
  for scope,ids in [('all',torch.arange(16,device='cuda')),('top4',selected)]:
   pp=parts[:,:,ids];M=metric[:,ids][:,:,ids];tg=torch.einsum('bki,bij,blj->bkl',pp,M,pp);rows[f'{name}_{scope}_term_gram']=tg.reshape(128,4,3,3).sum(1).cpu().numpy()
  delta=z.double()-x.double();coord=torch.einsum('bi,bik->bk',delta,V);proj=coord.square();energy=proj*lam
  tangent=parts[:,:2].sum(1);exact=torch.einsum('bi,bij,bj->b',tangent,metric,tangent);sensitivity_replay=max(sensitivity_replay,float((energy.sum(1)-exact).norm()/exact.norm()))
  rows[name+'_site_sensitivity']=torch.stack([delta.square().sum(1),proj.sum(1),energy.sum(1),energy[:,:4].sum(1),proj[:,:4].sum(1)],1).reshape(128,4,5).cpu().numpy()
  rows[name+'_sensitivity']=rows[name+'_site_sensitivity'].sum(1)
  print(name,'complete',flush=True)
 assert geometry<1e-4 and term_replay<1e-4 and sensitivity_replay<1e-5

 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,**spectral,output_metric=metric.cpu().numpy())
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(cfg['bootstrap_samples'],128));summary={};boots={};A=rows['actual_site_sensitivity'];lmax=spectral['eigenvalues'][:,0].reshape(128,4)
 def stats(A,C,lmax):
  P=A[...,1];Q=C[...,1];ga=A[...,2]/P;gc=C[...,2]/Q;wa=P/P.sum();wc=Q/Q.sum()
  local=float((((wa+wc)/2)*(gc-ga)).sum());weight=float(((wc-wa)*((ga+gc)/2)).sum());delta=float((wc*gc).sum()-(wa*ga).sum());assert np.isclose(local+weight,delta,rtol=1e-10,atol=1e-12)
  return np.array([np.exp(np.log(gc/ga).mean()),(gc>ga).mean(),(gc/lmax).mean()/(ga/lmax).mean(),local/delta,weight/delta,local,weight])
 for n in ['moment','donor']:
  C=np.stack([rows[f'{n}{i}_site_sensitivity'] for i in range(6)]);v=stats(A,C.mean(0),lmax);draw=rng.integers(0,6,(cfg['bootstrap_samples'],6));b=np.stack([stats(A[ix],C[rr][:,ix].mean(0),lmax[ix]) for ix,rr in zip(docs,draw)]);ci=np.quantile(b,[.025,.975],axis=0);keys=['geomean_site_gain_ratio','fraction_sites_gain_larger','equal_site_normalized_gain_ratio','local_gain_excess_fraction','weighting_excess_fraction','local_gain_contribution','weighting_contribution'];summary[n]={k:{'estimate':float(v[i]),'ci95':ci[:,i].tolist()} for i,k in enumerate(keys)};boots[n]=b
 gates={'local_majority':all(summary[n]['local_gain_excess_fraction']['ci95'][0]>.5 for n in summary),'equal_site_gain2':all(summary[n]['equal_site_normalized_gain_ratio']['ci95'][0]>2 for n in summary),'majority_sites':all(summary[n]['fraction_sites_gain_larger']['ci95'][0]>.5 for n in summary)}
 np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**boots);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'gates':gates,'geometry_error':geometry,'coefficient_replay':term_replay,'sensitivity_replay':sensitivity_replay},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
