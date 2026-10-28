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
STEM=ART/'bilin18_l17_head6_local_sensitivity'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;output=p['outputs'].double();saved=np.load(ART/'bilin18_l17_head6_compiled_modes_rows.npz');metric=torch.tensor(saved['output_metric'],device='cuda');pd={k:v.double() for k,v in p.items()};mread=mu.double()@pd['reads']-pd['offset'];mm=torch.einsum('k,okr->or',mread,pd['eigenvectors']);cx=torch.einsum('nk,okr->nor',(x.double()-mu.double())@pd['reads'],pd['eigenvectors']);selected=torch.tensor(cfg['selected_modes'],device='cuda');rows={};term_replay=0.
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
 a0=coefficients(x);compiled0=forward_normalized(x,p);rows={};old=np.load(ART/'bilin18_l17_head_pairing_confirmation_rows.npz');replay=0.;algebra=0.;geometry=0.;variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
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
  rows[name+'_sensitivity']=torch.stack([delta.square().sum(1),proj.sum(1),energy.sum(1),energy[:,:4].sum(1),proj[:,:4].sum(1)],1).reshape(128,4,5).sum(1).cpu().numpy()
  oldenergy=old[f'{name}_h6_gram'][:,1,1];replay=max(replay,float(np.linalg.norm(doc.sum((1,2))-oldenergy)/np.linalg.norm(oldenergy)));print(name,'complete',flush=True)
 assert geometry<1e-4 and term_replay<1e-4 and sensitivity_replay<1e-5
 summary={}
 for name in ['actual','moment','donor']:
  arr=rows['actual_sensitivity'] if name=='actual' else np.mean([rows[f'{name}{i}_sensitivity'] for i in range(6)],axis=0)
  v=arr.sum(0);summary[name]={'input_energy':float(v[0]),'active_projection_fraction':float(v[1]/v[0]),'tangent_energy':float(v[2]),'rayleigh':float(v[2]/v[0]),'within_active_gain':float(v[2]/v[1]),'top4_response_fraction':float(v[3]/v[2]),'top4_input_fraction':float(v[4]/v[0])}
 ratios={n:{k:summary[n][k]/summary['actual'][k] for k in ['input_energy','active_projection_fraction','rayleigh','within_active_gain']} for n in ['moment','donor']}
 gates={'rayleigh2':all(ratios[n]['rayleigh']>2 for n in ratios),'projection2':all(ratios[n]['active_projection_fraction']>2 for n in ratios),'within_active2':all(ratios[n]['within_active_gain']>2 for n in ratios)}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,**spectral)
 STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'ratios':ratios,'gates':gates,'sensitivity_replay':sensitivity_replay,'rank_range':[int(spectral['rank'].min()),int(spectral['rank'].max())]},indent=2)+'\n');print('FINISHED',summary,ratios,gates,flush=True)
if __name__=='__main__':main()
