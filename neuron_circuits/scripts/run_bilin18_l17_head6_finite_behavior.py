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
STEM=ART/'bilin18_l17_head6_finite_behavior'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;saved=np.load(ART/'bilin18_l17_head6_sensitivity_confirmation_rows.npz');compiled0=forward_normalized(x,p);rows={};replay=0.;geometry=0.
 def logprob(state):return (30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,))).double()/30)).log_softmax(-1)
 variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
 for name,seed in variants:
  ref=heads[:,6]
  if seed is None:pay=ref
  elif name.startswith('moment'):
   gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,128,4,128,generator=gen,device='cuda',dtype=torch.float64);direction=torch.einsum('npi,pji->npj',noise[6],roots[6])+means[6];pay=angle_matched_payload(pre,ref,sources[6],direction.reshape(512,128).float())
  else:
   site=(((torch.arange(128,device='cuda')+seed)%128)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[6],heads[site,6]@sources[6])
  geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()));z=F.rms_norm(pre-pay,(1152,));compiled=forward_normalized(z,p)-compiled0;cached=torch.tensor(saved[name+'_coefficients'],device='cuda')@p['outputs'].double();replay=max(replay,float((compiled.double()-cached).norm()/cached.norm()));mlp=B(z,z,L,R,D)-native0;full=-pay+mlp
  for path,delta in [('compiled',compiled),('full',full),('direct',-pay),('remaining',mlp-compiled)]:
   for scale in ([.25,1.] if path in ['compiled','full'] else [1.]):
    values=[]
    for off in range(0,512,32):
     sl=slice(off,off+32);lp=logprob(intact[sl]);lq=logprob(intact[sl]+scale*delta[sl]);kl=(lp.exp()*(lp-lq)).sum(1);agree=(lp.argmax(1)==lq.argmax(1)).double();values.append(torch.stack([kl,agree],1))
    arr=torch.cat(values).reshape(128,4,2).cpu().numpy();assert arr[:,:,0].min()>-1e-10;rows[f'{name}_{path}_{scale}']=arr
  print(name,'complete',flush=True)
 assert geometry<1e-4 and replay<1e-4
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows)
 summary={};ratios={};rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']}
 for key in ['compiled_0.25','compiled_1.0','full_0.25','full_1.0','direct_1.0','remaining_1.0']:
  A=rows['actual_'+key];summary[key]={'actual':{'mean_kl':float(A[:,:,0].mean()),'top1_agreement':float(A[:,:,1].mean())}};ratios[key]={}
  for n in ['moment','donor']:
   C=np.stack([rows[f'{n}{i}_'+key] for i in range(6)]);summary[key][n]={'mean_kl':float(C[:,:,:,0].mean()),'top1_agreement':float(C[:,:,:,1].mean())};b=np.array([C[rr][:,ix,:,0].mean()/A[ix,:,0].mean() for ix,rr in zip(docs,draws[n])]);ratios[key][n]={'estimate':summary[key][n]['mean_kl']/summary[key]['actual']['mean_kl'],'ci95':np.quantile(b,[.025,.975]).tolist()}
 approximation={}
 for n in ['actual','moment','donor']:
  gram=saved['actual_gram'] if n=='actual' else np.mean([saved[f'{n}{i}_gram'] for i in range(6)],0);local=.5*gram.sum()/512;approximation[n]={str(scale):summary[f'compiled_{scale}'][n]['mean_kl']/(local*scale**2) for scale in [.25,1.]}
 gates={'compiled_finite2':all(ratios['compiled_1.0'][n]['ci95'][0]>2 for n in ['moment','donor']),'full_finite2':all(ratios['full_1.0'][n]['ci95'][0]>2 for n in ['moment','donor'])}
 STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'ratios':ratios,'compiled_finite_over_local':approximation,'gates':gates,'geometry_error':geometry,'compiled_replay':replay},indent=2)+'\n');print('FINISHED',ratios,approximation,gates,flush=True)
if __name__=='__main__':main()
