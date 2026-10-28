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
STEM=ART/'bilin18_l17_head6_metric_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;saved=np.load(ART/'bilin18_l17_head6_sensitivity_confirmation_rows.npz');compiled0=forward_normalized(x,p);rows={};replay=0.;geometry=0.
 output=p['outputs'].double();metrics={k:[] for k in ['radial_prob','radial_uniform','scale_prob','scale_uniform']}
 for off in range(0,512,32):
  state=intact[off:off+32];sd=state.double();rms=(sd.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();delta=output[None].expand(len(state),-1,-1);raw=model.lm_head(F.rms_norm(state,(1152,))).double();cap=1-torch.tanh(raw/30).square();prob=(30*torch.tanh(raw/30)).softmax(-1)
  for radial in [True,False]:
   dn=delta/rms[:,None,:]
   if radial:dn=dn-sd[:,None,:]*(delta*sd[:,None,:]).mean(2,keepdim=True)/rms[:,None,:]**3
   dl=torch.einsum('bkd,vd->bkv',dn,model.lm_head.weight.double())*cap[:,None,:]
   for weighted in [True,False]:
    weights=prob if weighted else torch.ones_like(prob)/prob.shape[1];dc=dl-(dl*weights[:,None,:]).sum(2,keepdim=True);key=('radial' if radial else 'scale')+('_prob' if weighted else '_uniform');metrics[key].append(torch.einsum('bkv,blv,bv->bkl',dc,dc,weights))
 metrics={k:torch.cat(v) for k,v in metrics.items()};old=torch.tensor(saved['output_metric'],device='cuda');replay=float((metrics['radial_prob']-old).norm()/old.norm());assert replay<1e-8
 names=['actual']+[f'{n}{i}' for n in ['moment','donor'] for i in range(6)]
 for n in names:
  coeff=torch.tensor(saved[n+'_coefficients'],device='cuda');euclidean=(coeff@output).square().sum(1);rows[n+'_euclidean']=euclidean.reshape(128,4).cpu().numpy()
  for key,M in metrics.items():rows[n+'_'+key]=torch.einsum('bi,bij,bj->b',coeff,M,coeff).reshape(128,4).cpu().numpy()
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};summary={};boots={}
 for key in metrics:
  A=rows['actual_'+key]/rows['actual_euclidean'];summary[key]={}
  for n in ['moment','donor']:
   # Average per-realization output Rayleigh before site averaging; no aggregate norm reweighting.
   C=np.stack([rows[f'{n}{i}_'+key]/rows[f'{n}{i}_euclidean'] for i in range(6)]);v=C.mean()/A.mean();b=np.array([C[rr][:,ix].mean()/A[ix].mean() for ix,rr in zip(docs,draws[n])]);summary[key][n]={'control_actual_output_rayleigh':float(v),'ci95':np.quantile(b,[.025,.975]).tolist()};boots[key+'_'+n]=b
 contrasts={}
 for n in ['moment','donor']:
  # Paired comparison of probability-weighted vs uniform control/actual ratios at full RMS derivative.
  b=boots['radial_prob_'+n]/boots['radial_uniform_'+n];contrasts[n]={'probability_amplification_ratio':summary['radial_prob'][n]['control_actual_output_rayleigh']/summary['radial_uniform'][n]['control_actual_output_rayleigh'],'ci95':np.quantile(b,[.025,.975]).tolist()}
 gates={'probability_amplification2':all(contrasts[n]['ci95'][0]>2 for n in contrasts),'uniform_gap_under1_5':all(summary['radial_uniform'][n]['ci95'][1]<1.5 for n in contrasts)}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,**{k+'_metric':v.cpu().numpy() for k,v in metrics.items()});np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**boots);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrasts':contrasts,'gates':gates,'metric_replay':replay},indent=2)+'\n');print('FINISHED',summary,contrasts,gates,flush=True)
if __name__=='__main__':main()
