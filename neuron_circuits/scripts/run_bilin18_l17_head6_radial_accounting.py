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
STEM=ART/'bilin18_l17_head6_radial_accounting'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;saved=np.load(ART/'bilin18_l17_head6_sensitivity_confirmation_rows.npz');compiled0=forward_normalized(x,p);rows={};replay=0.;geometry=0.
 output=p['outputs'].double();metrics={k:[] for k in ['radial_prob','radial_uniform','scale_prob','scale_uniform','correction_prob','correction_uniform']}
 for off in range(0,512,32):
  state=intact[off:off+32];sd=state.double();rms=(sd.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();delta=output[None].expand(len(state),-1,-1);raw=model.lm_head(F.rms_norm(state,(1152,))).double();cap=1-torch.tanh(raw/30).square();prob=(30*torch.tanh(raw/30)).softmax(-1)
  for radial in [True,False]:
   dn=delta/rms[:,None,:]
   if radial:dn=dn-sd[:,None,:]*(delta*sd[:,None,:]).mean(2,keepdim=True)/rms[:,None,:]**3
   dl=torch.einsum('bkd,vd->bkv',dn,model.lm_head.weight.double())*cap[:,None,:]
   if radial:full_dl=dl
   else:correction=full_dl-dl
   for weighted in [True,False]:
    weights=prob if weighted else torch.ones_like(prob)/prob.shape[1];dc=dl-(dl*weights[:,None,:]).sum(2,keepdim=True);key=('radial' if radial else 'scale')+('_prob' if weighted else '_uniform');metrics[key].append(torch.einsum('bkv,blv,bv->bkl',dc,dc,weights))
    if not radial:
     cc=correction-(correction*weights[:,None,:]).sum(2,keepdim=True);metrics['correction'+('_prob' if weighted else '_uniform')].append(torch.einsum('bkv,blv,bv->bkl',cc,cc,weights))
 metrics={k:torch.cat(v) for k,v in metrics.items()};old=torch.tensor(saved['output_metric'],device='cuda');replay=float((metrics['radial_prob']-old).norm()/old.norm());assert replay<1e-8
 names=['actual']+[f'{n}{i}' for n in ['moment','donor'] for i in range(6)]
 for n in names:
  coeff=torch.tensor(saved[n+'_coefficients'],device='cuda');euclidean=(coeff@output).square().sum(1);rows[n+'_euclidean']=euclidean.reshape(128,4).cpu().numpy()
  response=coeff@output;radial_fraction=(response*intact.double()).sum(1).square()/(euclidean*intact.double().square().sum(1));rows[n+'_radial_fraction']=radial_fraction.reshape(128,4).cpu().numpy()
  for key,M in metrics.items():rows[n+'_'+key]=torch.einsum('bi,bij,bj->b',coeff,M,coeff).reshape(128,4).cpu().numpy()
 arrays={}
 for n in names:
  e=rows[n+'_euclidean'];scale=rows[n+'_scale_prob']/e;corr=rows[n+'_correction_prob']/e;full=rows[n+'_radial_prob']/e;cross=full-scale-corr
  arrays[n]=np.stack([scale,corr,cross,full,rows[n+'_radial_fraction']],-1)
 def stats(v):
  x=v.mean(axis=tuple(range(v.ndim-1)));return np.r_[x,1-x[3]/x[0],x[2]/(2*np.sqrt(x[0]*x[1]))]
 keys=['scale_energy','correction_energy','cross_energy','full_energy','radial_fraction','suppression_fraction','scale_correction_cosine'];A=arrays['actual'];summary={'actual':dict(zip(keys,map(float,stats(A))))};contrasts={};boots={};rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128))
 for n in ['moment','donor']:
  C=np.stack([arrays[f'{n}{i}'] for i in range(6)]);summary[n]=dict(zip(keys,map(float,stats(C))));rr=rng.integers(0,6,(4000,6));v=stats(A)-stats(C);b=np.stack([stats(A[ix])-stats(C[draw][:,ix]) for ix,draw in zip(docs,rr)]);boots[n]=b;ci=np.quantile(b,[.025,.975],axis=0);contrasts[n]={k:{'actual_minus_control':float(v[i]),'ci95':ci[:,i].tolist()} for i,k in enumerate(keys)}
 gates={'natural_more_radial':all(contrasts[n]['radial_fraction']['ci95'][0]>0 for n in contrasts),'natural_suppression10pp':all(contrasts[n]['suppression_fraction']['ci95'][0]>.1 for n in contrasts)}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**boots);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrasts':contrasts,'gates':gates,'metric_replay':replay},indent=2)+'\n');print('FINISHED',summary,contrasts,gates,flush=True)
if __name__=='__main__':main()
