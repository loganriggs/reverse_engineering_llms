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
STEM=ART/'bilin18_l17_head6_radial_intervention'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;saved=np.load(ART/'bilin18_l17_head6_sensitivity_confirmation_rows.npz');compiled0=forward_normalized(x,p);rows={};replay=0.;geometry=0.
 def logprob(state):return (30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,))).double()/30)).log_softmax(-1)
 names=['actual']+[f'{n}{i}' for n in ['moment','donor'] for i in range(6)];sd=intact.double();old=np.load(ART/'bilin18_l17_head6_finite_behavior_rows.npz');replay=0.;norm_error=0.;orth_error=0.;equivalence=0.
 for n in names:
  d=torch.tensor(saved[n+'_coefficients'],device='cuda')@p['outputs'].double();alpha=(sd*d).sum(1)/sd.square().sum(1);t=d-alpha[:,None]*sd;matched=t*(d.norm(dim=1)/t.norm(dim=1))[:,None];effective=t/(1+alpha)[:,None];assert (1+alpha).min()>0
  norm_error=max(norm_error,float(((matched.norm(dim=1)-d.norm(dim=1)).abs()/d.norm(dim=1)).max()));orth_error=max(orth_error,float(((sd*t).sum(1).abs()/(sd.norm(dim=1)*t.norm(dim=1))).max()));rows[n+'_alpha']=alpha.reshape(128,4).cpu().numpy()
  normalized0=F.rms_norm((sd+d).float(),(1152,));normalized1=F.rms_norm((sd+effective).float(),(1152,));equivalence=max(equivalence,float((normalized0-normalized1).norm()/normalized0.norm()))
  for key,delta in [('original',d),('tangent',t),('normmatched',matched),('effective',effective)]:
   values=[]
   for off in range(0,512,32):
    sl=slice(off,off+32);lp=logprob(intact[sl]);lq=logprob((sd[sl]+delta[sl]).float());values.append((lp.exp()*(lp-lq)).sum(1))
   vals=torch.cat(values).reshape(128,4).cpu().numpy();assert vals.min()>-1e-10;rows[n+'_'+key]=vals
  replay=max(replay,float(np.linalg.norm(rows[n+'_original']-old[n+'_compiled_1.0'][:,:,0])/np.linalg.norm(old[n+'_compiled_1.0'][:,:,0])));print(n,'complete',flush=True)
 assert norm_error<1e-10 and orth_error<1e-10 and equivalence<1e-6 and replay<1e-3
 summary={};ratios={};boots={};rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']}
 for key in ['original','tangent','normmatched','effective']:
  A=rows['actual_'+key];summary[key]={'actual':float(A.mean())};ratios[key]={}
  for n in ['moment','donor']:
   C=np.stack([rows[f'{n}{i}_'+key] for i in range(6)]);summary[key][n]=float(C.mean());v=C.mean()/A.mean();b=np.array([C[rr][:,ix].mean()/A[ix].mean() for ix,rr in zip(docs,draws[n])]);boots[key+'_'+n]=b;ratios[key][n]={'estimate':float(v),'ci95':np.quantile(b,[.025,.975]).tolist()}
 changes={}
 for n in ['moment','donor']:
  b=boots['tangent_'+n]/boots['original_'+n];changes[n]={'tangent_original_ratio':ratios['tangent'][n]['estimate']/ratios['original'][n]['estimate'],'ci95':np.quantile(b,[.025,.975]).tolist()}
 gates={'tangent_gap2':all(ratios['tangent'][n]['ci95'][0]>2 for n in changes),'radial_removal_halves_gap':all(changes[n]['ci95'][1]<.5 for n in changes)}
 alphas={n:{'mean':float(rows[n+'_alpha'].mean()),'min':float(rows[n+'_alpha'].min()),'max':float(rows[n+'_alpha'].max())} for n in names};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**boots);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'ratios':ratios,'changes':changes,'alpha':alphas,'gates':gates,'original_replay':replay,'norm_error':norm_error,'orthogonality_error':orth_error,'normalized_state_equivalence':equivalence},indent=2)+'\n');print('FINISHED',ratios,changes,gates,flush=True)
if __name__=='__main__':main()
