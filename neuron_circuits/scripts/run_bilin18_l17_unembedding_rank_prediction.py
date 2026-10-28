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
STEM=ART/'bilin18_l17_unembedding_rank_prediction'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias
 cache=np.load(ART/'bilin18_l17_head6_native_confirmation_rows.npz');indices=cache['indices'];u=F.normalize(intact.double(),dim=1)[torch.tensor(indices,device='cuda')];old=np.load(ART/'bilin18_l17_fixed_unembedding_prediction_rows.npz');K=torch.tensor(old['output_metric'],device='cuda');lam,V=torch.linalg.eigh(K);lam=lam.flip(0).clamp_min(0);V=V.flip(1);uv=u@V;ranks=cfg['ranks'];pred={str(k):[] for k in ranks};pred['random16']=[];gen=torch.Generator(device='cuda').manual_seed(cfg['random_seed']);Q,rr=torch.linalg.qr(torch.randn(1152,16,device='cuda',dtype=torch.float64,generator=gen));Q=Q*torch.sign(torch.diag(rr))[None];uq=u@Q;audit=0.
 for h in range(9):
  source=cache if h==6 else np.load(ART/f'bilin18_l17_allhead_native_pairing_h{h}_rows.npz');d=F.normalize(torch.tensor(source['responses'],device='cuda'),dim=-1);cos=torch.einsum('bsd,ksd->kbs',d,u);dv=d@V;energy=((dv[None]-cos[...,None]*uv[:,None]).square()*lam).cumsum(-1)
  for k in ranks:pred[str(k)].append(energy[...,k-1].cpu().numpy().reshape(7,13,128,4))
  audit=max(audit,float(np.linalg.norm(pred['1152'][-1]-old['predicted_rayleigh'][h])/np.linalg.norm(old['predicted_rayleigh'][h])));randomenergy=((d@Q)[None]-cos[...,None]*uq[:,None]).square()@lam[:16];pred['random16'].append(randomenergy.cpu().numpy().reshape(7,13,128,4));print('head',h,'complete',flush=True)
 assert audit<1e-10
 pred={k:np.stack(v) for k,v in pred.items()};actual=np.load(ART/'bilin18_l17_allhead_native_pairing_rows.npz')['rayleigh']
 def pairing(data,start):
  lr=np.log(data[:,:,start:start+6].mean(2)/data[:,:,0]);return lr[:,0].mean((1,2))-lr[:,1:].mean((1,2,3))
 summary={}
 for key,data in pred.items():
  summary[key]={}
  for n,start in [('moment',1),('donor',7)]:
   target=pairing(actual,start);full=pairing(pred['1152'],start);v=pairing(data,start);summary[key][n]={'actual_log_r2':float(1-np.sum((v-target)**2)/np.sum(target**2)),'full_metric_log_fidelity':float(1-np.sum((v-full)**2)/np.sum(full**2)),'same_sign_heads':int((np.sign(v)==np.sign(target)).sum()),'predicted_pairing':np.exp(v).tolist()}
 gates={'rank16_fidelity95':all(summary['16'][n]['full_metric_log_fidelity']>.95 for n in ['moment','donor']),'rank16_actual_half':all(summary['16'][n]['actual_log_r2']>.5 for n in ['moment','donor']),'rank16_orientation_gain20':all(summary['16'][n]['actual_log_r2']-summary['random16'][n]['actual_log_r2']>.2 for n in ['moment','donor'])}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**pred,eigenvalues=lam.cpu().numpy(),eigenvectors=V.cpu().numpy(),random_basis=Q.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'gates':gates,'full_replay':audit,'rank_trace_fraction':{str(k):float(lam[:k].sum()/lam.sum()) for k in ranks}},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
