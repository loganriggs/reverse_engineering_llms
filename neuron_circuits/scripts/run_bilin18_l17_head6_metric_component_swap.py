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
STEM=ART/'bilin18_l17_head6_metric_component_swap'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias
 cache=np.load(ART/'bilin18_l17_head6_native_confirmation_rows.npz');responses=torch.tensor(cache['responses'],device='cuda');den=responses.square().sum(-1).cpu().numpy();indices=cache['indices'][1:];old=cache['rayleigh'][:,:,0].reshape(7,13,512);rays={'OO':np.broadcast_to(old[:1],(6,13,512)).copy(),'DD':old[1:]};W=model.lm_head.weight.double()
 for key in ['OD','DO']:
  energy=np.empty((6,13,512))
  for k,idx in enumerate(indices):
   for off in range(0,512,8):
    sl=slice(off,off+8);ids=torch.tensor(idx[sl],device='cuda');own=intact[sl];donor=intact[ids];normstate=own if key[0]=='O' else donor;logstate=own if key[1]=='O' else donor;sd=normstate.double();rms=(sd.square().mean(1)+torch.finfo(torch.float32).eps).sqrt();d=responses[:,sl].permute(1,0,2);dn=(d-sd[:,None,:]*(d*sd[:,None,:]).mean(-1,keepdim=True)/rms[:,None,None]**2)/rms[:,None,None];raw=model.lm_head(F.rms_norm(logstate,(1152,))).double();prob=(30*torch.tanh(raw/30)).softmax(-1);dl=(dn@W.T)*(1-torch.tanh(raw/30).square())[:,None,:];variance=(dl.square()*prob[:,None,:]).sum(-1)-(dl*prob[:,None,:]).sum(-1).square();energy[k,:,sl]=variance.T.cpu().numpy()
  rays[key]=energy/den[None];print(key,'complete',flush=True)
 for k in rays:rays[k]=rays[k].reshape(6,13,128,4);assert rays[k].min()>0
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rays)
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));ranks=rng.integers(0,6,(4000,6));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};summary={};contributions={};boots={}
 for n,start in [('moment',1),('donor',7)]:
  point={};bypath={}
  for key in ['OO','OD','DO','DD']:
   A=rays[key][:,0];C=rays[key][:,start:start+6];point[key]=float(np.log(C.mean(1)/A).mean());b=np.array([np.log(C[mr][:,rr][:,:,ix].mean(1)/A[mr][:,ix]).mean() for ix,rr,mr in zip(docs,draws[n],ranks)]);bypath[key]=b
  summary[n]={key:{'control_actual_ratio':float(np.exp(point[key])),'ci95':np.exp(np.quantile(bypath[key],[.025,.975])).tolist()} for key in point}
  def parts(v):
   total=v['OO']-v['DD'];norm=.5*(v['OO']-v['DO']+v['OD']-v['DD']);logit=.5*(v['OO']-v['OD']+v['DO']-v['DD']);return np.array([total,norm,logit,norm/total,logit/total,v['OO']-v['OD']-v['DO']+v['DD']])
  v=parts(point);b=parts(bypath).T;assert np.max(np.abs(b[:,0]-b[:,1]-b[:,2]))<1e-10;ci=np.quantile(b,[.025,.975],axis=0);keys=['total_log_pairing','normalization_log_contribution','logit_log_contribution','normalization_share','logit_share','log_interaction'];contributions[n]={key:{'estimate':float(v[i]),'ci95':ci[:,i].tolist()} for i,key in enumerate(keys)};boots[n]=b
 gates={'normalization_majority':all(contributions[n]['normalization_share']['ci95'][0]>.5 for n in contributions),'logit_majority':all(contributions[n]['logit_share']['ci95'][0]>.5 for n in contributions)}
 original=json.loads((ART/'bilin18_l17_head6_native_confirmation.json').read_text());replay=max(abs(summary[n]['OO']['control_actual_ratio']/summary[n]['DD']['control_actual_ratio']-original['summary']['native_mlp']['invariant'][n]['own_over_matched']['estimate']) for n in contributions);assert replay<1e-10
 np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**boots);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contributions':contributions,'gates':gates,'endpoint_pairing_replay':replay},indent=2)+'\n');print('FINISHED',summary,contributions,gates,flush=True)
if __name__=='__main__':main()
