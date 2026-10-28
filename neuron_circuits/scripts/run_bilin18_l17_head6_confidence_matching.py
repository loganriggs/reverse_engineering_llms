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
STEM=ART/'bilin18_l17_head6_confidence_matching'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;saved=np.load(ART/'bilin18_l17_head6_sensitivity_confirmation_rows.npz');compiled0=forward_normalized(x,p);rows={};replay=0.;geometry=0.
 features=[]
 for off in range(0,512,32):
  logits=30*torch.tanh(model.lm_head(F.rms_norm(intact[off:off+32],(1152,))).double()/30);lp=logits.log_softmax(-1);prob=lp.exp();top=prob.max(1).values;features.append(torch.stack([-(prob*lp).sum(1),torch.logit(top.clamp(1e-10,1-1e-10))],1))
 features=torch.cat(features).reshape(128,4,2).cpu().numpy();np.savez_compressed(STEM.with_name(STEM.name+'_features.npz'),features=features)
 analyze(cfg,features)

def analyze(cfg,features):
 train=np.load(ART/'bilin18_l17_head6_compiled_modes_rows.npz');ev=np.load(ART/'bilin18_l17_head6_sensitivity_confirmation_rows.npz');p=np.load(ART/'bilin18_l17_compiled_partners_m16_factorized.npz');O=p['outputs'].astype('float64');G=O@O.T;inv=np.linalg.inv(G)
 def norm(M):
  tr=np.einsum('bij,ji->b',M,inv);assert tr.min()>0;return M/tr[:,None,None]
 M=norm(ev['output_metric']).reshape(128,4,16,16);f=(features-features.mean(0))/features.std(0);distance=((f[:,None]-f[None,:])**2).sum(-1);distance[np.arange(128),np.arange(128),:]=np.inf;neighbors=np.argsort(distance,axis=1)[:,:6,:];matched=np.stack([M[neighbors[:,i,:],np.arange(4)[None,:]] for i in range(6)]);metrics={'own':M[None],'matched':matched,'shuffled':np.stack([np.roll(M,-shift,axis=0) for shift in cfg['shifts']])};rows={}
 matchdist=np.sqrt(np.take_along_axis(distance,neighbors,axis=1));randomdist=np.stack([np.sqrt(distance[np.arange(128),(np.arange(128)+shift)%128,:]) for shift in cfg['shifts']]);audit={'matched_mean_standardized_distance':float(matchdist.mean()),'random_mean_standardized_distance':float(randomdist.mean()),'matched_max_distance':float(matchdist.max()),'self_matches':int((neighbors==np.arange(128)[:,None,None]).sum())};assert audit['self_matches']==0

 for n in ['actual']+[f'{k}{i}' for k in ['moment','donor'] for i in range(6)]:
  c=ev[n+'_coefficients'].reshape(128,4,16);den=np.einsum('bpi,ij,bpj->bp',c,G,c)
  for key,metric in metrics.items():rows[n+'_'+key]=np.einsum('bpi,rbpij,bpj->rbp',c,metric,c)/den[None]
 rng=np.random.default_rng(cfg['bootstrap_seed']);docs=rng.integers(0,128,(4000,128));draws={n:rng.integers(0,6,(4000,6)) for n in ['moment','donor']};metricdraw=rng.integers(0,6,(4000,6));summary={};boots={}
 for key in metrics:
  A=rows['actual_'+key];summary[key]={}
  for n in ['moment','donor']:
   C=np.stack([rows[f'{n}{i}_'+key] for i in range(6)]);v=C.mean()/A.mean();b=[]
   for ix,rr,mr in zip(docs,draws[n],metricdraw):
    mm=mr if key in ['shuffled','matched'] else np.array([0]);b.append(C[rr][:,mm][:,:,ix].mean()/A[mm][:,ix].mean())
   b=np.array(b);boots[key+'_'+n]=b;summary[key][n]={'ratio':float(v),'ci95':np.quantile(b,[.025,.975]).tolist()}
 contrast={n:{'own_over_matched':summary['own'][n]['ratio']/summary['matched'][n]['ratio'],'ci95':np.quantile(boots['own_'+n]/boots['matched_'+n],[.025,.975]).tolist()} for n in ['moment','donor']};gates={'pairing_survives_matching':all(contrast[n]['ci95'][0]>1.2 for n in contrast),'matching_distance_half':audit['matched_mean_standardized_distance']<.5*audit['random_mean_standardized_distance']}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,neighbors=neighbors,features=features);np.savez_compressed(STEM.with_name(STEM.name+'_bootstrap.npz'),**boots);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrast':contrast,'gates':gates,'matching_audit':audit},indent=2)+'\n');print('FINISHED',summary,contrast,gates,audit,flush=True)
if __name__=='__main__':main()
