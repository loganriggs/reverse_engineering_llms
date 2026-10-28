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
STEM=ART/'bilin18_l17_isotropic_geometry_prediction'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias
 cache=np.load(ART/'bilin18_l17_head6_native_confirmation_rows.npz');indices=cache['indices'];unit=F.normalize(intact.double(),dim=1).cpu().numpy();actual=np.load(ART/'bilin18_l17_allhead_native_pairing_rows.npz')['rayleigh'];pred=[];minfrac=1.
 for h in range(9):
  source=cache if h==6 else np.load(ART/f'bilin18_l17_allhead_native_pairing_h{h}_rows.npz');d=source['responses'];d=d/np.linalg.norm(d,axis=-1,keepdims=True);cos=np.einsum('bsd,ksd->kbs',d,unit[indices]);frac=1-cos**2;assert frac.min()>0;minfrac=min(minfrac,float(frac.min()));pred.append(frac.reshape(7,13,128,4))
 pred=np.stack(pred)
 def pairing(data,start):
  lr=np.log(data[:,:,start:start+6].mean(2)/data[:,:,0]);return lr[:,0].mean((1,2))-lr[:,1:].mean((1,2,3))
 summary={}
 for n,start in [('moment',1),('donor',7)]:
  target=pairing(actual,start);prediction=pairing(pred,start);r2=float(1-np.sum((prediction-target)**2)/np.sum(target**2));summary[n]={'actual_pairing':np.exp(target).tolist(),'predicted_pairing':np.exp(prediction).tolist(),'zero_baseline_log_r2':r2,'log_correlation':float(np.corrcoef(target,prediction)[0,1]),'same_sign_heads':int((np.sign(target)==np.sign(prediction)).sum()),'head6_actual_log':float(target[6]),'head6_predicted_log':float(prediction[6])}
 gates={'prediction_r2_half':all(summary[n]['zero_baseline_log_r2']>.5 for n in summary),'seven_signs':all(summary[n]['same_sign_heads']>=7 for n in summary)}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),predicted_rayleigh=pred,indices=indices);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'gates':gates,'minimum_tangent_fraction':minfrac},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
