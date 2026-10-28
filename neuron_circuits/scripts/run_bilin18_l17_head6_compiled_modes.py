"""H341 compiled-output mode accounting of head6 amplification."""
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
STEM=ART/'bilin18_l17_head6_compiled_modes'
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
  geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1)).max()));z=F.rms_norm(pre-pay,(1152,));dc=(coefficients(z)-a0).double();response=dc@output;original=(forward_normalized(z,p)-compiled0).double();algebra=max(algebra,float((response-original).norm()/original.norm()));gram=dc[:,:,None]*metric*dc[:,None,:];doc=gram.reshape(128,4,16,16).sum(1).cpu().numpy();rows[name+'_gram']=doc;rows[name+'_coefficients']=dc.cpu().numpy();oldenergy=old[f'{name}_h6_gram'][:,1,1];replay=max(replay,float(np.linalg.norm(doc.sum((1,2))-oldenergy)/np.linalg.norm(oldenergy)));print(name,'complete',flush=True)
 assert geometry<1e-4 and algebra<1e-4 and replay<1e-4
 actual=rows['actual_gram'].sum(0);control={kind:sum(rows[f'{kind}{i}_gram'].sum(0) for i in range(6))/6 for kind in ['moment','donor']};excess={k:G-actual for k,G in control.items()};score=sum(G.sum(1)/G.sum() for G in excess.values())/2;order=np.argsort(score)[::-1];ids=order[:4];other=order[4:];summary={}
 for name,G in [('actual',actual)]+list(control.items()):
  summary[name]={'total_energy':float(G.sum()),'diagonal_energy':float(np.trace(G)),'cross_energy':float(G.sum()-np.trace(G)),'mode_energy_allocation':G.sum(1).tolist(),'top4_kept_fidelity':float(1-G[np.ix_(other,other)].sum()/G.sum())}
  if name!='actual':summary[name].update({'excess_allocation_fraction':(excess[name].sum(1)/excess[name].sum()).tolist(),'top4_excess_allocation':float(excess[name][ids].sum()/excess[name].sum()),'top4_only_excess_energy':float(excess[name][np.ix_(ids,ids)].sum()/excess[name].sum())})
 gates={'top4_excess80':all(summary[n]['top4_excess_allocation']>=.8 for n in control),'top4_fidelity80':all(summary[n]['top4_kept_fidelity']>=.8 for n in control)};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,output_metric=metric.cpu().numpy(),ranking=order);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'top4_ids':ids.tolist(),'ranking':order.tolist(),'geometry_error':geometry,'coefficient_algebra_relative':algebra,'previous_energy_replay_relative':replay,'gates':gates},indent=2)+'\n');print('FINISHED',summary,'top4',ids.tolist(),'replay',replay,gates,flush=True)
if __name__=='__main__':main()
