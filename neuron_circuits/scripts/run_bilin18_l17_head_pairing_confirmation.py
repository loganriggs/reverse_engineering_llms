"""H339 joint fresh confirmation of head-specific pairing and source distribution."""
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
STEM=ART/'bilin18_l17_head_pairing_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;compiled0=forward_normalized(x,p);rows={};geometry=0.;
 def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
 variants=[('actual',None)]+[(f'moment{i}',seed) for i,seed in enumerate(cfg['draw_seeds'])]+[(f'donor{i}',shift) for i,shift in enumerate(cfg['donor_shifts'])]
 for condition,seed in variants:
  if condition.startswith('moment'):
   gen=torch.Generator(device='cuda').manual_seed(seed);noise=torch.randn(9,128,4,128,generator=gen,device='cuda',dtype=torch.float64)
  for h in range(9):
   ref=heads[:,h]
   if seed is None:pay=ref
   elif condition.startswith('donor'):
    site=(((torch.arange(128,device='cuda')+seed)%128)[:,None]*4+torch.arange(4,device='cuda')).reshape(-1);pay=angle_matched_payload(pre,ref,sources[h],heads[site,h]@sources[h])
   else:
    direction=torch.einsum('npi,pji->npj',noise[h],roots[h])+means[h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(512,128).float());geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1).clamp_min(1e-20)).max()),float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1)).clamp_min(1e-20)).max()))
   geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1).clamp_min(1e-20)).max()),float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1)).clamp_min(1e-20)).max()))
   z=F.rms_norm(pre-pay,(1152,));native=B(z,z,L,R,D);compiled=forward_normalized(z,p)-compiled0;parts=torch.stack([-pay,compiled,native-native0-compiled],1);grams=[]
   for off in range(0,512,cfg['logit_batch']):
    sl=slice(off,off+cfg['logit_batch']);state=intact[sl];prob=logits(state).double().softmax(-1);sd=state.double();rms=(sd.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();delta=parts[sl].double();dn=(delta-sd[:,None,:]*(delta*sd[:,None,:]).mean(2,keepdim=True)/rms[:,None,:].square())/rms[:,None,:];rawlog=model.lm_head(F.rms_norm(state,(1152,))).double();deriv=1-torch.tanh(rawlog/30).square();dl=torch.einsum('bkd,vd->bkv',dn,model.lm_head.weight.double())*deriv[:,None,:];dl=dl-(dl*prob[:,None,:]).sum(2,keepdim=True);grams.append(torch.einsum('bkv,blv,bv->bkl',dl,dl,prob))
   row=torch.cat(grams).reshape(128,4,3,3).sum(1).cpu().numpy();rows[f'{condition}_h{h}_gram']=row
  print(condition,'complete',flush=True)
 assert geometry<1e-4
 actual=np.stack([rows[f'actual_h{h}_gram'] for h in range(9)]);arrays={kind:np.stack([np.stack([rows[f'{kind}{i}_h{h}_gram'] for h in range(9)]) for i in range(6)]) for kind in ['moment','donor']};cos=lambda G:(G[...,0,1]+G[...,0,2])/np.sqrt(G[...,0,0]*(G[...,1,1]+G[...,2,2]+2*G[...,1,2]));summary={}
 for name,G in [('actual',actual),('moment',arrays['moment'].mean(0)),('donor',arrays['donor'].mean(0))]:
  total=G.sum(1);e=total[:,1,1]+total[:,2,2]+2*total[:,1,2];summary[name]={'pooled_cosine':float(cos(total.sum(0))),'head_cosines':cos(total).tolist(),'head6_mlp_energy_share':float(e[6]/e.sum())}
 av=np.array(summary['actual']['head_cosines']);mv=np.array(summary['moment']['head_cosines']);dv=np.array(summary['donor']['head_cosines']);gaps=np.stack([dv-av,mv-av,mv-dv]);rng=np.random.default_rng(cfg['bootstrap_seed']);bootstrap=[]
 for _ in range(4000):
  docs=rng.integers(128,size=128);mi=rng.integers(6,size=6);di=rng.integers(6,size=6);aa=cos(actual[:,docs].sum(1));mm=cos(arrays['moment'][mi].mean(0)[:,docs].sum(1));dd=cos(arrays['donor'][di].mean(0)[:,docs].sum(1));bootstrap.append(np.stack([dd-aa,mm-aa,mm-dd]))
 bootstrap=np.stack(bootstrap);radius=float(np.quantile(np.abs(bootstrap-gaps).max(axis=(1,2)),.95));bands=np.stack([gaps-radius,gaps+radius],-1);contrasts={name:{'gaps':gaps[i].tolist(),'simultaneous95':bands[i].tolist()} for i,name in enumerate(['donor_actual','moment_actual','moment_donor'])};gates={'heads03_pairing':all(gaps[0,h]>=.1 and bands[0,h,0]>0 for h in [0,3]),'head5_distribution':gaps[2,5]>=.1 and bands[2,5,0]>0,'head6_concentration':all(summary[k]['head6_mlp_energy_share']>=.65 for k in ['moment','donor'])}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,bootstrap_gaps=bootstrap);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrasts':contrasts,'simultaneous27_radius':radius,'geometry_error':geometry,'gates':{k:bool(v) for k,v in gates.items()}},indent=2)+'\n');print('FINISHED',summary,contrasts,gates,flush=True)
if __name__=='__main__':main()
