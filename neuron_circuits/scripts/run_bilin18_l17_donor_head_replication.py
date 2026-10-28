"""H338 donor-source pairing across six fixed document permutations."""
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
STEM=ART/'bilin18_l17_donor_head_replication'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));native0=B(x,x,L,R,D);intact=pre+native0+model.blocks[17].mlp.Down_bias;compiled0=forward_normalized(x,p);rows={};geometry=0.;old=np.load(ART/'bilin18_l17_source_moment_confirmation_rows.npz');replay=0.
 def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
 variants=[('actual',None)]+[(f'draw{i}',seed) for i,seed in enumerate(cfg['donor_shifts'])]
 for condition,seed in variants:
  if seed is not None:
   doc=(torch.arange(128,device='cuda')+seed)%128;site=(doc[:,None]*4+torch.arange(4,device='cuda')).reshape(-1)
  for h in range(9):
   ref=heads[:,h]
   if seed is None:pay=ref
   else:
    direction=heads[site,h]@sources[h];pay=angle_matched_payload(pre,ref,sources[h],direction);geometry=max(geometry,float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1).clamp_min(1e-20)).max()),float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1)).clamp_min(1e-20)).max()))
   z=F.rms_norm(pre-pay,(1152,));native=B(z,z,L,R,D);compiled=forward_normalized(z,p)-compiled0;parts=torch.stack([-pay,compiled,native-native0-compiled],1);grams=[]
   for off in range(0,512,cfg['logit_batch']):
    sl=slice(off,off+cfg['logit_batch']);state=intact[sl];prob=logits(state).double().softmax(-1);sd=state.double();rms=(sd.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();delta=parts[sl].double();dn=(delta-sd[:,None,:]*(delta*sd[:,None,:]).mean(2,keepdim=True)/rms[:,None,:].square())/rms[:,None,:];rawlog=model.lm_head(F.rms_norm(state,(1152,))).double();deriv=1-torch.tanh(rawlog/30).square();dl=torch.einsum('bkd,vd->bkv',dn,model.lm_head.weight.double())*deriv[:,None,:];dl=dl-(dl*prob[:,None,:]).sum(2,keepdim=True);grams.append(torch.einsum('bkv,blv,bv->bkl',dl,dl,prob))
   row=torch.cat(grams).reshape(128,4,3,3).sum(1).cpu().numpy();rows[f'{condition}_h{h}_gram']=row
   if condition in ['actual','draw0']:
    before=old[f'g{h if condition=="actual" else 18+h}_fisher_gram'];replay=max(replay,float(np.linalg.norm(row-before)/np.linalg.norm(before)))
  print(condition,'complete',flush=True)
 assert geometry<1e-4;assert replay<1e-5
 actual=np.stack([rows[f'actual_h{h}_gram'] for h in range(9)]);draws=np.stack([np.stack([rows[f'draw{i}_h{h}_gram'] for h in range(9)]) for i in range(len(cfg['donor_shifts']))]);cos=lambda G:(G[...,0,1]+G[...,0,2])/np.sqrt(G[...,0,0]*(G[...,1,1]+G[...,2,2]+2*G[...,1,2]));summary={}
 for name,G in [('actual',actual)]+[(f'draw{i}',v) for i,v in enumerate(draws)]+[('draw_mean',draws.mean(0))]:
  total=G.sum(1);energies=total[:,1,1]+total[:,2,2]+2*total[:,1,2];summary[name]={'pooled_cosine':float(cos(total.sum(0))),'head_cosines':cos(total).tolist(),'mean_head_cosine':float(cos(total).mean()),'head6_mlp_energy_share':float(energies[6]/energies.sum())}
 gap=cos(draws.mean(0).sum(1))-cos(actual.sum(1));rng=np.random.default_rng(cfg['bootstrap_seed']);bootg=[]
 for _ in range(4000):
  docs=rng.integers(128,size=128);seeds=rng.integers(len(draws),size=len(draws));G=draws[seeds].mean(0)[:,docs].sum(1);A=actual[:,docs].sum(1);bootg.append(cos(G)-cos(A))
 bootg=np.stack(bootg);radius=float(np.quantile(np.abs(bootg-gap).max(1),.95));bands=np.stack([gap-radius,gap+radius],1);allgaps=np.array([summary[f'draw{i}']['head_cosines'] for i in range(len(draws))])-np.array(summary['actual']['head_cosines']);gates={'heads03_repeat':all(bands[h,0]>0 and np.all(allgaps[:,h]>0) for h in [0,3]),'head6_dominates':all(summary[f'draw{i}']['head6_mlp_energy_share']>=.65 for i in range(len(draws)))}
 mr=np.load(ART/'bilin18_l17_moment_draw_replication_rows.npz');mom=np.stack([np.stack([mr[f'draw{i}_h{h}_gram'] for h in range(9)]) for i in range(6)]);moment_gap=cos(mom.mean(0).sum(1))-cos(draws.mean(0).sum(1));dif=[]
 for _ in range(4000):
  docs=rng.integers(128,size=128);mi=rng.integers(6,size=6);di=rng.integers(6,size=6);dif.append(cos(mom[mi].mean(0)[:,docs].sum(1))-cos(draws[di].mean(0)[:,docs].sum(1)))
 dif=np.stack(dif);rad=float(np.quantile(np.abs(dif-moment_gap).max(1),.95));moment_bands=np.stack([moment_gap-rad,moment_gap+rad],1);comparison={'moment_minus_donor':moment_gap.tolist(),'simultaneous95':moment_bands.tolist()};gates['moment_donor_heads03_equivalent5']=all(moment_bands[h,0]>-.05 and moment_bands[h,1]<.05 for h in [0,3])
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,bootstrap_head_gaps=bootg);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'head_gaps':gap.tolist(),'simultaneous95':bands.tolist(),'max_deviation_radius':radius,'per_draw_head_gaps':allgaps.tolist(),'geometry_error':geometry,'previous_gram_replay_relative':replay,'moment_comparison':comparison,'gates':gates},indent=2)+'\n');print('FINISHED',summary,'gaps',gap.tolist(),'bands',bands.tolist(),'moment_comparison',comparison,gates,flush=True)
if __name__=='__main__':main()
