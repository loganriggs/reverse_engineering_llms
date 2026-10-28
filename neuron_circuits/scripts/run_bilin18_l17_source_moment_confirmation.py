"""H335 fresh document confirmation of source moments with shared intact metric."""
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
STEM=ART/'bilin18_l17_source_moment_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 moments=np.load(ART/'bilin18_l17_source_moment_controls_moments.npz');means=[torch.tensor(v,device='cuda') for v in moments['means']];roots=[torch.tensor(v,device='cuda') for v in moments['roots']];sources=[torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q for h in range(9)]
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));gen=torch.Generator(device='cuda').manual_seed(cfg['fresh_payload_seed']);zs=[];changed_angle=[]
 for h in range(9):
  source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=gen,device='cuda'));zs.append(F.rms_norm(pre-pay,(1152,)));changed_angle.append(pre-pay)
 zs=[F.rms_norm(pre-heads[:,h],(1152,)) for h in range(9)]+zs
 changed=[pre-heads[:,h] for h in range(9)]+changed_angle;geometry={'max_norm_relative':0.,'max_dot_scaled':0.}
 for shift in cfg['donor_shifts']:
  donor=(torch.arange(128,device='cuda')+shift)%128;site=(donor[:,None]*4+torch.arange(4,device='cuda')).reshape(-1)
  for h in range(9):
   source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;ref=heads[:,h];pay=angle_matched_payload(pre,ref,source,heads[site,h]@source);geometry['max_norm_relative']=max(geometry['max_norm_relative'],float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1).clamp_min(1e-20)).max()));geometry['max_dot_scaled']=max(geometry['max_dot_scaled'],float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1)).clamp_min(1e-20)).max()));changed.append(pre-pay);zs.append(F.rms_norm(pre-pay,(1152,)))
 assert geometry['max_norm_relative']<1e-4 and geometry['max_dot_scaled']<1e-4
 momentgen=torch.Generator(device='cuda').manual_seed(cfg['moment_seed']);noise=torch.randn(9,128,4,128,generator=momentgen,device='cuda',dtype=torch.float64)
 for condition in ['mean','centered','mean_cov']:
  for h in range(9):
   direction=torch.einsum('npi,pji->npj',noise[h],roots[h])
   if condition=='mean':direction=means[h][None].expand(128,-1,-1)
   elif condition=='mean_cov':direction=direction+means[h]
   ref=heads[:,h];pay=angle_matched_payload(pre,ref,sources[h],direction.reshape(512,128).float());geometry['max_norm_relative']=max(geometry['max_norm_relative'],float(((pay.norm(dim=1)-ref.norm(dim=1)).abs()/ref.norm(dim=1).clamp_min(1e-20)).max()));geometry['max_dot_scaled']=max(geometry['max_dot_scaled'],float(((pre*(pay-ref)).sum(1).abs()/(pre.norm(dim=1)*ref.norm(dim=1)).clamp_min(1e-20)).max()));changed.append(pre-pay);zs.append(F.rms_norm(pre-pay,(1152,)))
 assert max(geometry.values())<1e-4
 native0=B(x,x,L,R,D);compiled0=forward_normalized(x,p);rows={};combinations={'direct':[0],'compiled':[1],'residual':[2],'direct_compiled':[0,1],'direct_residual':[0,2],'mlp':[1,2],'source':[0,1,2]}
 def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
 for g,z in enumerate(zs):
  native=B(z,z,L,R,D);direct=changed[g]-pre;compiled=forward_normalized(z,p)-compiled0;residual=native-native0-compiled;parts=torch.stack([direct,compiled,residual],1);gold=changed[g]+native+model.blocks[17].mlp.Down_bias;kls={n:[] for n in combinations};grams=[]
  for off in range(0,512,cfg['logit_batch']):
   sl=slice(off,off+cfg['logit_batch']);state=gold[sl];truth=logits(state);lp=truth.double().log_softmax(-1);prob=lp.exp()
   for name,ids in combinations.items():
    term=parts[sl][:,ids].sum(1);out=logits(state-term);kls[name].append((prob*(lp-out.double().log_softmax(-1))).sum(1))
   metric_state=pre[sl]+native0[sl]+model.blocks[17].mlp.Down_bias;metric_logits=logits(metric_state);metric_prob=metric_logits.double().softmax(-1);sd=metric_state.double();rms=(sd.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();delta=parts[sl].double();dn=(delta-sd[:,None,:]*(delta*sd[:,None,:]).mean(2,keepdim=True)/rms[:,None,:].square())/rms[:,None,:];rawlog=model.lm_head(F.rms_norm(metric_state,(1152,))).double();deriv=1-torch.tanh(rawlog/30).square();dl=torch.einsum('bkd,vd->bkv',dn,model.lm_head.weight.double())*deriv[:,None,:];dl=dl-(dl*metric_prob[:,None,:]).sum(2,keepdim=True);grams.append(torch.einsum('bkv,blv,bv->bkl',dl,dl,metric_prob))
  rows[f'g{g}_fisher_gram']=torch.cat(grams).reshape(128,4,3,3).sum(1).cpu().numpy()
  for name in combinations:
   row=torch.cat(kls[name]).reshape(128,4).sum(1).cpu().numpy();rows[f'{name}_g{g}_kl']=row
  print('group',g,'complete',flush=True)
 summary={};boot=np.random.default_rng(cfg['bootstrap_seed']).integers(128,size=(2000,128))
 for family,groups in [(name,range(9*i,9*(i+1))) for i,name in enumerate(cfg['conditions'])]:
  gram=sum(rows[f'g{g}_fisher_gram'] for g in groups);G=gram.sum(0);ks={n:sum(rows[f'{n}_g{g}_kl'] for g in groups) for n in combinations};result={'finite_mean_kl':{n:float(v.sum()/4608) for n,v in ks.items()},'tangent_mean_kl':{n:float(.5*G[np.ix_(ids,ids)].sum()/4608) for n,ids in combinations.items()},'fisher_gram':G.tolist()}
  cos=lambda gg: (gg[...,0,1]+gg[...,0,2])/np.sqrt(gg[...,0,0]*(gg[...,1,1]+gg[...,2,2]+2*gg[...,1,2]))
  result['direct_mlp_cosine']=float(cos(G));result['direct_mlp_cosine95']=np.quantile(cos(gram[boot].sum(1)),[.025,.975]).tolist();result['direct_compiled_cosine']=float(G[0,1]/np.sqrt(G[0,0]*G[1,1]));result['direct_residual_cosine']=float(G[0,2]/np.sqrt(G[0,0]*G[2,2]));result['finite_interaction_fraction']=float((ks['source'].sum()-ks['direct'].sum()-ks['mlp'].sum())/(ks['direct'].sum()+ks['mlp'].sum()));summary[family]=result
 contrasts={};actual=sum(rows[f'g{g}_fisher_gram'] for g in range(9));angle=sum(rows[f'g{g}_fisher_gram'] for g in range(9,18));cos=lambda gg:(gg[...,0,1]+gg[...,0,2])/np.sqrt(gg[...,0,0]*(gg[...,1,1]+gg[...,2,2]+2*gg[...,1,2]))
 for condition in cfg['conditions'][2:]:
  i=cfg['conditions'].index(condition);gram=sum(rows[f'g{g}_fisher_gram'] for g in range(9*i,9*(i+1)));contrasts[condition]={}
  for baseline,base in [('actual',actual),('angle',angle)]:
   diff=cos(gram[boot].sum(1))-cos(base[boot].sum(1));contrasts[condition][baseline]={'cosine_gap':float(cos(gram.sum(0))-cos(base.sum(0))),'paired95':np.quantile(diff,[.025,.975]).tolist()}
 gates={'mean_cov_actual_equivalence5':contrasts['mean_cov']['actual']['paired95'][0]>-.05 and contrasts['mean_cov']['actual']['paired95'][1]<.05,'centered_beats_angle10':contrasts['centered']['angle']['cosine_gap']<=-.1 and contrasts['centered']['angle']['paired95'][1]<0,'mean_cov_beats_angle10':contrasts['mean_cov']['angle']['cosine_gap']<=-.1 and contrasts['mean_cov']['angle']['paired95'][1]<0}

 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'geometry':geometry,'contrasts':contrasts,'gates':gates},indent=2)+'\n');print('FINISHED',summary,'geometry',geometry,'contrasts',contrasts,gates,flush=True)
if __name__=='__main__':main()
