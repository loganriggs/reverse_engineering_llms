"""H330 relative behavioral scale of exact source-response components."""
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
STEM=ART/'bilin18_l17_response_behavior_scale'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));gen=torch.Generator(device='cuda').manual_seed(cfg['fresh_payload_seed']);zs=[];changed_angle=[]
 for h in range(9):
  source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=gen,device='cuda'));zs.append(F.rms_norm(pre-pay,(1152,)));changed_angle.append(pre-pay)
 zs=[F.rms_norm(pre-heads[:,h],(1152,)) for h in range(9)]+zs
 changed=[pre-heads[:,h] for h in range(9)]+changed_angle;native0=B(x,x,L,R,D);compiled0=forward_normalized(x,p);rows={};audits={'response_identity_relative':0.,'missing_kl_replay_relative':0.};old=np.load(ART/'bilin18_l17_factor_mixture_behavior_rows.npz')
 def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
 for g,z in enumerate(zs):
  native=B(z,z,L,R,D);direct=changed[g]-pre;mlp=native-native0;compiled=forward_normalized(z,p)-compiled0;linear=mean_linear(z-x,mu,L,R,D,p);quad=torch.cat([quadratic_terms(x[i:i+256],z[i:i+256],mu,c,L,R,D,p) for i in range(0,512,256)]);partner=torch.cat([cross_response(x[i:i+256],z[i:i+256],mu,c,q,L,R,D,p) for i in range(0,512,256)]);missing=quad[:,1]-partner
  pieces={'direct':direct,'compiled':compiled,'linear':linear,'within':quad[:,0],'partner_cross':partner,'missing_cross':missing,'complement':quad[:,2]};source=direct+mlp;reconstructed=sum(pieces.values());audits['response_identity_relative']=max(audits['response_identity_relative'],float((reconstructed-source).norm()/source.norm()));terms={'source':source,'mlp':mlp,'residual':mlp-compiled,**pieces};gold=changed[g]+native+model.blocks[17].mlp.Down_bias;kls={n:[] for n in terms};tops={n:[] for n in terms}
  for off in range(0,512,cfg['logit_batch']):
   sl=slice(off,off+cfg['logit_batch']);truth=logits(gold[sl]);lp=truth.double().log_softmax(-1);prob=lp.exp()
   for name,term in terms.items():
    out=logits(gold[sl]-term[sl]);kl=(prob*(lp-out.double().log_softmax(-1))).sum(1);assert float(kl.min())>-1e-8;kls[name].append(kl);tops[name].append((out.argmax(-1)==truth.argmax(-1)).double())
  for name,term in terms.items():
   rows[f'{name}_g{g}_energy']=term.double().square().sum(1).reshape(128,4).sum(1).cpu().numpy();rows[f'{name}_g{g}_kl']=torch.cat(kls[name]).reshape(128,4).sum(1).cpu().numpy();rows[f'{name}_g{g}_top1']=torch.cat(tops[name]).reshape(128,4).sum(1).cpu().numpy()
  before=old[f'omitted_g{g}_kl'];after=rows[f'missing_cross_g{g}_kl'];audits['missing_kl_replay_relative']=max(audits['missing_kl_replay_relative'],float(np.linalg.norm(after-before)/np.linalg.norm(before)));print('group',g,'complete',flush=True)
 assert audits['response_identity_relative']<1e-4;assert audits['missing_kl_replay_relative']<1e-8
 boot=np.random.default_rng(cfg['bootstrap_seed']).integers(128,size=(2000,128));summary={};ratios={}
 for family,groups in [('actual',range(9)),('angle',range(9,18))]:
  summary[family]={};kls={n:sum(rows[f'{n}_g{g}_kl'] for g in groups) for n in terms};source_energy=sum(rows[f'source_g{g}_energy'].sum() for g in groups)
  for name in terms:
   summary[family][name]={'mean_kl':float(kls[name].sum()/(128*4*9)),'top1_agreement':float(sum(rows[f'{name}_g{g}_top1'].sum() for g in groups)/(128*4*9)),'energy_relative_to_source':float(sum(rows[f'{name}_g{g}_energy'].sum() for g in groups)/source_energy)}
  ratios[family]={}
  for denom in ['source','mlp','residual']:
   num=kls['missing_cross'];d=kls[denom];ratios[family][denom]={'ratio':float(num.sum()/d.sum()),'paired95':np.quantile(num[boot].sum(1)/d[boot].sum(1),[.025,.975]).tolist()}
 gates={'missing_under1pct_source':all(v['source']['ratio']<.01 for v in ratios.values()),'missing_under1pct_mlp':all(v['mlp']['ratio']<.01 for v in ratios.values())};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'ratios':ratios,'audits':audits,'gates':gates},indent=2)+'\n');print('FINISHED',summary,ratios,audits,gates,flush=True)
if __name__=='__main__':main()
