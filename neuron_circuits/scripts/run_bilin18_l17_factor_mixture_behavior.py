"""H329 isolated missing-component output-distribution mediation."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import B
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_factor_mixture_behavior'
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
 def coords(v):
  w=v.double()-mu.double();u=w@c.double();return u,w-u@c.double().T-(w@q.double())@q.double().T
 u,v=coords(x);features=[];targets=[];rows={}
 for target,z in enumerate(zs):
  w,t=coords(z);features.append((w,t));y=torch.cat([quadratic_terms(x[i:i+256],z[i:i+256],mu,c,L,R,D,p)[:,1]-cross_response(x[i:i+256],z[i:i+256],mu,c,q,L,R,D,p) for i in range(0,512,256)]).double();targets.append(y);rows[f't{target}_energy']=y.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
 changed=[pre-heads[:,h] for h in range(9)]+changed_angle;full=load_program(ART/'bilin18_l17_native_factor_readout_native_frozen.npz','cuda',torch.float64);sub=load_program(ART/'bilin18_l17_native_factor_subsets_ranked1024_frozen.npz','cuda',torch.float64);bases=load_program(ART/'bilin18_l17_factor_mixtures_bases.npz','cuda',torch.float64);rows={};replay=0.;old=np.load(ART/'bilin18_l17_native_factor_confirmation_rows.npz')
 def logits(state):return 30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
 for g,(w,t) in enumerate(features):
  ll=full['left'];rr=full['right'];lc=c.double().T@ll.T;rc=c.double().T@rr.T;raw=(w@lc)*(t@rr.T)+(w@rc)*(t@ll.T)-(u@lc)*(v@rr.T)-(u@rc)*(v@ll.T);prediction=raw/full['scale'].sqrt()@full['coef'];ids=sub['ids'].long();preds={'omitted':torch.zeros_like(prediction),'full':prediction,'inherited1024':raw[:,ids]/full['scale'].sqrt()@full['coef'][ids]}
  for width in [64,256,512]:
   V=bases['pooled'][:,:width];preds[f'pooled{width}']=(prediction@V)@V.T
  V=bases['random'][:,:256];preds['random256']=(prediction@V)@V.T
  gold=changed[g]+B(zs[g],zs[g],L,R,D)+model.blocks[17].mlp.Down_bias;truth_target=targets[g];energy=truth_target.square().sum(1);rows[f'g{g}_energy']=energy.reshape(128,4).sum(1).cpu().numpy();kls={n:[] for n in preds};tops={n:[] for n in preds}
  for off in range(0,512,cfg['logit_batch']):
   sl=slice(off,off+cfg['logit_batch']);truth=logits(gold[sl]);lp=truth.double().log_softmax(-1);prob=lp.exp()
   for name,pred in preds.items():
    state=gold[sl]+(pred[sl]-truth_target[sl]).float();out=logits(state);kl=(prob*(lp-out.double().log_softmax(-1))).sum(1);assert float(kl.min())>-1e-8;kls[name].append(kl);tops[name].append((out.argmax(-1)==truth.argmax(-1)).double())
  for name,pred in preds.items():
   rows[f'{name}_g{g}_error']=(pred-truth_target).square().sum(1).reshape(128,4).sum(1).cpu().numpy();rows[f'{name}_g{g}_kl']=torch.cat(kls[name]).reshape(128,4).sum(1).cpu().numpy();rows[f'{name}_g{g}_top1']=torch.cat(tops[name]).reshape(128,4).sum(1).cpu().numpy()
  for name,prior in [('full','full_inherited'),('inherited1024','inherited1024')]:
   before=old[f'{prior}_g{g}_error'];after=rows[f'{name}_g{g}_error'];replay=max(replay,float(np.linalg.norm(after-before)/np.linalg.norm(before)))
  print('group',g,'complete',flush=True)
 assert replay<1e-8
 boot=np.random.default_rng(cfg['bootstrap_seed']).integers(128,size=(2000,128));summary={}
 for family,groups in [('actual',range(9)),('angle',range(9,18))]:
  energy=sum(rows[f'g{g}_energy'] for g in groups);baseline=sum(rows[f'omitted_g{g}_kl'] for g in groups);summary[family]={}
  for name in preds:
   error=sum(rows[f'{name}_g{g}_error'] for g in groups);kl=sum(rows[f'{name}_g{g}_kl'] for g in groups);top=sum(rows[f'{name}_g{g}_top1'] for g in groups);summary[family][name]={'response_fidelity':float(1-error.sum()/energy.sum()),'mean_kl':float(kl.sum()/(128*4*9)),'kl_recovery':float(1-kl.sum()/baseline.sum()),'kl_recovery95':np.quantile(1-kl[boot].sum(1)/baseline[boot].sum(1),[.025,.975]).tolist(),'top1_agreement':float(top.sum()/(128*4*9))}
 gates={'full_kl80':all(v['full']['kl_recovery']>=.8 for v in summary.values()),'pooled256_kl80':all(v['pooled256']['kl_recovery']>=.8 for v in summary.values()),'pooled256_behavior_bonus10':all(v['pooled256']['kl_recovery']-v['pooled256']['response_fidelity']>=.1 for v in summary.values())}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'previous_error_replay_relative':replay,'gates':gates},indent=2)+'\n');print('FINISHED',summary,gates,flush=True)
if __name__=='__main__':main()
