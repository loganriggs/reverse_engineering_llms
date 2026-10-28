"""H327 fresh confirmation of frozen native-factor predictors."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_native_factor_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());folder=ART/'bilin18_l17_kernel_confirmation_frozen';assert cfg['layer']==17
 for n,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==digest
 assert hashlib.sha256((ROOT/cfg['corpus']).read_bytes()).hexdigest()==cfg['corpus_sha256']
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda');common=np.load(folder/'common.npz');c=torch.tensor(common['core'],device='cuda');q=torch.tensor(common['partner'],device='cuda');mu=torch.tensor(common['mean'],device='cuda');ux=torch.tensor(common['ux'],device='cuda');vx=torch.tensor(common['vx'],device='cuda')
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,));gen=torch.Generator(device='cuda').manual_seed(cfg['fresh_payload_seed']);zs=[]
 for h in range(9):
  source=torch.linalg.qr(model.blocks[17].attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;pay=angle_matched_payload(pre,heads[:,h],source,torch.randn(len(pre),128,generator=gen,device='cuda'));zs.append(F.rms_norm(pre-pay,(1152,)))
 zs=[F.rms_norm(pre-heads[:,h],(1152,)) for h in range(9)]+zs
 def coords(v):
  w=v.double()-mu.double();u=w@c.double();return u,w-u@c.double().T-(w@q.double())@q.double().T
 u,v=coords(x);features=[];targets=[];rows={}
 for target,z in enumerate(zs):
  w,t=coords(z);features.append((w,t));y=torch.cat([quadratic_terms(x[i:i+256],z[i:i+256],mu,c,L,R,D,p)[:,1]-cross_response(x[i:i+256],z[i:i+256],mu,c,q,L,R,D,p) for i in range(0,512,256)]).double();targets.append(y);rows[f't{target}_energy']=y.square().sum(-1).reshape(128,4).sum(1).cpu().numpy()
 full=load_program(ART/'bilin18_l17_native_factor_readout_native_frozen.npz','cuda',torch.float64);rotated=load_program(ART/'bilin18_l17_native_factor_readout_rotated_frozen.npz','cuda',torch.float64);subs={w:load_program(ART/f'bilin18_l17_native_factor_subsets_ranked{w}_frozen.npz','cuda',torch.float64) for w in [1024,2048]};rows={};identity=0.
 def feature(desc,ll,rr):
  aa,bb,cc,dd=desc;lc=c.double().T@ll.T;rc=c.double().T@rr.T
  return (cc@lc)*(dd@rr.T)+(cc@rc)*(dd@ll.T)-(aa@lc)*(bb@rr.T)-(aa@rc)*(bb@ll.T)
 for g,(w,t) in enumerate(features):
  desc=(u,v,w,t);raw=feature(desc,full['left'],full['right']);rot=feature(desc,rotated['left'],rotated['right']);preds={'full_native':raw@D.double().T,'full_inherited':raw/full['scale'].sqrt()@full['coef'],'full_rotated':rot/rotated['scale'].sqrt()@rotated['coef']}
  for width,sub in subs.items():
   ids=sub['ids'].long();pp=raw[:,ids];preds[f'native{width}']=pp@D.double().T[ids];preds[f'inherited{width}']=pp/full['scale'].sqrt()@full['coef'][ids];preds[f'refit{width}']=pp/sub['scale'].sqrt()@sub['coef'];direct=feature(desc,sub['left'],sub['right']);identity=max(identity,float((direct-pp).norm()/pp.norm()))
  rows[f'g{g}_energy']=targets[g].square().sum(1).reshape(128,4).sum(1).cpu().numpy()
  for name,pred in preds.items():rows[f'{name}_g{g}_error']=(pred-targets[g]).square().sum(1).reshape(128,4).sum(1).cpu().numpy()
 assert identity<1e-10
 boot=np.random.default_rng(cfg['bootstrap_seed']).integers(128,size=(2000,128));summary={};contrasts={}
 for family,groups in [('actual',range(9)),('angle',range(9,18))]:
  energy=sum(rows[f'g{g}_energy'] for g in groups);errors={name:sum(rows[f'{name}_g{g}_error'] for g in groups) for name in preds};summary[family]={}
  for name,err in errors.items():summary[family][name]={'fidelity':float(1-err.sum()/energy.sum()),'paired95':np.quantile(1-err[boot].sum(1)/energy[boot].sum(1),[.025,.975]).tolist()}
  contrasts[family]={}
  for label,better,worse in [('native_alignment','full_inherited','full_rotated'),('inherited1024_over_refit','inherited1024','refit1024'),('full_learned_over_native','full_inherited','full_native')]:
   gain=errors[worse]-errors[better];contrasts[family][label]={'gain':float(gain.sum()/energy.sum()),'paired95':np.quantile(gain[boot].sum(1)/energy[boot].sum(1),[.025,.975]).tolist()}
 gates={'full85':all(v['full_inherited']['fidelity']>=.85 for v in summary.values()),'alignment80':all(v['native_alignment']['gain']>=.8 and v['native_alignment']['paired95'][0]>0 for v in contrasts.values()),'inherited_refit10':all(v['inherited1024_over_refit']['gain']>=.1 and v['inherited1024_over_refit']['paired95'][0]>0 for v in contrasts.values())}
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'summary':summary,'contrasts':contrasts,'subset_feature_replay_relative':identity,'gates':gates},indent=2)+'\n');print('FINISHED',summary,contrasts,gates,flush=True)
if __name__=='__main__':main()
