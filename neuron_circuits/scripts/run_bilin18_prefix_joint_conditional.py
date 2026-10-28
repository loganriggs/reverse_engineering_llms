"""H615 matched direction budget: joint versus wording-conditioned spans."""
import json,time,hashlib,itertools
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_joint_conditional'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_prefix_attention_factors_captures.pt';d=torch.load(src,weights_only=False);mp=ART/'bilin18_prefix_boundary_control_manifest.json';rows=json.loads(mp.read_text());ex={r['id']:r for r in rows};idx={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows};cfg=dict(experiment='H615',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='Fit token-balanced output PCA on item0-3 in each wording. Compare one joint rankB basis to two separate rankB/2 bases, B=8/16/32/64. Joint fit gives equal total training weight to each wording. Evaluate items4-7 separately in each wording; both wording families now represented in fitting. Include single-wording rank32 transfer diagnostics and random64. No downstream objective fitting.',scope='Equal stored vector count, not equal active rank: joint usesB per token, conditional usesB/2 plus known wording label. No learned router, target-free coefficients or unique semantic coordinates. Adaptive familiar panel; evaluation item IDs excluded from both fits.',seed=615)
 headpath=ART/'bilin18_prefix_head_transfer_head_vectors.pt';hd=torch.load(headpath,weights_only=False);heads=hd['heads'];cfg['head_vectors_sha256']=hashlib.sha256(headpath.read_bytes()).hexdigest();(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 att=model.blocks[5].attn;bases={};singular={};train={};matrices={}
 for w in [0,1]:
  kk=[k for k in heads if ex[k]['wording']==w and ex[k]['item']<4];train[w]=kk;X=torch.cat([heads[k].double().sum(0) for k in kk]);X=X/X.norm(dim=-1,keepdim=True).clamp_min(1e-12);matrices[w]=X;_,sv,Vh=torch.linalg.svd(X,full_matrices=False);bases[f'word{w}']=Vh[:32].float();singular[f'word{w}']=sv
 joint=torch.cat([x/(len(x)**.5) for x in matrices.values()]);_,sv,Vh=torch.linalg.svd(joint,full_matrices=False);bases['joint']=Vh[:64].float();singular['joint']=sv;gen=torch.Generator().manual_seed(615);bases['random']=torch.linalg.qr(torch.randn(1152,64,generator=gen,dtype=torch.float64)).Q.T.float();budgets=[8,16,32,64];angles={}
 for r in [4,8,16,32]:
  cosines=torch.linalg.svdvals(bases['word0'][:r].double()@bases['word1'][:r].double().T).clamp(0,1);angles[r]=dict(cosines=cosines.tolist(),mean_squared_cosine=float(cosines.square().mean()))
 locked=ART/f'{STEM}_locked_bases.pt';torch.save(dict(bases=bases,singular_values=singular,discovery_ids=train,discovery_rows={w:len(x) for w,x in matrices.items()},budgets=budgets,overlap=angles),locked);lockedhash=hashlib.sha256(locked.read_bytes()).hexdigest();sum_errors=[float((heads[k].sum(0)-d['components'][k]['local']).norm()/d['components'][k]['local'].norm()) for k in heads];print('LOCKED',angles,flush=True)

 jobs=defaultdict(list)
 for k in heads:jobs[len(ex[k]['ids'])].append(k)
 def forward(kk,delta=None):
  cap={};hs=[]
  def target(m,args):cap['target']=args[0][:,-1].cpu()
  hs.append(model.blocks[11].mlp.register_forward_pre_hook(target))
  if delta is not None:
   def edit(m,args,out):y,bus=out;y=y.clone();y[:,4:]-=delta.to(y);return y,bus
   hs.append(att.register_forward_hook(edit))
  try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in kk],device='cuda'))
  finally:
   for h in hs:h.remove()
  return z.cpu(),cap['target']
 modes=['full','native','old32','new32','random64']+[f'{name}{budget}' for name in ['joint','conditional'] for budget in budgets];result=dict(config=cfg,overlap=angles,interventions=[],summary={},status='running');refs={};native_targets={}
 def vector(k,mode):
  hs=heads[k];full=hs.sum(0)
  if mode=='full':return full
  if mode=='native':return torch.zeros_like(full)
  if mode=='old32':B=bases['word0']
  elif mode=='new32':B=bases['word1']
  elif mode=='random64':B=bases['random']
  elif mode.startswith('joint'):B=bases['joint'][:int(mode[5:])]
  else:B=bases[f"word{ex[k]['wording']}"][:int(mode[11:])//2]
  return (full@B.T)@B
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save()
 # Native target baseline before full reference. This is read-only; all selection is already locked.
 replay=[]
 for kk in jobs.values():
  z,t=forward(kk)
  for i,k in enumerate(kk):native_targets[k]=t[i];replay.append(float((z[i]-d['native'][k]).abs().max()))
 for mode in modes:
  for kk in jobs.values():
   for st in range(0,len(kk),16):
    bb=kk[st:st+16];delta=torch.stack([vector(k,mode) for k in bb]);z,t=forward(bb,delta)
    for i,k in enumerate(bb):
     e=ex[k];zz=z[i];tt=t[i]
     if mode=='full':refs[k]=dict(logits=zz,target=tt)
     rp=refs[k]['logits'].double().log_softmax(0);lp=zz.double().log_softmax(0);full=heads[k].sum(0);ans=e['groups'][e['variant']][0];baseI=d['targets'][k];targetI=baseI+tt-native_targets[k];den=(refs[k]['target']-native_targets[k]).norm();result['interventions'].append(dict(id=k,wording=e['wording'],item=e['item'],split=('discovery' if e['item']<4 else 'holdout')+str(e['wording']),mode=mode,write_relative_error=float((vector(k,mode)-full).norm()/full.norm()),target_relative_error=float((tt-refs[k]['target']).norm()/den),target_interaction_ratio=float(targetI.norm()/baseI.norm()),reference_kl=float((rp.exp()*(rp-lp)).sum()),margin_error=measure(zz,e['groups'])['margin']-measure(refs[k]['logits'],e['groups'])['margin'],target_ce_error=float(rp[ans]-lp[ans]),full_vocab_correct=measure(zz,e['groups'])['top1'][e['variant']]))
  save();print('COMPLETE',mode,flush=True)
 for split in ['discovery0','discovery1','holdout0','holdout1']:
  cc=[c for c in result['interventions'] if c['split']==split];base=[c for c in cc if c['mode']=='native'];kd=np.mean([c['reference_kl'] for c in base]);md=np.mean([abs(c['margin_error']) for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
  for mode in modes:
   vv=[c for c in cc if c['mode']==mode];result['summary'][f'{split}_{mode}']=dict(n=len(vv),write_relative_error=float(np.mean([c['write_relative_error'] for c in vv])),target_relative_error=float(np.mean([c['target_relative_error'] for c in vv])),target_interaction_ratio=float(np.mean([c['target_interaction_ratio'] for c in vv])),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd),full_vocab_correct=float(np.mean([c['full_vocab_correct'] for c in vv])))
 result['audits']=dict(source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['manifest_sha256']==hashlib.sha256(mp.read_bytes()).hexdigest(),bases_unchanged=lockedhash==hashlib.sha256(locked.read_bytes()).hexdigest(),head_source_unchanged=cfg['head_vectors_sha256']==hashlib.sha256(headpath.read_bytes()).hexdigest(),orthonormality_max_abs=max(float((b@b.T-torch.eye(len(b))).abs().max()) for b in bases.values()),head_sum_max_relative_error=max(sum_errors),native_replay_max_logit_error=max(replay),discovery_endpoints=sum(map(len,train.values())),interventions=len(result['interventions']),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_relative_error','reference_kl','write_relative_error']));result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
