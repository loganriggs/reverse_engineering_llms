"""H614 frozen low-rank output directions for combined contextual write."""
import json,time,hashlib,itertools
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_output_subspace'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_prefix_attention_factors_captures.pt';d=torch.load(src,weights_only=False);mp=ART/'bilin18_prefix_boundary_control_manifest.json';rows=json.loads(mp.read_text());ex={r['id']:r for r in rows};idx={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows};cfg=dict(experiment='H614',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='Frozen combined-local-write output PCA on old wording item0-3 only. Compare uncentered raw-row PCA, per-row unit-norm PCA, and one seeded orthonormal random basis at ranks1/4/8/16/32. Same item/wording splits as H613, plus locked4-head baseline. Coefficients are projections of observed per-position interaction write; no target-free predictor or global compression claim. Evaluate full local-write removal fidelity, not just target shrinkage.',scope='Adaptive on familiar panel; old discovery eight endpoints, old holdout eight, new wording sixteen. Per-row weighting prevents large native write norms from dominating but is not a causal objective. No rank or basis selection on evaluation.',seed=614)
 headpath=ART/'bilin18_prefix_head_transfer_head_vectors.pt';hd=torch.load(headpath,weights_only=False);heads=hd['heads'];cfg['head_vectors_sha256']=hashlib.sha256(headpath.read_bytes()).hexdigest();(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 att=model.blocks[5].attn;train=[k for k in heads if ex[k]['wording']==0 and ex[k]['item']<4];X=torch.cat([heads[k].double().sum(0) for k in train]);norms=X.norm(dim=-1,keepdim=True);normalized=X/norms.clamp_min(1e-12);bases={};singular={}
 for name,mat in [('raw',X),('balanced',normalized)]:
  _,sv,Vh=torch.linalg.svd(mat,full_matrices=False);assert len(Vh)>=32;bases[name]=Vh[:32].float();singular[name]=sv
 gen=torch.Generator().manual_seed(614);bases['random']=torch.linalg.qr(torch.randn(1152,32,generator=gen,dtype=torch.float64)).Q.T.float();ranks=[1,4,8,16,32];selection=hd['selection'];locked=ART/f'{STEM}_locked_bases.pt';torch.save(dict(bases=bases,singular_values=singular,discovery_ids=train,discovery_rows=len(X),ranks=ranks,head_selection=selection),locked);lockedhash=hashlib.sha256(locked.read_bytes()).hexdigest();print('LOCKED',len(train),len(X),'training rows',flush=True);sum_errors=[float((heads[k].sum(0)-d['components'][k]['local']).norm()/d['components'][k]['local'].norm()) for k in heads]
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
 modes=['full','native','head4']+[f'{name}{rank}' for name in ['raw','balanced','random'] for rank in ranks];result=dict(config=cfg,training_energy={name:{rank:float(sv[:rank].square().sum()/sv.square().sum()) for rank in ranks} for name,sv in singular.items()},interventions=[],summary={},status='running');refs={};native_targets={}
 def vector(k,mode):
  hs=heads[k];full=hs.sum(0)
  if mode=='full':return full
  if mode=='native':return torch.zeros_like(full)
  if mode=='head4':return hs[selection[4]].sum(0)
  name=next(n for n in bases if mode.startswith(n));rank=int(mode[len(name):]);B=bases[name][:rank];return (full@B.T)@B
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
     rp=refs[k]['logits'].double().log_softmax(0);lp=zz.double().log_softmax(0);full=heads[k].sum(0);ans=e['groups'][e['variant']][0];baseI=d['targets'][k];targetI=baseI+tt-native_targets[k];den=(refs[k]['target']-native_targets[k]).norm();result['interventions'].append(dict(id=k,wording=e['wording'],item=e['item'],split='discovery' if e['wording']==0 and e['item']<4 else 'same_wording_holdout' if e['wording']==0 else 'new_wording',mode=mode,write_relative_error=float((vector(k,mode)-full).norm()/full.norm()),target_relative_error=float((tt-refs[k]['target']).norm()/den),target_interaction_ratio=float(targetI.norm()/baseI.norm()),reference_kl=float((rp.exp()*(rp-lp)).sum()),margin_error=measure(zz,e['groups'])['margin']-measure(refs[k]['logits'],e['groups'])['margin'],target_ce_error=float(rp[ans]-lp[ans]),full_vocab_correct=measure(zz,e['groups'])['top1'][e['variant']]))
  save();print('COMPLETE',mode,flush=True)
 for split in ['discovery','same_wording_holdout','new_wording']:
  cc=[c for c in result['interventions'] if c['split']==split];base=[c for c in cc if c['mode']=='native'];kd=np.mean([c['reference_kl'] for c in base]);md=np.mean([abs(c['margin_error']) for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
  for mode in modes:
   vv=[c for c in cc if c['mode']==mode];result['summary'][f'{split}_{mode}']=dict(n=len(vv),write_relative_error=float(np.mean([c['write_relative_error'] for c in vv])),target_relative_error=float(np.mean([c['target_relative_error'] for c in vv])),target_interaction_ratio=float(np.mean([c['target_interaction_ratio'] for c in vv])),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd),full_vocab_correct=float(np.mean([c['full_vocab_correct'] for c in vv])))
 result['audits']=dict(source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['manifest_sha256']==hashlib.sha256(mp.read_bytes()).hexdigest(),bases_unchanged=lockedhash==hashlib.sha256(locked.read_bytes()).hexdigest(),head_source_unchanged=cfg['head_vectors_sha256']==hashlib.sha256(headpath.read_bytes()).hexdigest(),orthonormality_max_abs=max(float((b@b.T-torch.eye(32)).abs().max()) for b in bases.values()),head_sum_max_relative_error=max(sum_errors),native_replay_max_logit_error=max(replay),discovery_endpoints=len(train),interventions=len(result['interventions']),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_relative_error','reference_kl','write_relative_error']));result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
