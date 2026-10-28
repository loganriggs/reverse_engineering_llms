"""H613 exact contextual head decomposition and locked small-subset transfer."""
import json,time,hashlib,itertools
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_head_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_prefix_attention_factors_captures.pt';d=torch.load(src,weights_only=False);mp=ART/'bilin18_prefix_boundary_control_manifest.json';rows=json.loads(mp.read_text());ex={r['id']:r for r in rows};idx={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows};cfg=dict(experiment='H613',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='Decompose Attention5 local-value four-condition body-write interaction into all9 heads. Test single-head removal, full-minus-one-head removal, random matched per-position single-head norms. Choose subsets of size1/2/3/4 by exhaustive minimization of mean relative squared local-write error on old wording item0-3 only, freeze before downstream evaluation. Compare each mode with full local-interaction removal, not maximum target shrinkage.',scope='Adaptive on previously explored panel. Discovery eight endpoints; held-out same-wording eight and new-wording sixteen endpoints. New wording includes both familiar and different items. No fresh vocabulary or independent benchmark claim. Head subsets still require observed four-condition head interactions.',seed=613)
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 att=model.blocks[5].attn;H=model.config.n_head;D=model.config.n_embd//H;weight=1-float(att.lamb);heads={};sum_errors=[]
 for w in [0,1]:
  for item in range(8):
   for v in [0,1]:
    ids=[idx[w,item,v,p,c] for p,c in [(0,0),(0,1),(1,0),(1,1)]];outputs=[]
    for k in ids:
     a=d['aligned'][k];R=a['R'].cuda().double();V=a['local'].cuda().double();q=torch.einsum('hqk,khd->hqd',R,V);out=torch.stack([F.linear(q[h],att.c_proj.weight[:,h*D:(h+1)*D].double())*weight for h in range(H)]).cpu();outputs.append(out)
    hs=outputs[3]-outputs[2]-outputs[1]+outputs[0];k=ids[3];heads[k]=hs.float();ref=d['components'][k]['local'].double();sum_errors.append(float((hs.sum(0)-ref).norm()/ref.norm()))
 assert max(sum_errors)<1e-6
 train=[k for k in heads if ex[k]['wording']==0 and ex[k]['item']<4];G=torch.zeros(H,H,dtype=torch.float64)
 for k in train:
  X=heads[k].double().flatten(1);G+=(X@X.T)/X.sum(0).square().sum()/len(train)
 selection={};scores={}
 for size in [1,2,3,4]:
  options=[]
  for combo in itertools.combinations(range(H),size):
   omitted=torch.ones(H,dtype=torch.float64);omitted[list(combo)]=0;options.append((float(omitted@G@omitted),combo))
  score,combo=min(options);selection[size]=list(combo);scores[size]=score
 locked=ART/f'{STEM}_locked_selection.json';locked.write_text(json.dumps(dict(selection=selection,discovery_ids=train,mean_relative_squared_write_errors=scores),indent=2));lockedhash=hashlib.sha256(locked.read_bytes()).hexdigest();print('LOCKED',selection,scores,flush=True)
 maxlen=max(x.shape[1] for x in heads.values());gen=torch.Generator().manual_seed(613);random=torch.randn(H,maxlen,1152,generator=gen);random/=random.norm(dim=-1,keepdim=True);jobs=defaultdict(list)
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
 modes=['full','native']+[f'single{h}' for h in range(H)]+[f'omit{h}' for h in range(H)]+[f'random{h}' for h in range(H)]+[f'subset{s}' for s in selection];result=dict(config=cfg,selection=selection,selection_scores=scores,interventions=[],summary={},status='running');refs={};native_targets={};raw=[]
 def vector(k,mode):
  hs=heads[k];full=hs.sum(0)
  if mode=='full':return full
  if mode=='native':return torch.zeros_like(full)
  if mode.startswith('single'):return hs[int(mode[6:])]
  if mode.startswith('omit'):return full-hs[int(mode[4:])]
  if mode.startswith('random'):
   h=int(mode[6:]);return random[h,:len(full)]*hs[h].norm(dim=-1,keepdim=True)
  return hs[selection[int(mode[6:])]].sum(0)
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
 result['audits']=dict(source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['manifest_sha256']==hashlib.sha256(mp.read_bytes()).hexdigest(),selection_unchanged=lockedhash==hashlib.sha256(locked.read_bytes()).hexdigest(),head_sum_max_relative_error=max(sum_errors),native_replay_max_logit_error=max(replay),discovery_endpoints=len(train),interventions=len(result['interventions']),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_relative_error','reference_kl','write_relative_error']));torch.save(dict(heads=heads,selection=selection),ART/f'{STEM}_head_vectors.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
