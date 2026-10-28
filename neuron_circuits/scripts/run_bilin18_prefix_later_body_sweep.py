"""H611 aligned-body writer interaction removals through MLP11 input."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_later_body_sweep'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_prefix_boundary_control_manifest.json';rows=json.loads(src.read_text());ex={r['id']:r for r in rows};index={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows};ops=[(kind,l) for l in range(11) for kind in ['attn','mlp']]+[('attn',11)];cfg=dict(experiment='H611',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='H609 aligned-body prompts. Sweep full four-condition interaction subtraction at attention/MLP0-10 plus attention11, with first/final/other/all body scopes. Prefix-only positions untouched. Per-position norm-matched random controls and identity. All23 sites and4 scopes declared before results. Attention output intervention leaves persistent value bus untouched.',seed=611,scope='Exploratory same-panel localization. Single-corner conditional removals, not additive shares. MLP11 final input interaction and full output behavior are separate outcomes.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 features={};native={};batches=defaultdict(list);application_errors=[]
 for k,e in ex.items():batches[len(e['ids'])].append(k)
 def forward(kk,capture=False,op=None,delta=None):
  cap={};hs=[]
  def target(mod,args):cap['target']=args[0][:,-1].cpu()
  hs.append(model.blocks[11].mlp.register_forward_pre_hook(target))
  if capture:
   for kind,l in ops:
    def h(mod,args,out,kind=kind,l=l):cap[f'{kind}{l}']=(out[0] if kind=='attn' else out).cpu()
    hs.append(getattr(model.blocks[l],kind).register_forward_hook(h))
  if op is not None:
   kind,l=op
   def edit(mod,args,out):
    original=out[0] if kind=='attn' else out;y=original.clone();v=delta.to(y);y[:,4:]-=v;application_errors.append(float(((y[:,4:]-original[:,4:])+v).norm()/v.norm().clamp_min(1e-20)));return (y,out[1]) if kind=='attn' else y
   hs.append(getattr(model.blocks[l],kind).register_forward_hook(edit))
  try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in kk],device='cuda'))
  finally:
   for h in hs:h.remove()
  return z.cpu(),cap
 for kk in batches.values():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];z,c=forward(bb,capture=True)
   for i,k in enumerate(bb):features[k]={n:v[i].clone() for n,v in c.items()};native[k]=z[i]
 interactions={};targets={};combined=[];geometry=[];aligned=0
 for w in [0,1]:
  for item in range(8):
   for v in [0,1]:
    ids=[index[w,item,v,p,c] for p,c in [(0,0),(0,1),(1,0),(1,1)]];assert ex[ids[2]]['ids'][4:]==ex[ids[0]]['ids'] and ex[ids[3]]['ids'][4:]==ex[ids[1]]['ids'];aligned+=2;k=ids[3];combined.append(k);fs=[features[ii] for ii in ids];targets[k]=fs[3]['target']-fs[2]['target']-fs[1]['target']+fs[0]['target'];iv={}
    for kind,l in ops:
     n=f'{kind}{l}';iv[n]=fs[3][n][4:]-fs[2][n][4:]-fs[1][n]+fs[0][n];x=iv[n];geometry.append(dict(id=k,wording=w,operation=n,first_norm=float(x[0].norm()),final_norm=float(x[-1].norm()),other_norm=float(x[1:-1].norm()),all_norm=float(x.norm())))
    interactions[k]=iv
 gen=torch.Generator().manual_seed(611);maxlen=max(len(interactions[k]['attn0']) for k in combined);random={}
 for op in ops:
  v=torch.randn(maxlen,1152,generator=gen);random[op]=v/v.norm(dim=-1,keepdim=True)
 jobs=defaultdict(list)
 for k in combined:jobs[len(ex[k]['ids'])].append(k)
 conditions=[(None,'self','self')]+[(op,scope,mode) for op in ops for scope in ['first','final','other','all'] for mode in ['interaction','random']];result=dict(config=cfg,geometry=geometry,interventions=[],summary={},status='running');replay=[];norm_errors=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save()
 for op,scope,mode in conditions:
  for kk in jobs.values():
   for st in range(0,len(kk),16):
    bb=kk[st:st+16];delta=None
    if op is not None:
     arr=[]
     for k in bb:
      x=interactions[k][f'{op[0]}{op[1]}'];mask=torch.zeros(len(x),1)
      if scope=='first':mask[0]=1
      elif scope=='final':mask[-1]=1
      elif scope=='other':mask[1:-1]=1
      else:mask[:]=1
      v=x if mode=='interaction' else random[op][:len(x)]*x.norm(dim=-1,keepdim=True);v=v*mask;norm_errors.append(float((v.norm(dim=-1)-(x*mask).norm(dim=-1)).abs().max()));arr.append(v)
     delta=torch.stack(arr)
    z,c=forward(bb,op=op,delta=delta)
    for i,k in enumerate(bb):
     e=ex[k];I=targets[k];shift=c['target'][i]-features[k]['target'];zz=z[i];rp=native[k].double().log_softmax(0);lp=zz.double().log_softmax(0);ans=e['groups'][e['variant']][0]
     if op is None:replay.append(float((zz-native[k]).abs().max()))
     result['interventions'].append(dict(id=k,wording=e['wording'],operation='self' if op is None else f'{op[0]}{op[1]}',scope=scope,mode=mode,target_interaction_ratio=float((I+shift).norm()/I.norm()),interaction_alignment_removed=float(-(shift@I)/I.square().sum()),reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce_delta=float(rp[ans]-lp[ans]),margin_delta=measure(zz,e['groups'])['margin']-measure(native[k],e['groups'])['margin'],full_vocab_correct=measure(zz,e['groups'])['top1'][e['variant']]))
  if mode=='random' and scope=='all':save();print('COMPLETE',op,flush=True)
 for w in [0,1]:
  for op,scope,mode in conditions[1:]:
   n=f'{op[0]}{op[1]}';vv=[c for c in result['interventions'] if c['wording']==w and c['operation']==n and c['scope']==scope and c['mode']==mode];result['summary'][f'{w}_{n}_{scope}_{mode}']={n:float(np.mean([c[n] for c in vv])) for n in ['target_interaction_ratio','interaction_alignment_removed','reference_kl','target_ce_delta','full_vocab_correct']}
 result['audits']=dict(runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),aligned_body_comparisons=aligned,native_prompts=len(ex),interventions=len(result['interventions']),identity_replay_max_logit_error=max(replay),application_max_relative_error=max(application_errors),random_per_position_norm_max_abs_error=max(norm_errors),attention11_nonfinal_target_ratio_max_error=max(abs(c['target_interaction_ratio']-1) for c in result['interventions'] if c['operation']=='attn11' and c['scope'] in ['first','other']),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_interaction_ratio','reference_kl','target_ce_delta']));torch.save(dict(targets=targets,interactions=interactions,native=native,random=random),ART/f'{STEM}_captures.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
