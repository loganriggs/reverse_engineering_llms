"""H610 aligned-body MLP0 interaction removals after boundary control."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_body_interactions'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_prefix_boundary_control_manifest.json';rows=json.loads(src.read_text());ex={r['id']:r for r in rows};index={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows};cfg=dict(experiment='H610',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='H609 matched-body-token prompts. At aligned body positions compute MLP0 full four-condition interaction and exact float64 decomposition into bilinear cross term and input-nonadditivity remainder. Subtract each component on combined prefix/lowercase run at first body, final body, other body, or all body positions. Prefix-only tokens remain untouched. Per-position norm-matched seeded random control for each component/scope; identity. Read native MLP11 normalized-input interaction and Attention9 final-write interaction.',scope='All conditions/scopes declared before outcomes. Conditional intervention changes one factorial corner. Component/site effects not additive shares. Random controls have matching per-position norm profile but only one seeded direction per relative body position.',seed=610)
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 m=model.blocks[0].mlp;features={};native={};batches=defaultdict(list)
 for k,e in ex.items():batches[len(e['ids'])].append(k)
 def forward(kk,delta=None):
  cap={};hs=[]
  def p0(mod,args):cap['input0']=args[0].cpu()
  def m0(mod,args,out):cap['write0']=out.cpu()
  def p11(mod,args):cap['target']=args[0][:,-1].cpu()
  def a9(mod,args,out):cap['attn9']=out[0][:,-1].cpu()
  hs=[m.register_forward_pre_hook(p0),m.register_forward_hook(m0),model.blocks[11].mlp.register_forward_pre_hook(p11),model.blocks[9].attn.register_forward_hook(a9)]
  if delta is not None:
   def edit(mod,args,out):y=out.clone();y[:,4:]-=delta.to(y);return y
   hs.append(m.register_forward_hook(edit))
  try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in kk],device='cuda'))
  finally:
   for h in hs:h.remove()
  return z.cpu(),cap
 for kk in batches.values():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];z,c=forward(bb)
   for i,k in enumerate(bb):features[k]={n:v[i].clone() for n,v in c.items()};native[k]=z[i]
 components={};targets={};geometry=[];identities=[];native_errors=[];combined=[];aligned=0;m.double()
 for w in [0,1]:
  for item in range(8):
   for v in [0,1]:
    ids=[index[w,item,v,p,lo] for p,lo in [(0,0),(0,1),(1,0),(1,1)]];assert ex[ids[2]]['ids'][4:]==ex[ids[0]]['ids'] and ex[ids[3]]['ids'][4:]==ex[ids[1]]['ids'];aligned+=2;k=ids[3];combined.append(k);fs=[features[ii] for ii in ids];z00,z01,z10,z11=[(f['input0'][4:] if i>1 else f['input0']).cuda().double() for i,f in enumerate(fs)];P=z10-z00;C=z01-z00;raw=m.Left(P)*m.Right(C)+m.Left(C)*m.Right(P);cross=m.Down(raw)-m.Down(torch.zeros_like(raw));rem=m(z11)-m(z00+P+C);full=m(z11)-m(z10)-m(z01)+m(z00);nativefull=fs[3]['write0'][4:]-fs[2]['write0'][4:]-fs[1]['write0']+fs[0]['write0'];identities.append(float((cross+rem-full).norm()/full.norm()));native_errors.append(float((full.cpu()-nativefull.double()).norm()/full.norm()));components[k]=dict(full=full.cpu().float(),cross=cross.cpu().float(),remainder=rem.cpu().float());targets[k]={n:fs[3][n]-fs[2][n]-fs[1][n]+fs[0][n] for n in ['target','attn9']}
    geometry.append(dict(id=k,wording=w,body_length=len(full),norms={name:dict(first=float(x[0].norm()),final=float(x[-1].norm()),other=float(x[1:-1].norm()),all=float(x.norm())) for name,x in [('full',full),('cross',cross),('remainder',rem)]}))
 m.float();gen=torch.Generator().manual_seed(610);maxlen=max(len(components[k]['full']) for k in combined);random=torch.randn(maxlen,1152,generator=gen);random/=random.norm(dim=-1,keepdim=True);jobs=defaultdict(list)
 for k in combined:jobs[len(ex[k]['ids'])].append(k)
 modes=[('self','self','self')]+[(c,scope,mode) for c in ['full','cross','remainder'] for scope in ['first','final','other','all'] for mode in ['interaction','random']];result=dict(config=cfg,geometry=geometry,interventions=[],summary={},status='running');replay=[];norm_errors=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save()
 for component,scope,mode in modes:
  for kk in jobs.values():
   for st in range(0,len(kk),16):
    bb=kk[st:st+16];delta=None
    if mode!='self':
     arr=[]
     for k in bb:
      x=components[k][component];mask=torch.zeros(len(x),1)
      if scope=='first':mask[0]=1
      elif scope=='final':mask[-1]=1
      elif scope=='other':mask[1:-1]=1
      else:mask[:]=1
      chosen=x if mode=='interaction' else random[:len(x)]*x.norm(dim=-1,keepdim=True);chosen=chosen*mask;norm_errors.append(float((chosen.norm(dim=-1)-(x*mask).norm(dim=-1)).abs().max()));arr.append(chosen)
     delta=torch.stack(arr)
    z,c=forward(bb,delta)
    for i,k in enumerate(bb):
     e=ex[k];I=targets[k]['target'];shift=c['target'][i]-features[k]['target'];aI=targets[k]['attn9'];ashift=c['attn9'][i]-features[k]['attn9'];zz=z[i];rp=native[k].double().log_softmax(0);lp=zz.double().log_softmax(0);ans=e['groups'][e['variant']][0]
     if mode=='self':replay.append(float((zz-native[k]).abs().max()))
     result['interventions'].append(dict(id=k,wording=e['wording'],component=component,scope=scope,mode=mode,target_interaction_ratio=float((I+shift).norm()/I.norm()),interaction_alignment_removed=float(-(shift@I)/I.square().sum()),attention9_interaction_ratio=float((aI+ashift).norm()/aI.norm()),reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce_delta=float(rp[ans]-lp[ans]),margin_delta=measure(zz,e['groups'])['margin']-measure(native[k],e['groups'])['margin'],full_vocab_correct=measure(zz,e['groups'])['top1'][e['variant']]))
  save();print('COMPLETE',component,scope,mode,flush=True)
 for w in [0,1]:
  for component,scope,mode in modes[1:]:
   vv=[c for c in result['interventions'] if c['wording']==w and c['component']==component and c['scope']==scope and c['mode']==mode];result['summary'][f'{w}_{component}_{scope}_{mode}']={n:float(np.mean([c[n] for c in vv])) for n in ['target_interaction_ratio','interaction_alignment_removed','attention9_interaction_ratio','reference_kl','target_ce_delta','full_vocab_correct']}
 result['audits']=dict(runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),aligned_body_comparisons=aligned,native_prompts=len(ex),interventions=len(result['interventions']),identity_replay_max_logit_error=max(replay),float64_component_sum_max_relative_error=max(identities),native_float32_interaction_replay_max_relative_error=max(native_errors),random_per_position_norm_max_abs_error=max(norm_errors),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_interaction_ratio','attention9_interaction_ratio','reference_kl','target_ce_delta']));torch.save(dict(features=features,native=native,targets=targets,components=components,random=random),ART/f'{STEM}_captures.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
