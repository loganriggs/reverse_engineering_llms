"""H654: held-out functional test of task versus instruction-control context shifts."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_operation_control_effects'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_operation_control_panel_manifest.json','captures':ART/'bilin18_operation_control_panel_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};allrows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['captures'],weights_only=False);Y=torch.cat([cap['normalized_mlp_contexts'][l].double() for l in range(10,18)],-1);lookup={(r['family'],r['control'],r['item'],r['form']):i for i,r in enumerate(allrows)};families=sorted({r['family'] for r in allrows});means={};rows=[];source=[];target=[]
 for fam in families:
  for axis,pairs in [('verb',[(0,1),(2,3)]),('suffix',[(0,2),(1,3)])]:
   for setting,(a,b) in enumerate(pairs):
    for ctrl in ['task','copy','constant']:
     means[fam,ctrl,axis,setting]=torch.stack([Y[lookup[fam,ctrl,i,b]]-Y[lookup[fam,ctrl,i,a]] for i in range(4)]).mean(0)
    for i in range(4,8):
     src=lookup[fam,'task',i,a];dst=lookup[fam,'task',i,b];rows.append(dict(id=f'{fam}-{axis}-{setting}-{i}',family=fam,axis=axis,setting=setting,item=i,ids=allrows[src]['ids']));source.append(src);target.append(dst)
 assert len(rows)==96
 gen=torch.Generator().manual_seed(654);Q=torch.linalg.qr(torch.randn(1152,3,generator=gen,dtype=torch.float64)).Q.T;locked=ART/f'{STEM}_locked.pt';torch.save(dict(means=means,probes=Q,source=source,target=target),locked);paths['locked']=locked;hashes['locked']=hashlib.sha256(locked.read_bytes()).hexdigest();cfg=dict(experiment='H654',hashes=hashes,design='96 heldout task contrasts, four discovery items per task/control/axis/setting. Six modes native/exact/own/copy/constant/other_tasks. Three seeded orthonormal unit probes injected directly after block9. Receiver attention, normalization and readout fixed; substitute MLP context shift throughout10-17. No output fitting.',limitations='Direct block9 probes differ from prior block5-propagated probes. Local derivative counterfactual, not finite behavioral change. Known task identity, same templates, small heldout item set; all task top1 zero on native panel.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,r in enumerate(rows):buckets[len(r['ids'])].append(i)
 raw={l:{} for l in range(10,18)};post={l:{} for l in range(10,18)};attnevals=0
 for l in range(10,18):
  block=model.blocks[l]
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];ss=[source[i] for i in bb];x=torch.stack([cap['caches'][l][i] for i in ss]).cuda();e=torch.stack([cap['x0'][i] for i in ss]).cuda();bus=torch.stack([cap['bus'][i] for i in ss]).cuda();r=block.lambdas[0]*x+block.lambdas[1]*e;a,_=block.attn(F.rms_norm(r,(1152,)),bus);attnevals+=len(bb)
    for k,i in enumerate(bb):raw[l][i]=r[k].cpu();post[l][i]=(r[k,-1]+a[k,-1]).cpu()
 factors={};context_error=0.
 for l in range(10,18):
  x=torch.stack([post[l][i] for i in range(len(rows))]).double().cuda();r=(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();m=model.blocks[l].mlp;factors[l]=(x,r,x/r,m.Left.weight.double(),m.Right.weight.double(),m.Down.weight.double());context_error=max(context_error,float((x/r-cap['normalized_mlp_contexts'][l][source].double().cuda()).abs().max()))
 jvps=0
 def action(l,d,whole=False):
  nonlocal jvps
  result=torch.empty_like(d)
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];ss=[source[i] for i in bb];bus=torch.stack([cap['bus'][i] for i in ss]).cuda();x=torch.stack([cap['caches'][l][s] if whole else raw[l][i] for s,i in zip(ss,bb)]).cuda();prefix=x[:,:-1];last=x[:,-1];mod=model.blocks[l]
    if whole:
     x0=torch.stack([cap['x0'][i] for i in ss]).cuda()
     def fn(tail):return mod(torch.cat([prefix,tail[:,None]],1),bus,x0)[0][:,-1]
    else:
     def fn(tail):return mod.attn(F.rms_norm(torch.cat([prefix,tail[:,None]],1),(1152,)),bus)[0][:,-1]
    _,out=torch.func.jvp(fn,(last,),(d[bb].cuda().float(),));result[bb]=out.cpu().double();jvps+=1
  return result
 modes=['native','exact','own','copy','constant','other_tasks'];shifts={m:[] for m in modes}
 for i,row in enumerate(rows):
  fam,axis,setting=row['family'],row['axis'],row['setting']
  for m in modes:
   if m=='native':v=torch.zeros(9216,dtype=torch.float64)
   elif m=='exact':v=Y[target[i]]-Y[source[i]]
   elif m=='other_tasks':v=torch.stack([means[f,'task',axis,setting] for f in families if f!=fam]).mean(0)
   else:v=means[fam,'task' if m=='own' else m,axis,setting]
   shifts[m].append(v.reshape(8,1152))
 shifts={m:torch.stack(v).cuda() for m,v in shifts.items()};X=cap['native_captures'][17][source].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/rr)@W.T/30).square();P=cap['native_logits'][source].double().softmax(-1).cuda()
 def read(d):
  dd=d.cuda();return sat*((dd/rr-X*(dd*X).mean(-1,keepdim=True)/rr.pow(3))@W.T)
 def norm(z):return (((z-(z*P).sum(-1,keepdim=True)).square()*P).sum(-1)).sqrt()
 records=[];finals={};replays=[]
 for qi,q in enumerate(Q):
  ff={}
  for m in modes:
   d=q.repeat(len(rows),1)
   for l in range(10,18):
    d=float(model.blocks[l].lambdas[0])*d;d+=action(l,d);x,r,y,L,R,D=factors[l];dd=d.cuda();t=dd/r-x*(dd*x).mean(-1,keepdim=True)/r.pow(3);ys=y+shifts[m][:,l-10];d+=(((ys@L.T)*(t@R.T)+(t@L.T)*(ys@R.T))@D.T).cpu()
   ff[m]=d
  whole=q.repeat(len(rows),1)
  for l in range(10,18):whole=action(l,whole,True)
  replays.append(float(((whole-ff['native']).norm(dim=-1)/whole.norm(dim=-1)).max()));zz={m:read(d) for m,d in ff.items()};den=norm(zz['exact']-zz['native']);assert bool((den>0).all())
  for m,z in zz.items():
   errors=norm(z-zz['exact'])/den
   for i,row in enumerate(rows):records.append(dict(**{k:v for k,v in row.items() if k!='ids'},probe=qi,mode=m,error=float(errors[i]),effect_norm=float(den[i]),native_response_norm=float(norm(zz['native'])[i])))
  finals[qi]=ff;print('COMPLETE',qi,flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},whole_model_forwards=0,cached_attention_evaluations=attnevals,batched_jvp_calls=jvps,native_whole_block_replay_errors=replays,captured_context_max_abs=context_error,finite=all(np.isfinite(r['error']) for r in records));assert max(replays)<1e-4 and context_error<1e-5;torch.save(dict(final_tangents=finals,rows=rows),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
