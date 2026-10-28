"""H637 matched cumulative bilinear-context replacement in MLP6-9 and5-9."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_matched_early_context_chain'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'manifest':ART/'bilin18_cross_task_response_panel_manifest.json','capture':ART/'bilin18_matched_mlp5_contexts_captures.pt','source':ART/'bilin18_cross_task_response_panel_captures.pt','internal':ART/'bilin18_internal_response_subspaces_captures.pt','runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};rows=json.loads(paths['manifest'].read_text());cap=torch.load(paths['capture'],weights_only=False);source=torch.load(paths['source'],weights_only=False);internal=torch.load(paths['internal'],weights_only=False);names=['original','span2','orthogonal2'];caches=cap['caches'];bus=cap['bus'];x0=cap['x0'];lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};donors={kind:[lookup[e['family'],1-e['form'],e['item']] if kind=='cross' else lookup[e['family'],e['form'],(e['item']+4)%8] for e in rows] for kind in ['cross','within']};modes=['native','cross_6_9','cross_5_9','within_6_9','within_5_9'];cfg=dict(experiment='H637',hashes=hashes,probes=names,modes=modes,design='Matched donor bilinear operating contexts in MLP6-9, alone or with MLP5. Same-item opposite template or same-template item+4 controls. Native normalization, attention, residual paths and readout. Frozen native states, local tangent counterfactuals; no fitting.',primary='Remaining actual donor-response gap, measured using receiver probability weights; magnitude and alignment secondary.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 raw={l:{} for l in range(6,10)};post={l:{} for l in range(6,10)};attnevals=0
 for l in range(6,10):
  block=model.blocks[l]
  for ii in buckets.values():
   bb=ii;x=torch.stack([caches[l][i] for i in bb]).cuda();e=torch.stack([x0[i] for i in bb]).cuda();b=torch.stack([bus[i] for i in bb]).cuda();r=block.lambdas[0]*x+block.lambdas[1]*e;a,_=block.attn(F.rms_norm(r,(1152,)),b);attnevals+=len(bb)
   for k,i in enumerate(bb):raw[l][i]=r[k].cpu();post[l][i]=(r[k,-1]+a[k,-1]).cpu()
 factors={}
 for l in range(6,10):
  X=torch.stack([post[l][i] for i in range(len(rows))]).double().cuda();r=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();m=model.blocks[l].mlp;factors[l]=(X,r,X/r,m.Left.weight.double(),m.Right.weight.double(),m.Down.weight.double())
 jvps=0
 def action(l,d,kind):
  nonlocal jvps
  result=torch.empty_like(d)
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];b=torch.stack([bus[i] for i in bb]).cuda()
    if kind=='attn':
     x=torch.stack([raw[l][i] for i in bb]).cuda();prefix=x[:,:-1];last=x[:,-1];mod=model.blocks[l].attn
     def fn(tail):return mod(F.rms_norm(torch.cat([prefix,tail[:,None]],1),(1152,)),b)[0][:,-1]
    else:
     x=torch.stack([caches[l][i] for i in bb]).cuda();prefix=x[:,:-1];last=x[:,-1];e0=torch.stack([x0[i] for i in bb]).cuda();block=model.blocks[l]
     def fn(tail):return block(torch.cat([prefix,tail[:,None]],1),b,e0)[0][:,-1]
    _,out=torch.func.jvp(fn,(last,),(d[bb].cuda().float(),));result[bb]=out.cpu().double();jvps+=1
  return result
 X=internal['native_captures'][17].double().cuda();r=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/r)@W.T/30).square();outputs={};trajectories={};records=[];validation=[]
 for name in names:
  outputs[name]={};trajectories[name]={}
  for mode in modes:
   kind=mode.split('_')[0];d=cap['initials'][name][kind+'_context' if mode.endswith('5_9') else 'native'];trajectories[name][mode]={}
   for l in range(6,18):
    if l<10:
     d=float(model.blocks[l].lambdas[0])*d;d+=action(l,d,'attn');xx,rr,y,L,R,D=factors[l];dd=d.cuda();t=dd/rr-xx*(xx*dd).mean(-1,keepdim=True)/rr.pow(3);yy=y if mode=='native' else y[donors[kind]];dg=((yy@L.T)*(t@R.T)+(t@L.T)*(yy@R.T))@D.T;d+=dg.cpu()
    else:d=action(l,d,'block')
    if l in [9,17]:trajectories[name][mode][l]=d.clone()
   d=d.cuda();outputs[name][mode]=(sat*((d/r-X*(X*d).mean(-1,keepdim=True)/r.pow(3))@W.T)).cpu()
  for i,e in enumerate(rows):
   p=source['native_logits'][i].double().softmax(0)
   def center(v):return v-(p*v).sum()
   def dot(x,y):return (p*x*y).sum()
   g=center(outputs[name]['native'][i]);truth=center(source['responses'][name][i]);validation.append(dict(id=e['id'],probe=name,error=float(dot(g-truth,g-truth).sqrt()/dot(truth,truth).sqrt()),replay=float(dot(g-center(cap['outputs'][name]['native'][i]),g-center(cap['outputs'][name]['native'][i])).sqrt()/dot(g,g).sqrt())))
   for mode in modes[1:]:
    kind=mode.split('_')[0];h=center(outputs[name][mode][i]);gap=center(source['responses'][name][donors[kind][i]])-g;change=h-g;res=gap-change;records.append(dict(id=e['id'],family=e['family'],form=e['form'],item=e['item'],probe=name,mode=mode,relative_change=float(dot(change,change).sqrt()/dot(g,g).sqrt()),remaining_relative_gap=float(dot(res,res).sqrt()/dot(gap,gap).sqrt()),alignment=float(dot(change,gap)/(dot(change,change)*dot(gap,gap)).sqrt()),target_derivative=float(h[e['answer_id']]),native_target_derivative=float(g[e['answer_id']])))
  print('COMPLETE',name,flush=True)
 audits=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_whole_model_forwards=0,cached_attention_evaluations=attnevals,batched_jvp_calls=jvps,native_chain_max_error=max(v['error'] for v in validation),native_factorization_replay_max_error=max(v['replay'] for v in validation),finite=all(np.isfinite(v['remaining_relative_gap']) for v in records));torch.save(dict(outputs=outputs,trajectories=trajectories),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,validation=validation,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
