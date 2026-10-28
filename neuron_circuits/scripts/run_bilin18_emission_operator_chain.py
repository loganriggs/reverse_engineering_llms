"""H628 forward-mode local operator chain and shared-context region tests."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_emission_operator_chain'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();mp=ART/'bilin18_emission_surface_transfer_manifest.json';vp=ART/'bilin18_emission_surface_transfer_locked.pt';tp=ART/'bilin18_emission_readout_trace_captures.pt';fp=ART/'bilin18_mlp5_response_factors_captures.pt';rows=json.loads(mp.read_text());v=torch.load(vp,weights_only=False)['common'];trace=torch.load(tp,weights_only=False);fac=torch.load(fp,weights_only=False);paths={'manifest':mp,'vector':vp,'trace':tp,'factors':fp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};refs=[i for i,e in enumerate(rows) if e['family']=='capital' and e['form']==0 and e['item']<4];regions={'native':[], 'shared_all':list(range(6,18)),'shared_early':list(range(6,10)),'shared_middle':list(range(10,14)),'shared_late':list(range(14,18))};cfg=dict(experiment='H628',hashes=hashes,reference_ids=[rows[i]['id'] for i in refs],regions=regions,design='Forward-mode JVP through blocks6-17 at fixed native states, last-position input tangent only; embeddings and persistent value bus constant. Start from analytic block5 tangent. Shared regions use arithmetic mean Jacobian action of four capital0 discovery contexts on the same incoming tangent. Native final readout retained.',scope='Local operator interventions, not altered native forward states or fixed-output replacement. Region effects nonadditive; reference operators evaluated on donor sequence contexts/positions, recipients can differ in length.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 caches={l:{} for l in range(6,18)};bus={};x0={};buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 nativeZ=torch.empty_like(trace['logits'][0.])
 for ii in buckets.values():
  for st in range(0,len(ii),16):
   bb=ii[st:st+16];hs=[]
   for l in range(6,18):
    def hook(m,args,l=l):
     for n,i in enumerate(bb):
      caches[l][i]=args[0][n].cpu().clone()
      if l==6:bus[i]=args[1][n].cpu().clone();x0[i]=args[2][n].cpu().clone()
    hs.append(model.blocks[l].register_forward_pre_hook(hook))
   try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   nativeZ[bb]=z.cpu()
 def action(layer,contexts,tangents):
  x=torch.stack([caches[layer][i] for i in contexts]).cuda();b=torch.stack([bus[i] for i in contexts]).cuda();e=torch.stack([x0[i] for i in contexts]).cuda();last=x[:,-1];prefix=x[:,:-1];block=model.blocks[layer]
  def fn(tail):return block(torch.cat([prefix,tail[:,None]],dim=1),b,e)[0][:,-1]
  _,out=torch.func.jvp(fn,(last,),(tangents.cuda().float(),));return out.cpu().double()
 trajectories={};jvp_calls=0
 for mode,shared in regions.items():
  d=v.double()+fac['exact'];trajectory={5:d.clone()}
  for l in range(6,18):
   nxt=torch.empty_like(d)
   if l in shared:
    for st in range(0,len(rows),16):
     ii=list(range(st,min(st+16,len(rows))));nxt[ii]=sum(action(l,[ref]*len(ii),d[ii]) for ref in refs)/len(refs);jvp_calls+=len(refs)
   else:
    for ii in buckets.values():
     for st in range(0,len(ii),16):
      bb=ii[st:st+16];nxt[bb]=action(l,bb,d[bb]);jvp_calls+=1
   d=nxt;trajectory[l]=d.clone()
  trajectories[mode]=trajectory;print('COMPLETE',mode,flush=True)
 X=trace['captures'][0.]['block17'].double().cuda();r=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/r)@W.T/30).square()
 def readout(d):
  d=d.cuda();dy=d/r-X*(X*d).mean(-1,keepdim=True)/r.pow(3);return (sat*(dy@W.T)).cpu()
 responses={m:readout(t[17]) for m,t in trajectories.items()};actual=(trace['logits'][.02].double()-trace['logits'][-.02].double())/.04;records=[];validation=[]
 for i,e in enumerate(rows):
  p=nativeZ[i].double().softmax(0);g=actual[i];ans=e['variants'][e['item']][0][0]
  def center(x):return x-(p*x).sum()
  def norm(x):return (center(x).square()*p).sum().sqrt()
  for mode,h in responses.items():records.append(dict(id=e['id'],family=e['family'],form=e['form'],split='fit' if e['item']<4 else 'holdout',mode=mode,response_change=float(norm(h[i]-g)/norm(g)),response_norm_ratio=float(norm(h[i])/norm(g)),answer_logprob_derivative=float(center(h[i])[ans]),final_residual_change=float((trajectories[mode][17][i]-trace['tangents']['block17'][i]).norm()/trace['tangents']['block17'][i].norm())))
  for l in range(5,18):validation.append(dict(id=e['id'],layer=l,relative_error=float((trajectories['native'][l][i]-trace['tangents'][f'block{l}'][i]).norm()/trace['tangents'][f'block{l}'][i].norm())))
 result=dict(config=cfg,records=records,validation=validation,summary={},status='complete')
 for family,form in sorted({(e['family'],e['form']) for e in rows}):
  for split in ['fit','holdout']:
   for mode in regions:
    rr=[r for r in records if r['family']==family and r['form']==form and r['split']==split and r['mode']==mode];result['summary'][f'{family}_{form}_{split}_{mode}']={n:float(np.mean([r[n] for r in rr])) for n in ['response_change','response_norm_ratio','answer_logprob_derivative','final_residual_change']}
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_forwards=len(rows),batched_jvp_calls=jvp_calls,native_replay_max_logit_error=float((nativeZ-trace['logits'][0.]).abs().max()),native_chain_max_residual_error=max(r['relative_error'] for r in validation),native_chain_max_logit_error=max(r['response_change'] for r in records if r['mode']=='native'),finite=all(np.isfinite(r['response_change']) for r in records));result['elapsed_seconds']=time.time()-start;torch.save(dict(trajectories=trajectories,responses=responses,caches=caches,bus=bus,x0=x0),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
