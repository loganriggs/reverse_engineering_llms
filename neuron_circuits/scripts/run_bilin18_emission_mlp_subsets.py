"""H630 exhaustive MLP subsets and normalization/bilinear factor controls."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_emission_mlp_subsets'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();mp=ART/'bilin18_emission_surface_transfer_manifest.json';vp=ART/'bilin18_emission_surface_transfer_locked.pt';tp=ART/'bilin18_emission_readout_trace_captures.pt';fp=ART/'bilin18_mlp5_response_factors_captures.pt';rows=json.loads(mp.read_text());v=torch.load(vp,weights_only=False)['common'];trace=torch.load(tp,weights_only=False);fac=torch.load(fp,weights_only=False);paths={'manifest':mp,'vector':vp,'trace':tp,'factors':fp,'runner':Path(__file__)};chainpath=ART/'bilin18_emission_operator_chain_captures.pt';paths['chain']=chainpath;hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};refs=[i for i,e in enumerate(rows) if e['family']=='capital' and e['form']==0 and e['item']<4];regions={('native' if mask==0 else f'subset{mask}'):[f'mlp{l}' for bit,l in enumerate(range(6,10)) if mask&(1<<bit)] for mask in range(16)};regions.update({name:[] for name in ['analytic_native','share_context','share_norm','share_factorwise','share_paired']});cfg=dict(experiment='H630',hashes=hashes,reference_ids=[rows[i]['id'] for i in refs],regions=regions,design='All16 subsets of normalizedMLP6-9 capital-reference operator replacements. Additional analytic factor controls: native bilinear/native RMS, mean reference normalized context/native RMS, native context/mean reference RMS Jacobian, product of means and mean paired products. Attention and later blocks native. No new direction fit.',scope='Exhaustive finite subset scan, not independently confirmed minimal circuit. Shared factors define local operators and need not correspond to an attainable state. Subset interaction means nonadditivity of these interventions, not a universal mechanism share.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 chain=torch.load(chainpath,weights_only=False);caches=chain['caches'];bus=chain['bus'];x0=chain['x0'];buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 nativeZ=trace['logits'][0.];raw={l:{} for l in range(6,10)};post={l:{} for l in range(6,10)};attention_evals=0
 for l in range(6,10):
  block=model.blocks[l]
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];x=torch.stack([caches[l][i] for i in bb]).cuda();e=torch.stack([x0[i] for i in bb]).cuda();b=torch.stack([bus[i] for i in bb]).cuda();r=block.lambdas[0]*x+block.lambdas[1]*e;a,_=block.attn(F.rms_norm(r,(1152,)),b);attention_evals+=len(bb)
    for n,i in enumerate(bb):raw[l][i]=r[n].cpu();post[l][i]=(r[n,-1]+a[n,-1]).cpu()
 def action(layer,contexts,tangents):
  x=torch.stack([caches[layer][i] for i in contexts]).cuda();b=torch.stack([bus[i] for i in contexts]).cuda();e=torch.stack([x0[i] for i in contexts]).cuda();last=x[:,-1];prefix=x[:,:-1];block=model.blocks[layer]
  def fn(tail):return block(torch.cat([prefix,tail[:,None]],dim=1),b,e)[0][:,-1]
  _,out=torch.func.jvp(fn,(last,),(tangents.cuda().float(),));return out.cpu().double()
 def micro(layer,kind,contexts,tangents):
  if kind=='attn':
   x=torch.stack([raw[layer][i] for i in contexts]).cuda();prefix=x[:,:-1];last=x[:,-1];b=torch.stack([bus[i] for i in contexts]).cuda();mod=model.blocks[layer].attn
   def fn(tail):return mod(F.rms_norm(torch.cat([prefix,tail[:,None]],dim=1),(1152,)),b)[0][:,-1]
  else:
   last=torch.stack([post[layer][i] for i in contexts]).cuda();mod=model.blocks[layer].mlp
   def fn(tail):return mod(F.rms_norm(tail,(1152,)))
  _,out=torch.func.jvp(fn,(last,),(tangents.cuda().float(),));return out.cpu().double()
 analytic_cache={}
 for l in range(6,10):
  X=torch.stack([post[l][i] for i in range(len(rows))]).double().cuda();r=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();mlp=model.blocks[l].mlp;analytic_cache[l]=(X,r,X/r,mlp.Left.weight.double(),mlp.Right.weight.double(),mlp.Down.weight.double())
 def factored(layer,d,mode):
  X,r,y,L,R,D=analytic_cache[layer];d=d.cuda()
  def normpart(x,rr):return d/rr-x*(x*d).mean(-1,keepdim=True)/rr.pow(3)
  def bilin(c,t):return ((c@L.T)*(t@R.T)+(t@L.T)*(c@R.T))@D.T
  t=normpart(X,r);yc=y[refs].mean(0).expand_as(y)
  if mode=='analytic_native':out=bilin(y,t)
  elif mode=='share_context':out=bilin(yc,t)
  elif mode=='share_paired':out=sum(bilin(y[i].expand_as(y),normpart(X[i].expand_as(X),r[i].expand_as(r))) for i in refs)/len(refs)
  else:
   tc=sum(normpart(X[i].expand_as(X),r[i].expand_as(r)) for i in refs)/len(refs);out=bilin(y if mode=='share_norm' else yc,tc)
  return out.cpu()
 trajectories={};jvp_calls=0
 def mapped(layer,kind,d,shared):
  nonlocal jvp_calls
  out=torch.empty_like(d)
  if shared:
   for st in range(0,len(rows),16):
    bb=list(range(st,min(st+16,len(rows))));out[bb]=sum(micro(layer,kind,[ref]*len(bb),d[bb]) for ref in refs)/len(refs);jvp_calls+=len(refs)
  else:
   for ii in buckets.values():
    for st in range(0,len(ii),16):
     bb=ii[st:st+16];out[bb]=micro(layer,kind,bb,d[bb]);jvp_calls+=1
  return out
 for mode,shared in regions.items():
  d=v.double()+fac['exact'];trajectory={5:d.clone()}
  for l in range(6,18):
   if l<10 and mode!='paired_early':
    d=float(model.blocks[l].lambdas[0])*d;d=d+mapped(l,'attn',d,False);d=d+(factored(l,d,mode) if mode in ['analytic_native','share_context','share_norm','share_factorwise','share_paired'] else mapped(l,'mlp',d,f'mlp{l}' in shared))
   else:
    nxt=torch.empty_like(d)
    if l<10:
     for st in range(0,len(rows),16):
      bb=list(range(st,min(st+16,len(rows))));nxt[bb]=sum(action(l,[ref]*len(bb),d[bb]) for ref in refs)/len(refs);jvp_calls+=len(refs)
    else:
     for ii in buckets.values():
      for st in range(0,len(ii),16):
       bb=ii[st:st+16];nxt[bb]=action(l,bb,d[bb]);jvp_calls+=1
    d=nxt
   trajectory[l]=d.clone()
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
 interactions=[]
 for i,e in enumerate(rows):
  rr={0:next(r for r in records if r['id']==e['id'] and r['mode']=='native')['answer_logprob_derivative']}
  for mask in range(1,16):rr[mask]=next(r for r in records if r['id']==e['id'] and r['mode']==f'subset{mask}')['answer_logprob_derivative']
  mob={}
  for mask in range(16):mob[mask]=sum((-1)**(mask.bit_count()-sub.bit_count())*rr[sub] for sub in range(16) if (sub&mask)==sub)
  for mask in range(16):
   first=rr[0]+sum(mob[1<<b] for b in range(4) if mask&(1<<b));second=sum(mob[sub] for sub in range(16) if (sub&mask)==sub and sub.bit_count()<=2);interactions.append(dict(id=e['id'],family=e['family'],form=e['form'],split='fit' if e['item']<4 else 'holdout',mask=mask,layers=[6+b for b in range(4) if mask&(1<<b)],actual=rr[mask],additive_prediction=first,additive_residual=rr[mask]-first,pairwise_prediction=second,pairwise_residual=rr[mask]-second,mobius=mob[mask]))
 result['subset_interactions']=interactions
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_forwards=0,cached_attention_evaluations=attention_evals,batched_jvp_calls=jvp_calls,native_factorization_max_relative_error=max(float(((trajectories['native'][l]-chain['trajectories']['native'][l]).norm(dim=-1)/chain['trajectories']['native'][l].norm(dim=-1)).max()) for l in range(6,18)),analytic_native_max_relative_error=float(((trajectories['analytic_native'][17]-trajectories['native'][17]).norm(dim=-1)/trajectories['native'][17].norm(dim=-1)).max()),paired_factor_max_relative_error=float(((trajectories['share_paired'][17]-trajectories['subset15'][17]).norm(dim=-1)/trajectories['subset15'][17].norm(dim=-1)).max()),native_chain_max_residual_error=max(r['relative_error'] for r in validation),native_chain_max_logit_error=max(r['response_change'] for r in records if r['mode']=='native'),finite=all(np.isfinite(r['response_change']) for r in records));result['elapsed_seconds']=time.time()-start;torch.save(dict(trajectories=trajectories,responses=responses,caches=caches,bus=bus,x0=x0),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
