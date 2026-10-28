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
STEM='bilin18_emission_probe_directions'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();mp=ART/'bilin18_emission_prospective_panel_manifest.json';cp=ART/'bilin18_emission_prospective_panel_captures.pt';vp=ART/'bilin18_emission_surface_transfer_locked.pt';bp=ART/'bilin18_prefix_joint_conditional_locked_bases.pt';rows=json.loads(mp.read_text());source=torch.load(cp,weights_only=False);trace=source['trace'];caches=source['caches'];bus=source['bus'];x0=source['x0'];v=torch.load(vp,weights_only=False)['common'].double();B=torch.load(bp,weights_only=False)['bases']['joint'].double();unit=v/v.norm();probes={'original':v};orth=[unit]
 for i in range(4):
  q=B[i].clone()
  for old in orth:q-=q@old*old
  q/=q.norm();orth.append(q);probes[f'span{i}']=q*v.norm()
 gen=torch.Generator().manual_seed(632)
 for i in range(4):
  q=torch.randn(1152,generator=gen,dtype=torch.float64);q-=q@B.T@B
  for old in orth:q-=q@old*old
  q/=q.norm();orth.append(q);probes[f'orthogonal{i}']=q*v.norm()
 locked=ART/f'{STEM}_locked.pt';torch.save(probes,locked);paths={'manifest':mp,'source':cp,'vector':vp,'basis':bp,'probes':locked,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};refs=[i for i,e in enumerate(rows) if e['split']=='reference'];cfg=dict(experiment='H632',hashes=hashes,seed=632,probes=list(probes),reference_ids=[rows[i]['id'] for i in refs],design='Equal-norm orthogonal probes: original, joint-basis rows0-3 orthogonalized, four seeded random joint-span-complement directions. Native and shared earlyMLP bilinear-context/RMS operators; no response selection. Native finite-difference checks on original/span0/orthogonal0 at .02.',scope='Nine sampled input directions, not full operator rank. Original is historically selected; random signs arbitrary. Same prospective panel and frozen four reference contexts.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 buckets=defaultdict(list)
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
 jvp_calls=0
 def native_action(layer,kind,d):
  nonlocal jvp_calls
  out=torch.empty_like(d)
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];out[bb]=(action(layer,bb,d[bb]) if kind=='block' else micro(layer,kind,bb,d[bb]));jvp_calls+=1
  return out
 X5=torch.stack([caches[6][i][-1] for i in range(len(rows))]).double()-trace['captures'][0.]['mlp5'].double();r5=(X5.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();y5=(X5/r5).cuda();mlp=model.blocks[5].mlp;L5=mlp.Left.weight.double();R5=mlp.Right.weight.double();D5=mlp.Down.weight.double()
 def initial(q):
  t=(q/r5-X5*(X5*q).mean(-1,keepdim=True)/r5.pow(3)).cuda();return q+(((y5@L5.T)*(t@R5.T)+(t@L5.T)*(y5@R5.T))@D5.T).cpu()
 X=trace['captures'][0.]['block17'].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();W=model.lm_head.weight.double();sat=1-torch.tanh((X/rr)@W.T/30).square()
 def readout(d):
  d=d.cuda();dy=d/rr-X*(X*d).mean(-1,keepdim=True)/rr.pow(3);return (sat*(dy@W.T)).cpu()
 outputs={};records=[];modes=['native','share_context','share_norm']
 for name,q in probes.items():
  outputs[name]={}
  for mode in modes:
   d=initial(q)
   for l in range(6,18):
    if l<10:
     d=float(model.blocks[l].lambdas[0])*d;d+=native_action(l,'attn',d);d+=factored(l,d,'analytic_native' if mode=='native' else mode)
    else:d=native_action(l,'block',d)
   outputs[name][mode]=readout(d)
  print('COMPLETE',name,flush=True)
 # Fixed, predeclared finite-difference validation probes.
 validation=[];finite_forwards=0
 for name in ['original','span0','orthogonal0']:
  zz={}
  for alpha in [-.02,.02]:
   zall=torch.empty_like(nativeZ)
   for ii in buckets.values():
    for st in range(0,len(ii),16):
     bb=ii[st:st+16]
     def edit(m,args,out):y,b=out;y=y.clone();y[:,-1]+=alpha*probes[name].to(y);return y,b
     h=model.blocks[5].attn.register_forward_hook(edit)
     try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
     finally:h.remove()
     zall[bb]=z.cpu();finite_forwards+=len(bb)
   zz[alpha]=zall
  actual=(zz[.02].double()-zz[-.02].double())/.04
  for i,e in enumerate(rows):
   p=nativeZ[i].double().softmax(0);a=actual[i];pred=outputs[name]['native'][i]
   def norm(z):return ((z-(p*z).sum()).square()*p).sum().sqrt()
   validation.append(dict(id=e['id'],probe=name,error=float(norm(pred-a)/norm(a))))
 grams={}
 for i,e in enumerate(rows):
  p=nativeZ[i].double().softmax(0)
  def center(x):return x-(p*x).sum()
  def norm(x):return (center(x).square()*p).sum().sqrt()
  for name in probes:
   g=outputs[name]['native'][i]
   for mode in modes:
    h=outputs[name][mode][i];records.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],probe=name,mode=mode,native_response_norm=float(norm(g)),response_norm=float(norm(h)),change_norm=float(norm(h-g)),relative_change=float(norm(h-g)/norm(g)),answer_derivative=float(center(h)[e['answer_id']])))
  for kind in ['native','context_change']:
   M=torch.stack([center(outputs[name]['native'][i] if kind=='native' else outputs[name]['share_context'][i]-outputs[name]['native'][i]) for name in probes]);gram=(M*p)@M.T;key=f"{e['family']}_{e['form']}_{kind}";grams.setdefault(key,[]).append(gram)
 spectrum={}
 for key,gg in grams.items():
  G=torch.stack(gg).mean(0);eig=torch.linalg.eigvalsh(G).flip(0).clamp_min(0);spectrum[key]=dict(eigenvalues=eig.tolist(),top1=float(eig[:1].sum()/eig.sum()),top2=float(eig[:2].sum()/eig.sum()),top4=float(eig[:4].sum()/eig.sum()),participation_ratio=float(eig.sum().square()/eig.square().sum()))
 result=dict(config=cfg,records=records,validation=validation,spectrum=spectrum,summary={},status='complete')
 for family,form in sorted({(e['family'],e['form']) for e in rows}):
  for name in probes:
   for mode in modes:
    rs=[r for r in records if r['family']==family and r['form']==form and r['probe']==name and r['mode']==mode];result['summary'][f'{family}_{form}_{name}_{mode}']={n:float(np.mean([r[n] for r in rs])) for n in ['native_response_norm','response_norm','change_norm','relative_change','answer_derivative']};result['summary'][f'{family}_{form}_{name}_{mode}']['positive_count']=sum(r['answer_derivative']>0 for r in rs)
 replay=max(float(((outputs['original'][mode]-source['responses'][mode]).norm(dim=-1)/source['responses'][mode].norm(dim=-1)).max()) for mode in modes);Q=torch.stack(list(probes.values()))/v.norm();result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},orthogonality_max_abs=float((Q@Q.T-torch.eye(len(Q))).abs().max()),original_response_replay_relative_max=replay,finite_difference_max_relative_error=max(r['error'] for r in validation),finite_forwards=finite_forwards,cached_attention_evaluations=attention_evals,batched_jvp_calls=jvp_calls,finite=all(np.isfinite(r['relative_change']) for r in records));result['elapsed_seconds']=time.time()-start;torch.save(dict(responses=outputs,grams=grams),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
