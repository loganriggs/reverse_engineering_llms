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
STEM='bilin18_emission_prospective_panel'
@torch.no_grad()
def main():
 from transformers import AutoTokenizer
 from audit_bilin18_mlp_discriminators import TOKENIZER
 torch.set_num_threads(6);start=time.time();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);oldmp=ART/'bilin18_emission_surface_transfer_manifest.json';vp=ART/'bilin18_emission_surface_transfer_locked.pt';oldcp=ART/'bilin18_emission_operator_chain_captures.pt';oldrows=json.loads(oldmp.read_text());v=torch.load(vp,weights_only=False)['common'];oldchain=torch.load(oldcp,weights_only=False)
 rows=[]
 for e in oldrows:
  if (e['family']=='capital' and e['form']==0 and e['item']<4) or (e['family']=='language' and e['form']==1):
   rows.append(dict(e,split='reference' if e['family']=='capital' else 'anchor'))
 countries='Portugal Brazil Austria Mexico Poland Turkey Greece Sweden'.split();answers='Portuguese Portuguese German Spanish Polish Turkish Greek Swedish'.split();candidate_text=list(dict.fromkeys('French German Italian Japanese Spanish Chinese Russian English Portuguese Polish Turkish Greek Swedish'.split()));candidate_ids=[]
 for text in candidate_text:
  ids=tok.encode(' '+text,add_special_tokens=False);assert len(ids)==1;(candidate_ids.append(ids[0]))
 oldsubjects={e['subject'] for e in oldrows};assert not set(countries)&oldsubjects
 for form,template in enumerate(['People in {} speak','Residents of {} speak','The main language spoken in {} is']):
  for i,(country,answer) in enumerate(zip(countries,answers)):
   prompt=template.format(country);variants=[[[t]] for t in candidate_ids];rows.append(dict(id=f'fresh-{form}-{i}',family='fresh',form=form,item=i,subject=country,answer=answer,answer_id=tok.encode(' '+answer,add_special_tokens=False)[0],candidate_ids=candidate_ids,variants=variants,prompt=prompt,ids=tok.encode(prompt,add_special_tokens=False),split='prospective'))
 for e in rows:
  if e['family']!='fresh':e['answer_id']=e['variants'][e['item']][0][0];e['candidate_ids']=candidate_ids if e['family']=='language' else [vv[0][0] for vv in e['variants']]
 oldprompts={e['prompt'] for e in oldrows};assert all(e['prompt'] not in oldprompts for e in rows if e['split']=='prospective')
 mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));paths={'manifest':mp,'vector':vp,'old_manifest':oldmp,'reference_chain':oldcp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};refs=[i for i,e in enumerate(rows) if e['split']=='reference'];regions={'native':[],'subset15':[f'mlp{l}' for l in range(6,10)],'analytic_native':[],'share_context':[],'share_norm':[],'share_paired':[]};cfg=dict(experiment='H631',hashes=hashes,reference_ids=[rows[i]['id'] for i in refs],regions=regions,design='Prospective24 prompts: eight countries absent from prior panel crossed with speak/residents/main-language wordings. Frozen vector and original four capital-reference contexts; native, normalizedMLP6-9 sharing, bilinear-context-only and RMS-only sharing. Eight original speak anchors. No fitting, output-based filtering or new subset search.',scope='New relative to current experimental panel, not model-training novelty. Predeclared conventional-language targets are not exhaustive descriptions of multilingual countries. Candidate set includes13 languages; all failures retained.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 caches={l:{} for l in range(6,18)};bus={};x0={};buckets=defaultdict(list)
 for i,e in enumerate(rows):buckets[len(e['ids'])].append(i)
 captures={};logits={}
 for alpha in [-.02,0.,.02]:
  captures[alpha]={f'block{l}':torch.empty(len(rows),1152) for l in range(5,18)};captures[alpha]['mlp5']=torch.empty(len(rows),1152);logits[alpha]=torch.empty(len(rows),model.lm_head.weight.shape[0])
  for ii in buckets.values():
   for st in range(0,len(ii),16):
    bb=ii[st:st+16];cap={};hs=[]
    def edit(m,args,out):y,b=out;y=y.clone();y[:,-1]+=alpha*v.to(y);return y,b
    hs.append(model.blocks[5].attn.register_forward_hook(edit))
    def mh(m,args,out):cap['mlp5']=out[:,-1].cpu()
    hs.append(model.blocks[5].mlp.register_forward_hook(mh))
    for l in range(5,18):
     def bh(m,args,out,l=l):cap[f'block{l}']=out[0][:,-1].cpu()
     hs.append(model.blocks[l].register_forward_hook(bh))
     if l>=6 and alpha==0:
      def pre(m,args,l=l):
       for n,i in enumerate(bb):
        caches[l][i]=args[0][n].cpu().clone()
        if l==6:bus[i]=args[1][n].cpu().clone();x0[i]=args[2][n].cpu().clone()
      hs.append(model.blocks[l].register_forward_pre_hook(pre))
    try:z,_=run(model,torch.tensor([rows[i]['ids'] for i in bb],device='cuda'))
    finally:
     for h in hs:h.remove()
    logits[alpha][bb]=z.cpu()
    for name in cap:captures[alpha][name][bb]=cap[name]
  print('CAPTURED',alpha,flush=True)
 trace=dict(captures=captures,logits=logits,tangents={name:(captures[.02][name].double()-captures[-.02][name].double())/.04 for name in captures[0.]});fac={'exact':trace['tangents']['mlp5']};referr=max(float((caches[l][i]-oldchain['caches'][l][next(k for k,e in enumerate(oldrows) if e['id']==rows[i]['id'])]).abs().max()) for l in range(6,18) for i in refs);competence=[]
 # Use original cached reference states exactly; native replay can differ slightly with batch shape.
 for i in refs:
  oi=next(k for k,e in enumerate(oldrows) if e['id']==rows[i]['id'])
  for l in range(6,18):caches[l][i]=oldchain['caches'][l][oi].clone()
  bus[i]=oldchain['bus'][oi].clone();x0[i]=oldchain['x0'][oi].clone()
 for i,e in enumerate(rows):
  z=logits[0.][i];ans=e['answer_id'];cs=e['candidate_ids'];competence.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],top1_correct=int(z.argmax())==ans,candidate_correct=cs[int(z[cs].argmax())]==ans,answer_mass=float(z.double().softmax(0)[ans]),top_token=tok.decode([int(z.argmax())])))
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
  p=nativeZ[i].double().softmax(0);g=actual[i];ans=e['answer_id']
  def center(x):return x-(p*x).sum()
  def norm(x):return (center(x).square()*p).sum().sqrt()
  for mode,h in responses.items():records.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],mode=mode,response_change=float(norm(h[i]-g)/norm(g)),response_norm_ratio=float(norm(h[i])/norm(g)),answer_logprob_derivative=float(center(h[i])[ans]),final_residual_change=float((trajectories[mode][17][i]-trace['tangents']['block17'][i]).norm()/trace['tangents']['block17'][i].norm())))
  for l in range(5,18):validation.append(dict(id=e['id'],layer=l,relative_error=float((trajectories['native'][l][i]-trace['tangents'][f'block{l}'][i]).norm()/trace['tangents'][f'block{l}'][i].norm())))
 result=dict(config=cfg,records=records,validation=validation,summary={},status='complete')
 for family,form in sorted({(e['family'],e['form']) for e in rows}):
  for split in sorted({e['split'] for e in rows if e['family']==family and e['form']==form}):
   for mode in regions:
    rr=[r for r in records if r['family']==family and r['form']==form and r['split']==split and r['mode']==mode];result['summary'][f'{family}_{form}_{split}_{mode}']={n:float(np.mean([r[n] for r in rr])) for n in ['response_change','response_norm_ratio','answer_logprob_derivative','final_residual_change']}
 result['competence']=competence
 result['audits']=dict(hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},native_forwards=3*len(rows),reference_input_replay_max_abs=referr,cached_attention_evaluations=attention_evals,batched_jvp_calls=jvp_calls,native_factorization_max_relative_error=max(float(((trajectories['native'][l]-trace['tangents'][f'block{l}']).norm(dim=-1)/trace['tangents'][f'block{l}'].norm(dim=-1)).max()) for l in range(6,18)),analytic_native_max_relative_error=float(((trajectories['analytic_native'][17]-trajectories['native'][17]).norm(dim=-1)/trajectories['native'][17].norm(dim=-1)).max()),paired_factor_max_relative_error=float(((trajectories['share_paired'][17]-trajectories['subset15'][17]).norm(dim=-1)/trajectories['subset15'][17].norm(dim=-1)).max()),native_chain_max_residual_error=max(r['relative_error'] for r in validation),native_chain_max_logit_error=max(r['response_change'] for r in records if r['mode']=='native'),finite=all(np.isfinite(r['response_change']) for r in records));result['elapsed_seconds']=time.time()-start;torch.save(dict(trajectories=trajectories,responses=responses,caches=caches,bus=bus,x0=x0,trace=trace),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
