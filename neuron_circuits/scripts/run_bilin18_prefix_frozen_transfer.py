"""H612 exact Attention5 interaction routing/local/persistent-value decomposition."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_frozen_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time()
 from transformers import AutoTokenizer
 from audit_bilin18_mlp_discriminators import TOKENIZER
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 pairs=[list(zip('Andrew Patrick Stephen Anthony Christopher Jonathan William Charles'.split(),'Rachel Rebecca Diana Julia Victoria Elizabeth Michelle Jennifer'.split())),list(zip('Arthur Albert Fred Philip Richard Steven Tony Vincent'.split(),'Olivia Sophia Maria Nicole Amy Karen Julie Angela'.split()))]
 forms=['He sent a letter to {n}. He sent a letter to {n}. He sent a letter to','A letter was delivered to {n}. A letter was delivered to {n}. A letter was delivered to','She mailed a parcel to {n}. She mailed a parcel to {n}. She mailed a parcel to','Name: {n}. Name: {n}. Name:']
 rows=[]
 for lex,pp in enumerate(pairs):
  for form,template in enumerate(forms):
   w=lex*4+form
   for item,pair in enumerate(pp):
    groups=[tok.encode(' '+n,add_special_tokens=False) for n in pair];assert all(len(g)==1 for g in groups)
    for v,n in enumerate(pair):
     for p in [0,1]:
      for c in [0,1]:
       body=template.format(n=n);body=body[0].lower()+body[1:] if c else body;prompt=('For the record, ' if p else ' ')+body
       rows.append(dict(id=f'{w}-{item}-{v}-{p}-{c}',wording=w,lexical_group=lex,form=form,item=item,variant=v,prefix=p,lower=c,prompt=prompt,ids=tok.encode(prompt,add_special_tokens=False),groups=groups))
 mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));ex={r['id']:r for r in rows};index={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows}
 for w in range(8):
  for item in range(8):
   for v in [0,1]:
    for c in [0,1]:assert ex[index[w,item,v,1,c]]['ids'][4:]==ex[index[w,item,v,0,c]]['ids']
 for w in range(8):
  for item in range(8):
   for v in [0,1]:
    for p in [0,1]:assert len(ex[index[w,item,v,p,0]]['ids'])==len(ex[index[w,item,v,p,1]]['ids'])
 bp=ART/'bilin18_prefix_joint_conditional_locked_bases.pt';bh=hashlib.sha256(bp.read_bytes()).hexdigest();bases=torch.load(bp,weights_only=False)['bases']
 cfg=dict(experiment='H616',runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),basis_sha256=bh,design='Frozen H615 joint32/joint64 and random64. Cross familiar/different names with two familiar and two unseen forms. No refitting or output filtering. Full contextual Attention5 interaction removal is reference; all body positions edited.',scope='Different names relative to H615; no training-data novelty claim. Projection requires observed interaction. Copying accuracy separate from fidelity.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 att=model.blocks[5].attn;assert att.squared_attn;H=model.config.n_head;D=model.config.n_embd//H;lam=float(att.lamb);native={};features={};batches=defaultdict(list)
 for k,e in ex.items():batches[len(e['ids'])].append(k)
 def forward(kk,capture=False,delta=None):
  cap={};hs=[]
  def target(m,args):cap['target']=args[0][:,-1].cpu()
  hs.append(model.blocks[11].mlp.register_forward_pre_hook(target))
  if capture:
   def pre(m,args):cap['bus']=args[1].cpu()
   def out(m,args,out):cap['write']=out[0].cpu()
   hs.extend([att.register_forward_pre_hook(pre),att.register_forward_hook(out)])
   for name in ['c_q','c_k','c_q2','c_k2','c_v']:
    def h(m,args,out,name=name):cap[name]=out.cpu()
    hs.append(getattr(att,name).register_forward_hook(h))
  if delta is not None:
   def edit(m,args,out):y,bus=out;y=y.clone();y[:,4:]-=delta.to(y);return y,bus
   hs.append(att.register_forward_hook(edit))
  try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in kk],device='cuda'))
  finally:
   for h in hs:h.remove()
  return z.cpu(),cap
 for kk in batches.values():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];z,c=forward(bb,capture=True)
   for i,k in enumerate(bb):features[k]={n:v[i].clone() for n,v in c.items()};native[k]=z[i]
 aligned={};replays=[]
 def apply(R,V):
  mixed=torch.einsum('hqk,khd->qhd',R.cuda().double(),V.cuda().double());return F.linear(mixed.reshape(len(mixed),H*D),att.c_proj.weight.double()).cpu()
 for k,e in ex.items():
  f=features[k];T=len(e['ids']);qraw=f['c_q'].cuda().reshape(1,T,H,D);cos,sin=att.rotary(qraw);qs={}
  for n in ['c_q','c_k','c_q2','c_k2']:
   x=f[n].cuda().reshape(1,T,H,D);qs[n]=model.tt.apply_rotary_emb(F.rms_norm(x,(D,)),cos,sin)[0]
  R=(torch.einsum('qhd,khd->hqk',qs['c_q'],qs['c_k'])/D)*(torch.einsum('qhd,khd->hqk',qs['c_q2'],qs['c_k2'])/D);R*=torch.ones(T,T,device='cuda').tril();R=R.cpu();local=f['c_v'].reshape(T,H,D);bus=f['bus'].reshape(T,H,D);pred=(1-lam)*apply(R,local)+lam*apply(R,bus);replays.append(float((pred-f['write'].double()).norm()/f['write'].double().norm()))
  if e['prefix']:aligned[k]=dict(R=R[:,4:],local=local,bus=bus)
  else:
   padded=torch.zeros(H,T,T+4);padded[:,:,4:]=R;aligned[k]=dict(R=padded,local=torch.cat([torch.zeros(4,H,D),local]),bus=torch.cat([torch.zeros(4,H,D),bus]))
 assert max(replays)<1e-5
 components={};targets={};combined=[];geometry=[];sum_errors=[];native_interaction_errors=[]
 for w in range(8):
  for item in range(8):
   for v in [0,1]:
    ids=[index[w,item,v,p,c] for p,c in [(0,0),(0,1),(1,0),(1,1)]];assert ex[ids[2]]['ids'][4:]==ex[ids[0]]['ids'];k=ids[3];combined.append(k);fs=[features[ii] for ii in ids];aa=[aligned[ii] for ii in ids];rr=[a['R'].double() for a in aa];ri=rr[3]-rr[2]-rr[1]+rr[0];vv={};busvi=None
    for stream,weight in [('local',1-lam),('bus',lam)]:
     vals=[a[stream].double() for a in aa];vi=vals[3]-vals[2]-vals[1]+vals[0];outputs=[apply(r,x)*weight for r,x in zip(rr,vals)];full=outputs[3]-outputs[2]-outputs[1]+outputs[0];routing=apply(ri,vals[0])*weight;value=apply(rr[0],vi)*weight;mixed=full-routing-value;vv[stream]=full;vv[stream+'_routing']=routing;vv[stream+'_value']=value;vv[stream+'_mixed']=mixed
     if stream=='bus':busvi=float(vi.norm())
    vv['full']=vv['local']+vv['bus'];vv['routing']=vv['local_routing']+vv['bus_routing'];vv['value']=vv['local_value']+vv['bus_value'];vv['mixed']=vv['local_mixed']+vv['bus_mixed'];nf=fs[3]['write'][4:]-fs[2]['write'][4:]-fs[1]['write']+fs[0]['write'];native_interaction_errors.append(float((vv['full']-nf.double()).norm()/nf.double().norm()));sum_errors.append(float((vv['routing']+vv['value']+vv['mixed']-vv['full']).norm()/vv['full'].norm()));geometry.append(dict(id=k,wording=w,norms={n:float(x.norm()) for n,x in vv.items()},local_bus_cosine=float(F.cosine_similarity(vv['local'].flatten(),vv['bus'].flatten(),dim=0)),routing_value_cosine=float(F.cosine_similarity(vv['routing'].flatten(),vv['value'].flatten(),dim=0)),bus_value_interaction_norm=busvi));components[k]={n:x.float() for n,x in vv.items()};targets[k]=fs[3]['target']-fs[2]['target']-fs[1]['target']+fs[0]['target']
 assert max(native_interaction_errors)<1e-4
 competence={}
 for w in range(8):
  for p in [0,1]:
   for c in [0,1]:
    both=[];top=[]
    for item in range(8):
     mm=[measure(native[index[w,item,v,p,c]],ex[index[w,item,v,p,c]]['groups']) for v in [0,1]];both.append(mm[0]['margin']>0 and mm[1]['margin']<0);top.extend(mm[v]['top1'][v] for v in [0,1])
    competence[f'{w}_{p}_{c}']=dict(both_endpoint_choice=float(np.mean(both)),full_vocab_top1=float(np.mean(top)),passes=bool(np.mean(both)>=.75 and np.mean(top)>=.5))
 print('CAPTURED',max(replays),competence,flush=True)
 jobs=defaultdict(list)
 for k in combined:jobs[len(ex[k]['ids'])].append(k)
 modes=['full','native','joint32','joint64','random64'];refs={};result=dict(config=cfg,competence=competence,interventions=[],summary={},status='running')
 def vector(k,mode):
  full=components[k]['local']
  if mode=='full':return full
  if mode=='native':return torch.zeros_like(full)
  B=bases['random'] if mode=='random64' else bases['joint'][:int(mode[5:])]
  return (full@B.T)@B
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save();identity=[]
 for mode in modes:
  for kk in jobs.values():
   for st in range(0,len(kk),16):
    bb=kk[st:st+16];delta=torch.stack([vector(k,mode) for k in bb]);z,c=forward(bb,delta=delta)
    for i,k in enumerate(bb):
     e=ex[k];zz=z[i];tt=c['target'][i]
     if mode=='full':refs[k]=dict(logits=zz,target=tt)
     if mode=='native':identity.append(float((zz-native[k]).abs().max()))
     rp=refs[k]['logits'].double().log_softmax(0);lp=zz.double().log_softmax(0);full=components[k]['local'];ans=e['groups'][e['variant']][0];den=(refs[k]['target']-features[k]['target']).norm();baseI=targets[k]
     result['interventions'].append(dict(id=k,wording=e['wording'],item=e['item'],mode=mode,write_relative_error=float((vector(k,mode)-full).norm()/full.norm()),target_relative_error=float((tt-refs[k]['target']).norm()/den),target_effect_norm=float(den),target_interaction_ratio=float((baseI+tt-features[k]['target']).norm()/baseI.norm()),reference_kl=float((rp.exp()*(rp-lp)).sum()),margin_error=measure(zz,e['groups'])['margin']-measure(refs[k]['logits'],e['groups'])['margin'],target_ce_error=float(rp[ans]-lp[ans]),full_vocab_correct=measure(zz,e['groups'])['top1'][e['variant']]))
  save();print('COMPLETE',mode,flush=True)
 for w in range(8):
  cc=[c for c in result['interventions'] if c['wording']==w];base=[c for c in cc if c['mode']=='native'];kd=np.mean([c['reference_kl'] for c in base]);md=np.mean([abs(c['margin_error']) for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
  for mode in modes:
   vv=[c for c in cc if c['mode']==mode];out={n:float(np.mean([c[n] for c in vv])) for n in ['write_relative_error','target_relative_error','target_effect_norm','target_interaction_ratio','reference_kl','full_vocab_correct']};out.update(n=len(vv),omission_kl=float(kd),relative_reference_kl=out['reference_kl']/kd,relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd));result['summary'][f'{w}_{mode}']=out
 old=json.loads((ART/'bilin18_prefix_boundary_control_manifest.json').read_text());oldprompts={r['prompt'] for r in old}
 result['audits']=dict(runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['manifest_sha256']==hashlib.sha256(mp.read_bytes()).hexdigest(),bases_unchanged=bh==hashlib.sha256(bp.read_bytes()).hexdigest(),native_prompts=len(rows),interventions=len(result['interventions']),native_write_replay_max_relative_error=max(replays),native_interaction_replay_max_relative_error=max(native_interaction_errors),identity_replay_max_logit_error=max(identity),prior_prompt_overlap={w:sum(r['prompt'] in oldprompts for r in rows if r['wording']==w) for w in range(8)},finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_relative_error','reference_kl','write_relative_error']))
 torch.save(dict(components={k:v['local'] for k,v in components.items()},targets=targets,native=native),ART/f'{STEM}_captures.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
