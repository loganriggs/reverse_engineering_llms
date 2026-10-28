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
STEM='bilin18_prefix_attention_factors'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_prefix_boundary_control_manifest.json';rows=json.loads(src.read_text());ex={r['id']:r for r in rows};index={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows};cfg=dict(experiment='H612',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='Attention5 aligned-body four-condition interaction. Separate contextual local and persistent values with native scalar coefficients. For each stream expand around unprefixed uppercase corner: routing-only=F(R_interaction,V00), value-only=F(R00,V_interaction), mixed=full interaction minus those two. Zero-pad missing prefix-key slots in unprefixed runs. Verify native write replay and exact sums before removals. Subtract components across body on combined prompt with norm-matched random controls.',scope='Baseline-anchored expansion, not a unique attribution or an additive behavioral share. Mixed terms include prefix-slot appearance and routing/value co-variation. Persistent bus remains live during output-write interventions. Same panel, adaptive follow-up to H611.',seed=612)
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
 for w in [0,1]:
  for item in range(8):
   for v in [0,1]:
    ids=[index[w,item,v,p,c] for p,c in [(0,0),(0,1),(1,0),(1,1)]];assert ex[ids[2]]['ids'][4:]==ex[ids[0]]['ids'];k=ids[3];combined.append(k);fs=[features[ii] for ii in ids];aa=[aligned[ii] for ii in ids];rr=[a['R'].double() for a in aa];ri=rr[3]-rr[2]-rr[1]+rr[0];vv={};busvi=None
    for stream,weight in [('local',1-lam),('bus',lam)]:
     vals=[a[stream].double() for a in aa];vi=vals[3]-vals[2]-vals[1]+vals[0];outputs=[apply(r,x)*weight for r,x in zip(rr,vals)];full=outputs[3]-outputs[2]-outputs[1]+outputs[0];routing=apply(ri,vals[0])*weight;value=apply(rr[0],vi)*weight;mixed=full-routing-value;vv[stream]=full;vv[stream+'_routing']=routing;vv[stream+'_value']=value;vv[stream+'_mixed']=mixed
     if stream=='bus':busvi=float(vi.norm())
    vv['full']=vv['local']+vv['bus'];vv['routing']=vv['local_routing']+vv['bus_routing'];vv['value']=vv['local_value']+vv['bus_value'];vv['mixed']=vv['local_mixed']+vv['bus_mixed'];nf=fs[3]['write'][4:]-fs[2]['write'][4:]-fs[1]['write']+fs[0]['write'];native_interaction_errors.append(float((vv['full']-nf.double()).norm()/nf.double().norm()));sum_errors.append(float((vv['routing']+vv['value']+vv['mixed']-vv['full']).norm()/vv['full'].norm()));geometry.append(dict(id=k,wording=w,norms={n:float(x.norm()) for n,x in vv.items()},local_bus_cosine=float(F.cosine_similarity(vv['local'].flatten(),vv['bus'].flatten(),dim=0)),routing_value_cosine=float(F.cosine_similarity(vv['routing'].flatten(),vv['value'].flatten(),dim=0)),bus_value_interaction_norm=busvi));components[k]={n:x.float() for n,x in vv.items()};targets[k]=fs[3]['target']-fs[2]['target']-fs[1]['target']+fs[0]['target']
 assert max(native_interaction_errors)<1e-4
 gen=torch.Generator().manual_seed(612);maxlen=max(len(components[k]['full']) for k in combined);random=torch.randn(maxlen,1152,generator=gen);random/=random.norm(dim=-1,keepdim=True);jobs=defaultdict(list)
 for k in combined:jobs[len(ex[k]['ids'])].append(k)
 names=['full','local','bus','routing','value','mixed','local_routing','local_value','local_mixed','bus_routing','bus_value','bus_mixed'];conditions=[('self','self')]+[(n,m) for n in names for m in ['interaction','random']];result=dict(config=cfg,value_mix=lam,geometry=geometry,interventions=[],summary={},status='running');replay=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save();print('DECOMPOSED',max(replays),max(native_interaction_errors),max(sum_errors),flush=True)
 for component,mode in conditions:
  for kk in jobs.values():
   for st in range(0,len(kk),16):
    bb=kk[st:st+16];delta=None
    if mode!='self':delta=torch.stack([components[k][component] if mode=='interaction' else random[:len(components[k][component])]*components[k][component].norm(dim=-1,keepdim=True) for k in bb])
    z,c=forward(bb,delta=delta)
    for i,k in enumerate(bb):
     e=ex[k];I=targets[k];shift=c['target'][i]-features[k]['target'];zz=z[i];rp=native[k].double().log_softmax(0);lp=zz.double().log_softmax(0);ans=e['groups'][e['variant']][0]
     if mode=='self':replay.append(float((zz-native[k]).abs().max()))
     result['interventions'].append(dict(id=k,wording=e['wording'],component=component,mode=mode,target_interaction_ratio=float((I+shift).norm()/I.norm()),interaction_alignment_removed=float(-(shift@I)/I.square().sum()),reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce_delta=float(rp[ans]-lp[ans]),full_vocab_correct=measure(zz,e['groups'])['top1'][e['variant']]))
  save();print('COMPLETE',component,mode,flush=True)
 for w in [0,1]:
  for component,mode in conditions[1:]:
   vv=[c for c in result['interventions'] if c['wording']==w and c['component']==component and c['mode']==mode];result['summary'][f'{w}_{component}_{mode}']={n:float(np.mean([c[n] for c in vv])) for n in ['target_interaction_ratio','interaction_alignment_removed','reference_kl','target_ce_delta','full_vocab_correct']}
 result['audits']=dict(runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),native_prompts=len(ex),interventions=len(result['interventions']),identity_replay_max_logit_error=max(replay),native_write_replay_max_relative_error=max(replays),native_interaction_replay_max_relative_error=max(native_interaction_errors),component_sum_max_relative_error=max(sum_errors),bus_value_interaction_max_norm=max(g['bus_value_interaction_norm'] for g in geometry),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_interaction_ratio','reference_kl','target_ce_delta']));torch.save(dict(aligned=aligned,components=components,targets=targets,native=native),ART/f'{STEM}_captures.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
