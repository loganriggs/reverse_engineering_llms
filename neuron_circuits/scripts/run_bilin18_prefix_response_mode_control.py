"""H619 explicit-prefix and response-format control."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_response_mode_control'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time()
 from transformers import AutoTokenizer
 from audit_bilin18_mlp_discriminators import TOKENIZER
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 from run_bilin18_diverse_writer_transfer import suite
 from run_bilin18_diverse_causal_tasks import groups as answer_groups
 families=['sequence_copy','agreement','capital_recall'];rows=[];token_exclusions=[]
 for r in suite():
  if r['family'] not in families:continue
  gg=[[tok.encode(text,add_special_tokens=False)[0] for text in group] for group in answer_groups(r['family'],r['answers'])]
  assert all(len(tok.encode(text,add_special_tokens=False))==1 for group in answer_groups(r['family'],r['answers']) for text in group)
  for form in [0,1]:
   w=families.index(r['family'])*4+r['template']*2+form
   for v,original in enumerate(r['prompts'][:2]):
    original=original+('\nNext word:' if form else '')
    for p in [0,1]:
     for c in [0,1]:
      body=original[0].lower()+original[1:] if c else original;prompt=('For the record, ' if p else 'As a reminder, ')+body
      rows.append(dict(id=f"{w}-{r['item']}-{v}-{p}-{c}",wording=w,family=r['family'],template=r['template'],format=form,item=r['item'],variant=v,prefix=p,lower=c,prompt=prompt,ids=tok.encode(prompt,add_special_tokens=False),groups=gg))
 generic=[tok.encode(t,add_special_tokens=False)[0] for t in [' the',' a',' not']]
 mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));ex={r['id']:r for r in rows};index={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows}
 for w in range(12):
  for item in range(8):
   for v in [0,1]:
    for c in [0,1]:assert ex[index[w,item,v,1,c]]['ids'][4:]==ex[index[w,item,v,0,c]]['ids'][4:]
 for w in range(12):
  for item in range(8):
   for v in [0,1]:
    for p in [0,1]:assert len(ex[index[w,item,v,p,0]]['ids'])==len(ex[index[w,item,v,p,1]]['ids'])
 bp=ART/'bilin18_prefix_joint_conditional_locked_bases.pt';bh=hashlib.sha256(bp.read_bytes()).hexdigest();bases=torch.load(bp,weights_only=False)['bases']
 cfg=dict(experiment='H619',runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),basis_sha256=bh,token_exclusions=token_exclusions,families=families,design='Two four-token explicit prefixes As a reminder versus For the record, crossed with initial case. Three tasks, two templates, free continuation versus appended Next word label. Frozen joint64 span/complement/full removals at .25 and1; no output filtering.',scope='Changing prefix contrast changes interaction, not merely its magnitude. Answer suffix also changes positions and wording; not pure response-mode isolation. Compare factual choice and generic-token competition separately.')
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
  aligned[k]=dict(R=R[:,4:],local=local,bus=bus)
 assert max(replays)<1e-5
 components={};targets={};combined=[];geometry=[];sum_errors=[];native_interaction_errors=[]
 for w in range(12):
  for item in range(8):
   for v in [0,1]:
    ids=[index[w,item,v,p,c] for p,c in [(0,0),(0,1),(1,0),(1,1)]];assert ex[ids[2]]['ids'][4:]==ex[ids[0]]['ids'][4:];k=ids[3];combined.append(k);fs=[features[ii] for ii in ids];aa=[aligned[ii] for ii in ids];rr=[a['R'].double() for a in aa];ri=rr[3]-rr[2]-rr[1]+rr[0];vv={};busvi=None
    for stream,weight in [('local',1-lam),('bus',lam)]:
     vals=[a[stream].double() for a in aa];vi=vals[3]-vals[2]-vals[1]+vals[0];outputs=[apply(r,x)*weight for r,x in zip(rr,vals)];full=outputs[3]-outputs[2]-outputs[1]+outputs[0];routing=apply(ri,vals[0])*weight;value=apply(rr[0],vi)*weight;mixed=full-routing-value;vv[stream]=full;vv[stream+'_routing']=routing;vv[stream+'_value']=value;vv[stream+'_mixed']=mixed
     if stream=='bus':busvi=float(vi.norm())
    vv['full']=vv['local']+vv['bus'];vv['routing']=vv['local_routing']+vv['bus_routing'];vv['value']=vv['local_value']+vv['bus_value'];vv['mixed']=vv['local_mixed']+vv['bus_mixed'];nf=fs[3]['write'][4:]-fs[2]['write'][4:]-fs[1]['write'][4:]+fs[0]['write'][4:];native_interaction_errors.append(float((vv['full']-nf.double()).norm()/nf.double().norm()));sum_errors.append(float((vv['routing']+vv['value']+vv['mixed']-vv['full']).norm()/vv['full'].norm()));geometry.append(dict(id=k,wording=w,norms={n:float(x.norm()) for n,x in vv.items()},local_bus_cosine=float(F.cosine_similarity(vv['local'].flatten(),vv['bus'].flatten(),dim=0)),routing_value_cosine=float(F.cosine_similarity(vv['routing'].flatten(),vv['value'].flatten(),dim=0)),bus_value_interaction_norm=busvi));components[k]={n:x.float() for n,x in vv.items()};targets[k]=fs[3]['target']-fs[2]['target']-fs[1]['target']+fs[0]['target']
 assert max(native_interaction_errors)<1e-4
 competence={}
 for w in range(12):
  for p in [0,1]:
   for c in [0,1]:
    both=[];top=[]
    for item in range(8):
     mm=[measure(native[index[w,item,v,p,c]],ex[index[w,item,v,p,c]]['groups']) for v in [0,1]];both.append(mm[0]['margin']>0 and mm[1]['margin']<0);top.extend(mm[v]['top1'][v] for v in [0,1])
    competence[f'{w}_{p}_{c}']=dict(both_endpoint_choice=float(np.mean(both)),full_vocab_top1=float(np.mean(top)),passes=bool(np.mean(both)>=.75 and np.mean(top)>=.5))
 print('CAPTURED',max(replays),competence,flush=True)
 jobs=defaultdict(list)
 for k in combined:jobs[len(ex[k]['ids'])].append(k)
 modes=['native']+[f'{name}_{dose}' for dose in [.25,1.] for name in ['full','span','complement']];refs={};result=dict(config=cfg,competence=competence,interventions=[],summary={},status='running')
 def vector(k,mode):
  full=components[k]['local'];P=(full@bases['joint'][:64].T)@bases['joint'][:64]
  if mode=='native':return torch.zeros_like(full)
  name,dose=mode.split('_');return float(dose)*({'full':full,'span':P,'complement':full-P}[name])
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save();identity=[]
 for mode in modes:
  for kk in jobs.values():
   for st in range(0,len(kk),16):
    bb=kk[st:st+16];delta=torch.stack([vector(k,mode) for k in bb]);z,c=forward(bb,delta=delta)
    for i,k in enumerate(bb):
     e=ex[k];zz=z[i];tt=c['target'][i]
     if mode=='native':refs[k]=dict(logits=zz,target=tt)
     if mode=='native':identity.append(float((zz-native[k]).abs().max()))
     rp=refs[k]['logits'].double().log_softmax(0);lp=zz.double().log_softmax(0);full=components[k]['local'];ans=e['groups'][e['variant']][0];den=targets[k].norm();baseI=targets[k]
     result['interventions'].append(dict(id=k,wording=e['wording'],item=e['item'],mode=mode,write_norm=float(full.norm()),candidate_correct=bool((measure(zz,e['groups'])['margin']>0) if e['variant']==0 else (measure(zz,e['groups'])['margin']<0)),direct_answer_mass=float(lp[e['groups'][e['variant']]].exp().sum()),answer_generic_margin=float(torch.logsumexp(zz[e['groups'][e['variant']]].double(),0)-torch.logsumexp(zz[generic].double(),0)),top_token=tok.decode([int(zz.argmax())]),write_relative_error=float((vector(k,mode)-full).norm()/full.norm()),target_relative_error=float((tt-refs[k]['target']).norm()/den),target_effect_norm=float(den),target_interaction_ratio=float((baseI+tt-features[k]['target']).norm()/baseI.norm()),reference_kl=float((rp.exp()*(rp-lp)).sum()),native_kl=float((native[k].double().softmax(0)*(native[k].double().log_softmax(0)-lp)).sum()),group_nll_error=float(torch.logsumexp(rp[e['groups'][e['variant']]],0)-torch.logsumexp(lp[e['groups'][e['variant']]],0)),margin_error=measure(zz,e['groups'])['margin']-measure(refs[k]['logits'],e['groups'])['margin'],target_ce_error=float(rp[ans]-lp[ans]),full_vocab_correct=measure(zz,e['groups'])['top1'][e['variant']]))
  save();print('COMPLETE',mode,flush=True)
 for w in range(12):
  for mode in modes:
   vv=[r for r in result['interventions'] if r['wording']==w and r['mode']==mode];result['summary'][f'{w}_{mode}']={n:float(np.mean([v[n] for v in vv])) for n in ['native_kl','write_norm','candidate_correct','direct_answer_mass','answer_generic_margin','full_vocab_correct','target_relative_error','target_ce_error']}
 old=json.loads((ART/'bilin18_prefix_boundary_control_manifest.json').read_text());oldprompts={r['prompt'] for r in old}
 result['audits']=dict(runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=cfg['manifest_sha256']==hashlib.sha256(mp.read_bytes()).hexdigest(),bases_unchanged=bh==hashlib.sha256(bp.read_bytes()).hexdigest(),native_prompts=len(rows),interventions=len(result['interventions']),native_write_replay_max_relative_error=max(replays),native_interaction_replay_max_relative_error=max(native_interaction_errors),identity_replay_max_logit_error=max(identity),prior_prompt_overlap={w:sum(r['prompt'] in oldprompts for r in rows if r['wording']==w) for w in range(12)},finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['target_relative_error','reference_kl','write_relative_error']))
 torch.save(dict(components={k:v['local'] for k,v in components.items()},targets=targets,native=native),ART/f'{STEM}_captures.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
