"""H607 copying prefix-by-capitalization factorial and component transfers."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure,continuation_id
from run_bilin18_diverse_writer_transfer import suite as old_suite
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_prefix_case_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();rows=[];tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 for r in old_suite():
  if r['family']!='sequence_copy' or r['template']!=0:continue
  for wording in [0,1]:
   for v,name in enumerate(r['answers']):
    base=r['prompts'][v] if wording==0 else f'A letter was delivered to {name}. A letter was delivered to {name}. A letter was delivered to'
    for prefix in [0,1]:
     for lower in [0,1]:
      body=base[0].lower()+base[1:] if lower else base;text=('For the record, ' if prefix else '')+body;gs=[[continuation_id(tok,text,' '+a)] for a in r['answers']];assert all(g[0] is not None for g in gs);rows.append(dict(id=f'{wording}-{r["item"]}-{v}-{prefix}-{lower}',wording=wording,item=r['item'],variant=v,prefix=prefix,lower=lower,prompt=text,ids=tok.encode(text,add_special_tokens=False),groups=gs))
 mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();cfg=dict(experiment='H607',design='Copying old/new wording, eight name pairs, two endpoints, prefix presence crossed with initial-character lowercase. Native MLP11 contexts z00/z01/z10/z11 define case=z01-z00, prefix=z10-z00, interaction=z11-z10-z01+z00. Test local full response and components under fixed-norm number/task/random Attention5 signals, both signs. Cross-format component replacements use same name/endpoint. No fitting or gate filtering.',scope='Prefix-only retains uppercase initial word after comma. Case factor is a tokenization/orthography change, not a clean semantic variable. Same lexical inventory and known formats; new intervention decomposition. Behavior effects are not additive shares.',signal_norm=41.81718826293945,seed=600,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));ex={r['id']:r for r in rows};index={(r['wording'],r['item'],r['variant'],r['prefix'],r['lower']):r['id'] for r in rows};model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 m=model.blocks[11].mlp;native={};inputs={};attn={};batches=defaultdict(list)
 for k,e in ex.items():batches[len(e['ids'])].append(k)
 def forward(kk,delta=None,rep=None,capture=False):
  cap={};hs=[]
  if capture:
   def pre(m,args):cap['input']=args[0][:,-1].cpu()
   def post(m,args,out):cap['write']=out[:,-1].cpu()
   def ah(m,args,out):cap['attn']=out[0][:,-1].cpu()
   hs=[m.register_forward_pre_hook(pre),m.register_forward_hook(post),model.blocks[5].attn.register_forward_hook(ah)]
  if rep is not None:
   def h(m,args,out):y=out.clone();y[:,-1]=rep.to(y);return y
   hs.append(m.register_forward_hook(h))
  try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in kk],device='cuda'),patch=None if delta is None else (5,delta))
  finally:
   for h in hs:h.remove()
  return z.cpu(),cap
 for kk in batches.values():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];z,c=forward(bb,capture=True)
   for i,k in enumerate(bb):native[k]=z[i];inputs[k]=c['input'][i];attn[k]=c['attn'][i]
 competence={}
 for w in [0,1]:
  for p in [0,1]:
   for lo in [0,1]:
    both=[];top=[]
    for item in range(8):
     mm=[measure(native[index[w,item,v,p,lo]],ex[index[w,item,v,p,lo]]['groups']) for v in [0,1]];both.append(mm[0]['margin']>0 and mm[1]['margin']<0);top.extend(mm[v]['top1'][v] for v in [0,1])
    competence[f'{w}_{p}_{lo}']=dict(pairs=8,both_endpoint_choice=float(np.mean(both)),full_vocab_top1=float(np.mean(top)),passes=bool(np.mean(both)>=.75 and np.mean(top)>=.5))
 print('COMPETENCE',competence,flush=True);components={};contextgeometry=[]
 for w in [0,1]:
  for item in range(8):
   for v in [0,1]:
    ids=[index[w,item,v,p,lo] for p,lo in [(0,0),(0,1),(1,0),(1,1)]];z00,z01,z10,z11=[inputs[k] for k in ids];components[ids[0]]=dict(total=z11-z00,case=z01-z00,prefix=z10-z00,interaction=z11-z10-z01+z00)
 for k,cs in components.items():
  e=ex[k];other=index[1-e['wording'],e['item'],e['variant'],0,0];oc=components[other];contextgeometry.append(dict(id=k,wording=e['wording'],norms={n:float(v.norm()) for n,v in cs.items()},transfer_error={n:float((v-oc[n]).norm()/v.norm()) for n,v in cs.items()},cosine={n:float(torch.nn.functional.cosine_similarity(v,oc[n],dim=0)) for n,v in cs.items()}))
 nb=torch.load(ART/'bilin18_agreement_write_subspace_locked_bases.pt',weights_only=False);axis=nb['pca'][0]*torch.sign(nb['pca'][0]@nb['mean']);size=float(axis@nb['mean'])*.0625;assert abs(size-cfg['signal_norm'])<1e-4;rand=torch.randn(1152,generator=torch.Generator().manual_seed(600));rand/=rand.norm();deltas={};jobs=defaultdict(list)
 for k in components:
  e=ex[k];other=index[e['wording'],e['item'],1-e['variant'],0,0];task=attn[other]-attn[k];task/=task.norm()
  for signal,vec in [('number',axis),('task',task),('random',rand)]:
   for sign in [-1,1]:j=(k,signal,sign);deltas[j]=vec*size*sign;jobs[len(e['ids'])].append(j)
 def T(c,e):
  c=c.cuda();e=e.cuda();v=m.Left(c)*m.Right(e)+m.Left(e)*m.Right(c);return (m.Down(v)-m.Down(torch.zeros_like(v))).cpu()
 modes=['full','native','case_only','prefix_only','additive','interaction_only','swap_case','swap_prefix','swap_interaction','swap_total'];vectors={};cache={};identities=[];responsegeometry=[]
 for jj in jobs.values():
  for st in range(0,len(jj),16):
   bb=jj[st:st+16];z,c=forward([j[0] for j in bb],torch.stack([deltas[j] for j in bb]),capture=True)
   for i,j in enumerate(bb):
    k,signal,sign=j;e=ex[k];de=c['input'][i]-inputs[k];cs=components[k];oc=components[index[1-e['wording'],e['item'],e['variant'],0,0]];rr={n:T(v,de) for n,v in cs.items()};full=rr['total'];vv=dict(full=full,native=torch.zeros_like(full),case_only=rr['case'],prefix_only=rr['prefix'],additive=rr['case']+rr['prefix'],interaction_only=rr['interaction'],swap_total=T(oc['total'],de))
    for term in ['case','prefix','interaction']:vv['swap_'+term]=full-rr[term]+T(oc[term],de)
    identities.append(float((full-rr['case']-rr['prefix']-rr['interaction']).norm()/full.norm()));vectors[j]=vv;cache[j]=dict(logits=z[i],write=c['write'][i],de=de);responsegeometry.append(dict(id=k,signal=signal,sign=sign,norms={n:float(v.norm()) for n,v in rr.items()}))
 result=dict(config=cfg,competence=competence,context_geometry=contextgeometry,response_geometry=responsegeometry,interventions=[],summary={},status='running');refs={};replay=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save();print('INTERVENTIONS',len(vectors)*len(modes),flush=True)
 for mode in modes:
  for jj in jobs.values():
   for st in range(0,len(jj),16):
    bb=jj[st:st+16];rep=torch.stack([cache[j]['write']+vectors[j][mode] for j in bb]);z,_=forward([j[0] for j in bb],torch.stack([deltas[j] for j in bb]),rep=rep)
    for i,j in enumerate(bb):
     k,signal,sign=j;e=ex[k];zz=z[i]
     if mode=='full':refs[j]=zz
     if mode=='native':replay.append(float((zz-cache[j]['logits']).abs().max()))
     rp=refs[j].double().log_softmax(0);lp=zz.double().log_softmax(0);f=vectors[j]['full'];target=e['groups'][e['variant']][0];result['interventions'].append(dict(id=k,wording=e['wording'],signal=signal,sign=sign,mode=mode,margin_error=measure(zz,e['groups'])['margin']-measure(refs[j],e['groups'])['margin'],reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce_error=float(rp[target]-lp[target]),internal_response_error=float((vectors[j][mode]-f).norm()/f.norm())))
  save();print('COMPLETE',mode,flush=True)
 for w in [0,1]:
  for signal in ['number','task','random']:
   cc=[c for c in result['interventions'] if c['wording']==w and c['signal']==signal];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
   for mode in modes:
    vv=[c for c in cc if c['mode']==mode];result['summary'][f'{w}_{signal}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd))
 result['audits']=dict(runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest(),native_prompts=len(ex),perturbed_captures=len(cache),interventions=len(result['interventions']),replay_max_logit_error=max(replay),component_sum_max_relative_error=max(identities),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['reference_kl','margin_error','internal_response_error','target_ce_error']));torch.save(dict(examples=ex,inputs=inputs,native=native,components=components,cache=cache,vectors=vectors),ART/f'{STEM}_captures.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
