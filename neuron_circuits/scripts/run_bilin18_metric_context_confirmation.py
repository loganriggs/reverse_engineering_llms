"""H602 frozen broad basis: new wording and nuisance versus answer-changing contexts."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,measure,continuation_id
from run_bilin18_diverse_writer_transfer import suite as old_suite
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_metric_context_confirmation'
def suite():
 rows=[]
 for r in old_suite():
  if r['template']!=0:continue
  fam=r['family'];a,b=r['answers'];p=r['prompts'];i=r['item']
  if fam=='sequence_copy':texts=[f'A package arrived for {x}. A package arrived for {x}. A package arrived for' for x in [a,b]]
  elif fam=='entity_binding':
   # The original answers are recipient names; each giver is the opposite answer.
   texts=[f'{b} and {a} met at the museum. {x} offered a ticket to' for x in [b,a]]
  elif fam=='agreement':
   nouns=[('actor','actors'),('pilot','pilots'),('nurse','nurses'),('lawyer','lawyers'),('chef','chefs'),('guard','guards'),('dancer','dancers'),('painter','painters')][i]
   texts=[f'The {x} close to the gate usually' for x in nouns]
  elif fam=='capital_recall':
   cs=['France','Germany','Italy','Japan','Spain','China','Russia','England'];texts=[f'The city that serves as the capital of {cs[ii]} is' for ii in [i,(i+1)%8]]
  else:
   adjs=[('wonderful','terrible'),('excellent','awful'),('delightful','dreadful'),('enjoyable','boring'),('great','bad'),('amazing','horrible'),('superb','poor'),('fantastic','disappointing')][i]
   texts=[f'Review: The concert felt {x}. Sentiment:' for x in adjs]
  nuisance=['For the record, '+x[0].lower()+x[1:] for x in texts]
  rows.append(dict(r,id='confirm-'+r['id'],prompts=texts+nuisance))
 return rows
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();bp=ART/'bilin18_metric_coverage_factorial_locked_maps.pt';npth=ART/'bilin18_agreement_write_subspace_locked_bases.pt';hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [bp,npth,ROOT/'scripts/run_bilin18_diverse_writer_transfer.py',ROOT/f'scripts/run_{STEM}.py']};f=torch.load(bp,weights_only=False);A=f['bases']['broad'];K=f['maps']['broad','number'];KM=f['maps']['broad','multi'];b=torch.load(npth,weights_only=False);axis=b['pca'][0]*torch.sign(b['pca'][0]@b['mean']);size=float(axis@b['mean'])*.0625;gen=torch.Generator().manual_seed(600);random=torch.randn(axis.shape,generator=gen);random/=random.norm();rows=suite();mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();prior=set()
 for stem in ['bilin18_diverse_writer_transfer','bilin18_task_boundary_factorial']:
  for rr in json.loads((ART/f'{stem}_manifest.json').read_text()):prior.update(rr['prompts'])
 overlap=sum(p in prior for r in rows for p in r['prompts']);assert overlap==0
 cfg=dict(experiment='H602',hashes=hashes,signal_norm=size,signals=['number','task','random'],signs=[-1,1],seed=600,design='Freeze H601 broad rank64 basis and number/multi maps. New wording for five tasks, eight pairs each, same lexical inventory. Compare same-answer prefixed context versus opposite-answer context with three fixed-norm signals and both signs. No fit or per-item filtering.',gate='Per task/template >=4 pairs, >=.75 both-endpoint candidate correctness and >=.5 intended-group full-vocab top1. Nuisance competence separately reported.',scope='New exact prompt strings against H581/H582 manifests, familiar lexical inventory. Context prefixed with For the record versus opposite-answer native input. Same-answer context changes differ by task and are not proven semantically inert. Equal Attention5 norms do not imply equal downstream e. One random direction only. Single-token CE uses explicit intended answer, group NLL separately. No full-model or canonical uniqueness claim.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex={};excluded=[]
 for r in rows:
  ee=[]
  for v,text in enumerate(r['prompts']):
   gs=[[continuation_id(tok,text,t) for t in g] for g in groups(r['family'],r['answers'])];gs=[[t for t in g if t is not None] for g in gs];target=continuation_id(tok,text,' '+r['answers'][v%2])
   if any(not g for g in gs) or set(gs[0])&set(gs[1]) or target is None:break
   ee.append(dict(r,variant=v,ids=tok.encode(text,add_special_tokens=False),groups=gs,target=target))
  if len(ee)!=4:excluded.append(r['id']);continue
  for e in ee:ex[e['id'],e['variant']]=e
 model=load_full_model('cuda');assert not model.config.gated
 for m in [model.embedding,model.blocks,model.lm_head]:
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 mlp=model.blocks[11].mlp;native={};inputs={};writes={};attn={};batches=defaultdict(list)
 for k,e in ex.items():batches[len(e['ids'])].append(k)
 def forward(bb,delta=None,capture=False,replacement=None):
  cap={};hs=[]
  if capture:
   def pre(m,args):cap['input']=args[0][:,-1].cpu()
   def post(m,args,out):cap['write']=out[:,-1].cpu()
   def ah(m,args,out):cap['attn']=out[0][:,-1].cpu()
   hs=[mlp.register_forward_pre_hook(pre),mlp.register_forward_hook(post),model.blocks[5].attn.register_forward_hook(ah)]
  if replacement is not None:
   def hook(m,args,out):y=out.clone();y[:,-1]=replacement.to(y);return y
   hs.append(mlp.register_forward_hook(hook))
  try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in bb],device='cuda'),patch=None if delta is None else (5,delta))
  finally:
   for h in hs:h.remove()
  return z.cpu(),cap
 for kk in batches.values():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];z,c=forward(bb,capture=True)
   for i,k in enumerate(bb):native[k]=z[i];inputs[k]=c['input'][i];writes[k]=c['write'][i];attn[k]=c['attn'][i]
 competence={}
 for fam in sorted({r['family'] for r in rows}):
  for t in (0,):
   rr=[r for r in rows if r['family']==fam and r['template']==t and (r['id'],0) in ex];mm={(r['id'],v):measure(native[r['id'],v],ex[r['id'],v]['groups']) for r in rr for v in range(4)};both=float(np.mean([mm[r['id'],0]['margin']>0 and mm[r['id'],1]['margin']<0 for r in rr]));top=float(np.mean([mm[r['id'],v]['top1'][v] for r in rr for v in (0,1)]));nt=float(np.mean([mm[r['id'],v+2]['top1'][v] for r in rr for v in (0,1)]));competence[f'{fam}_{t}']=dict(pairs=len(rr),both_endpoint_choice=both,full_vocab_top1=top,nuisance_full_vocab_top1=nt,passes=len(rr)>=4 and both>=.75 and top>=.5)
 print('COMPETENCE',json.dumps(competence),flush=True)
 def T(c,e):
  c=c.cuda();e=e.cuda();return (mlp.Down(mlp.Left(c)*mlp.Right(e)+mlp.Left(e)*mlp.Right(c))-mlp.Down(torch.zeros_like(mlp.Left(c)*mlp.Right(e)))).cpu()
 jobs=defaultdict(list);deltas={};norms=[]
 for k,e in ex.items():
  if k[1]>1:continue
  task=attn[k[0],1-k[1]]-attn[k];assert task.norm()>1e-8;task=task/task.norm()
  for signal,vec in [('number',axis),('task',task),('random',random)]:
   for sign in [-1,1]:
    for context in ['nuisance','answer_change']:
     j=(k,signal,sign,context);deltas[j]=vec*size*sign;norms.append(float(deltas[j].norm()));jobs[len(e['ids'])].append(j)
 cache={};vectors={};geometry=[];modes=['full','native','projection','metric','multi','oracle'];identity=[]
 for jj in jobs.values():
  for st in range(0,len(jj),16):
   bb=jj[st:st+16];z,c=forward([j[0] for j in bb],torch.stack([deltas[j] for j in bb]),True)
   for i,j in enumerate(bb):
    k,signal,sign,context=j;de=c['input'][i]-inputs[k];dc=inputs[k[0],k[1]+2 if context=='nuisance' else 1-k[1]]-inputs[k];full=T(dc,de);vv={'full':full,'native':torch.zeros_like(full),'projection':T(A.T@(A@dc),de),'metric':T(A.T@(K@dc),de),'multi':T(A.T@(KM@dc),de)};M=T(A,de).T.double();U,S,Vh=torch.linalg.svd(M,full_matrices=False);keep=S>S[0]*1e-10;coef=Vh[keep].T@((U[:,keep].T@full.double())/S[keep]);pred=M@coef;vv['oracle']=pred.float();err=M.T@(pred-full.double());geometry.append(dict(id=k[0],variant=k[1],signal=signal,sign=sign,normal_error=float(err.norm()/(M.norm()*full.norm())),response_norm=float(full.norm()),input_change_norm=float(de.norm()),context_change_norm=float(dc.norm())))
    # Independent finite-difference identity including any affine biases.
    zz=inputs[k].cuda();dd=dc.cuda();ee=de.cuda();actual=(mlp(zz+dd+ee)-mlp(zz+dd))-(mlp(zz+ee)-mlp(zz));identity.append(float((actual.cpu()-full).norm()/full.norm()))
    cache[j]=dict(logits=z[i],write=c['write'][i],de=de,dc=dc)
    vectors[j]=vv
 print('CAPTURED',len(cache),'signals; interventions',len(cache)*len(modes),flush=True)
 result=dict(config=cfg,competence=competence,excluded=excluded,geometry=geometry,interventions=[],summary={},status='running');refs={};replay=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save()
 for mode in modes:
  for jj in jobs.values():
   for st in range(0,len(jj),16):
    bb=jj[st:st+16];rep=torch.stack([cache[j]['write']+vectors[j][mode] for j in bb]);z,_=forward([j[0] for j in bb],torch.stack([deltas[j] for j in bb]),replacement=rep)
    for i,j in enumerate(bb):
     k,signal,sign,context=j;e=ex[k];zz=z[i]
     if mode=='full':refs[j]=zz
     if mode=='native':replay.append(float((zz-cache[j]['logits']).abs().max()))
     ref=refs[j];lp=zz.double().log_softmax(0);rp=ref.double().log_softmax(0);margin=measure(zz,e['groups'])['margin'];rm=measure(ref,e['groups'])['margin'];target=e['target'];ans=e['groups'][k[1]];full=vectors[j]['full'];result['interventions'].append(dict(id=k[0],variant=k[1],family=e['family']+'_'+context,context=context,template=e['template'],signal=signal,sign=sign,mode=mode,margin_error=margin-rm,reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce=float(-lp[target]),target_ce_error=float(rp[target]-lp[target]),group_nll_error=float(torch.logsumexp(rp[ans],0)-torch.logsumexp(lp[ans],0)),internal_response_error=float((vectors[j][mode]-full).norm()/full.norm())))
  save();print('COMPLETE',mode,flush=True)
 for fam in sorted({r['family']+'_'+context for r in rows for context in ['nuisance','answer_change']}):
  for t in (0,):
   for signal in cfg['signals']:
    cc=[c for c in result['interventions'] if c['family']==fam and c['template']==t and c['signal']==signal];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
    for mode in modes:
     vv=[c for c in cc if c['mode']==mode];result['summary'][f'{fam}_{t}_{signal}_{mode}']=dict(n=len(vv),internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),margin_abs_error=float(np.mean([abs(c['margin_error']) for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),target_ce_abs_error=float(np.mean([abs(c['target_ce_error']) for c in vv])),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd),target_ce=float(np.mean([c['target_ce'] for c in vv])))
 result['audits']=dict(frozen_files_unchanged=all(hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest(),refit=False,checked_prior_prompt_overlap=overlap,native_prompt_count=len(ex),capture_count=len(cache),intervention_count=len(result['interventions']),reference_replay_max_logit_error=max(replay),finite_context_identity_max_relative_error=max(identity),oracle_normal_equation_max_relative_error=max(g['normal_error'] for g in geometry),signal_norm_min=min(norms),signal_norm_max=max(norms),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['reference_kl','margin_error','internal_response_error','target_ce']))
 torch.save(dict(examples=ex,inputs=inputs,writes=writes,attn=attn,native=native,cache=cache,vectors=vectors),ART/f'{STEM}_captures.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
