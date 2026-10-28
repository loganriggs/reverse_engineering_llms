"""H558: fit context PCA on lexical discovery blocks; test response and KL fidelity."""
import json,hashlib,time
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure,continuation_id
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_context_coverage_curve'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_context_span_oracle.json';saved=torch.load(ART/'bilin18_position_vocabulary_control_native_cache.pt',weights_only=False);rows=saved['rows'];inputs=torch.load(ART/'bilin18_mlp_response_decomposition_inputs.pt',weights_only=False);train=[r for r in rows if int(r['id'].rsplit('-',2)[1])<4];fresh=torch.load(ART/'bilin18_response_weighted_fresh_inputs.pt',weights_only=False);test=fresh['rows'];inputs.update(fresh['inputs']);tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);extra=[]
 names=[('John','Mary'),('Alice','Bob'),('Carol','Dave'),('Sarah','Tom'),('Peter','Jane'),('James','Anna'),('David','Kate'),('Mark','Lucy')];words=[('apple','banana'),('table','chair'),('dog','cat'),('book','paper'),('stone','glass'),('music','water'),('coffee','bread'),('sun','moon')]
 for i,((a,b),(v,w)) in enumerate(zip(names,words)):
  for fam,prompts,answers in [('word_prose',[f'The code was {q}. The code was {q}. The code was' for q in (v,w)],[v,w]),('word_record',[f'record {q} | record {q} | record' for q in (v,w)],[v,w]),('name_prose',[f'The code was {q}. The code was {q}. The code was' for q in (a,b)],[a,b]),('name_record',[f'record {q} | record {q} | record' for q in (a,b)],[a,b]),('binding',[f'{a} and {b} entered the room. {g} passed the note to' for g in (a,b)],[b,a])]:
   ids=[tok.encode(t,add_special_tokens=False) for t in prompts];assert len(ids[0])==len(ids[1])
   for v in (0,1):
    gs=[[continuation_id(tok,prompts[v],' '+ans)] for ans in answers];assert all(g[0] is not None for g in gs);extra.append(dict(id=f'extra-{fam}-{i}-{v}',family=fam,variant=v,prompt=prompts[v],ids=ids[v],groups=gs))
 evaltokens={t for r in test for g in r['groups'] for t in g};traintokens={t for r in rows+extra for g in r['groups'] for t in g};assert not evaltokens&traintokens
 cfg=dict(experiment='H563',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),train_items=[0,1,2,3],test_items=[4,5,6,7],coverage=[40,80,160],global_ranks=[16,32],scales=[.0625,1.],design='Nested context coverage 40 original discovery, 80 original panel, 160 after adding 80 new name/common-word prompts. Fixed ranks16/32, fixed H561 evaluation prompts, no evaluation candidate tokens in fitting. Coverage expansion changes lexical diversity as well as sample size; not an iid scaling law.',metrics=['relative internal response error','answer margin MAE','KL(exact perturbed distribution || approximate perturbed distribution)','KL ratio to zero MLP response baseline'],scope='Retrospective split on previously inspected prompts; withheld from fitting, not prospective fresh evidence. Only local context modulation compressed, not full MLP or downstream circuit. PCA optimizes context reconstruction, not response or KL.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 m=model.blocks[8].mlp;block=model.blocks[8];extra_native=[]
 (ART/f'{STEM}_extra_manifest.json').write_text(json.dumps(extra,indent=2))
 for r in extra:
  cap={}
  def pre(mod,args):cap['x']=args[0][:,-1].clone();cap['x0']=args[2][:,-1].clone()
  def ah(mod,args,out):cap['y']=out[0][:,-1].clone()
  handles=[block.register_forward_pre_hook(pre),block.attn.register_forward_hook(ah)]
  try:zz,_=run(model,torch.tensor([r['ids']],device='cuda'))
  finally:
   for handle in handles:handle.remove()
  inputs[r['id']]=dict(x=(block.lambdas[0]*cap['x']+block.lambdas[1]*cap['x0']+cap['y'])[0].cpu());extra_native.append(dict(id=r['id'],family=r['family'],variant=r['variant'],**measure(zz[0],r['groups'])))
 z={k:F.rms_norm(v['x'].cuda(),(v['x'].numel(),)) for k,v in inputs.items()};bases={}
 for count,rr in [(40,train),(80,rows),(160,rows+extra)]:
  mat=torch.stack([z[r['id']] for r in rr]).double();mean=mat.mean(0);_,sv,V=torch.linalg.svd(mat-mean,full_matrices=False);bases[str(count)]=dict(mean=mean.float(),V=V[:count-1].float(),singular_values=sv.cpu(),fit_ids=[r['id'] for r in rr]);assert not(set(bases[str(count)]['fit_ids'])&{r['id'] for r in test})
 bp=ART/f'{STEM}_bases.pt';torch.save(bases,bp);bh=hashlib.sha256(bp.read_bytes()).hexdigest();print('BASES LOCKED',len(train),len(test),flush=True)
 result=dict(config=cfg,extra_native=extra_native,bases_sha256=bh,geometry=[],interventions=[],status='running');jobs=defaultdict(list);exactlog={};nativekl={};scored={}
 def kl(a,b):
  la=a.double().log_softmax(-1);lb=b.double().log_softmax(-1);return float((la.exp()*(la-lb)).sum())
 for r in test:
  x=inputs[r['id']]['x'].cuda();base=z[r['id']];y0=m(base);contexts={}
  for count in cfg['coverage']:
   b=bases[str(count)];center=base-b['mean']
   for rank in cfg['global_ranks']:V=b['V'][:rank];contexts[f'coverage{count}_rank{rank}']=b['mean']+(center@V.T)@V
  ids=torch.tensor([r['ids']],device='cuda');nat,_=run(model,ids)
  for scale in cfg['scales']:
   delta=scale*inputs[r['id']]['delta'].cuda();u=F.rms_norm(x+delta,(x.numel(),));e=u-base;actual=m(u)-y0;quad=m.Down(m.Left(e)*m.Right(e));responses=dict(exact=actual,zero=torch.zeros_like(actual))
   for mode,c in contexts.items():responses[mode]=m.Down(m.Left(c)*m.Right(e)+m.Left(e)*m.Right(c))+quad
   logits,_=run(model,ids,patch=(8,delta[None]));exactlog[r['id'],scale]=logits[0].cpu();nativekl[r['id'],scale]=kl(logits[0],nat[0]);scored[r['id'],scale]=measure(logits[0],r['groups'])['margin']
   for mode,response in responses.items():
    result['geometry'].append(dict(id=r['id'],family=r['family'],scale=scale,mode=mode,relative_response_error=float((response-actual).norm()/actual.norm())));jobs[len(r['ids'])].append((r,scale,mode,delta.cpu(),(y0+response).cpu()))
 for T,jj in jobs.items():
  for begin in range(0,len(jj),24):
   bb=jj[begin:begin+24]
   def hook(mod,args,out):out=out.clone();out[:,-1]=torch.stack([y for r,s,mode,d,y in bb]).to(out);return out
   handle=m.register_forward_hook(hook)
   try:zz,_=run(model,torch.tensor([r['ids'] for r,s,mode,d,y in bb],device='cuda'),patch=(8,torch.stack([d for r,s,mode,d,y in bb])))
   finally:handle.remove()
   for ix,(r,scale,mode,d,y) in enumerate(bb):result['interventions'].append(dict(id=r['id'],family=r['family'],scale=scale,mode=mode,margin_error=measure(zz[ix],r['groups'])['margin']-scored[r['id'],scale],kl=kl(exactlog[r['id'],scale].cuda(),zz[ix]),exact_to_native_kl=nativekl[r['id'],scale]))
 result['summary']={}
 for fam in sorted({r['family'] for r in test}):
  result['summary'][fam]={}
  for scale in cfg['scales']:
   result['summary'][fam][str(scale)]={};cc=[c for c in result['interventions'] if c['family']==fam and c['scale']==scale];zero=np.mean([c['kl'] for c in cc if c['mode']=='zero'])
   for mode in responses:
    vv=[c for c in cc if c['mode']==mode];gg=[g for g in result['geometry'] if g['family']==fam and g['scale']==scale and g['mode']==mode];mean_kl=float(np.mean([c['kl'] for c in vv]));result['summary'][fam][str(scale)][mode]=dict(relative_response_error=float(np.mean([g['relative_response_error'] for g in gg])),margin_mae=float(np.mean([abs(c['margin_error']) for c in vv])),mean_kl=mean_kl,kl_ratio_to_zero=float(mean_kl/zero),mean_exact_to_native_kl=float(np.mean([c['exact_to_native_kl'] for c in vv])))
 result['audits']=dict(exact_margin_max_error=max(abs(c['margin_error']) for c in result['interventions'] if c['mode']=='exact'),exact_kl_max=max(c['kl'] for c in result['interventions'] if c['mode']=='exact'),min_kl=min(c['kl'] for c in result['interventions']),finite=all(np.isfinite(c['kl']) for c in result['interventions']),bases_unchanged=bh==hashlib.sha256(bp.read_bytes()).hexdigest(),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),disjoint_fit_test_ids=True,disjoint_fit_test_candidate_tokens=True);result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['audits']),flush=True)
if __name__=='__main__':main()
