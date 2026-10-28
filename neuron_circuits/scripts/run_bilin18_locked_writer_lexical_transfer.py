"""H583: causal retest on H582 discovery-locked formats and lexical transfer."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,measure,continuation_id
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_locked_writer_lexical_transfer'
OPS=[('mlp',3,4,0),('mlp',2,4,0),('mlp',6,7,8),('mlp',5,7,8),('attn',5,7,8),('attn',4,7,8),('attn',6,7,8)]
def suite():
 selection=json.loads((ART/'bilin18_task_boundary_factorial_locked_selection.json').read_text())
 rows=json.loads((ART/'bilin18_task_boundary_factorial_manifest.json').read_text())
 return [dict(r,template=0 if r['split']=='discovery' else 1) for r in rows if r['condition']==selection[r['family']]]
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_task_boundary_factorial.json';rows=suite();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 cfg=dict(experiment='H583',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),operations=OPS,sites=[-1,-2],design='H582 discovery-selected formats, five tasks, two lexical splits, up to eight pairs each and four endpoints. Reuse H582 prompts; template field means lexical split (0 discovery, 1 transfer), not a new template. Sites fixed to final and penultimate tokens for every task; no semantic role equivalence assumed. Swap writer output from opposite-answer donor or same-answer nuisance donor; self control. Replay only resulting selected head local-value change through native routing in a separate run. All token-valid pairs retained regardless of competence.',gate='Per task/template: >=4 pairs; both endpoints candidate correct >=.75; intended group full-vocabulary top1 >=.5. No per-item filtering. Nuisance competence separately reported.',hypotheses=['Candidate operations are task-general contextual writers.','Candidate operations preferentially affect copying/binding.','Internal write/value magnitude is insufficient to predict output influence.','Apparent task transfer may instead track fixed token position and template.'],precision='float32 compute with bf16-rounded weights',metrics='Internal writer and value displacement; full and isolated native-to-intervened KL; intended-answer-group NLL change; counterfactual donor-directed margin. Group NLL is not single-token CE.',selection='Operations/sites fixed from H580; format selected by H582 discovery only before transfer inference. Retain failed transfer cells. This tests lexical transfer at fixed selected wording, not independent template transfer.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));manifest=ART/f'{STEM}_manifest.json';manifest.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(manifest.read_bytes()).hexdigest();examples={};excluded=[]
 for r in rows:
  ee=[]
  for v,text in enumerate(r['prompts']):
   gs=[[continuation_id(tok,text,t) for t in g] for g in groups(r['family'],r['answers'])];gs=[[t for t in g if t is not None] for g in gs]
   if any(not g for g in gs) or set(gs[0])&set(gs[1]):break
   ee.append(dict(**r,variant=v,ids=tok.encode(text,add_special_tokens=False),groups=gs))
  if len(ee)!=4:excluded.append(r['id']);continue
  for e in ee:examples[e['id'],e['variant']]=e
 print('LOCKED',len(examples),'prompts; exclusions',excluded,flush=True);model=load_full_model('cuda')
 for mod in (model.embedding,model.blocks,model.lm_head):
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 H=model.config.n_head;D=model.config.n_embd//H;capstore={};native={};logits={};buckets=defaultdict(list)
 for k,e in examples.items():buckets[len(e['ids'])].append(k)
 for T,kk in buckets.items():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];cap={};handles=[]
   for kind,u in sorted({(o[0],o[1]) for o in OPS}):
    def wh(m,args,out,kind=kind,u=u):cap[kind,u]=(out[0] if kind=='attn' else out)[:,-2:].cpu()
    handles.append(getattr(model.blocks[u],kind).register_forward_hook(wh))
   for l in (4,7):
    for n in ('c_v','c_q','c_k','c_q2','c_k2'):
     def vh(m,args,out,l=l,n=n):cap[l,n]=out[:,-2:].view(len(bb),2,H,D).cpu()
     handles.append(getattr(model.blocks[l].attn,n).register_forward_hook(vh))
   try:zz,_=run(model,torch.tensor([examples[k]['ids'] for k in bb],device='cuda'))
   finally:
    for hook in handles:hook.remove()
   for ix,k in enumerate(bb):capstore[k]={n:v[ix].clone() for n,v in cap.items()};logits[k]=zz[ix].cpu();native[k]=measure(zz[ix],examples[k]['groups'])
 result=dict(config=cfg,manifest_sha256=mh,excluded=excluded,competence={},native=[dict(id=k[0],variant=k[1],**v) for k,v in native.items()],interventions=[],status='running')
 for fam in sorted({r['family'] for r in rows}):
  for t in (0,1):
   rr=[r for r in rows if r['family']==fam and r['template']==t and (r['id'],0) in native];n=len(rr);both=np.mean([native[r['id'],0]['margin']>0 and native[r['id'],1]['margin']<0 for r in rr]);top=np.mean([native[r['id'],v]['top1'][v] for r in rr for v in (0,1)]);nuis=np.mean([native[r['id'],v+2]['top1'][v] for r in rr for v in (0,1)]);result['competence'][f'{fam}_{t}']=dict(pairs=n,both_endpoint_choice=float(both),full_vocab_top1=float(top),nuisance_full_vocab_top1=float(nuis),passes=bool(n>=4 and both>=.75 and top>=.5))
 print('COMPETENCE',json.dumps(result['competence']),flush=True)
 routing={}
 for k,e in examples.items():
  T=len(e['ids'])
  for l,h in ((4,0),(7,8)):
   att=model.blocks[l].attn;cos,sin=att.rotary(torch.zeros(1,T,H,D,device='cuda'));vv={}
   for n in ('c_q','c_k','c_q2','c_k2'):
    x=capstore[k][l,n].cuda().view(1,2,H,D);vv[n]=model.tt.apply_rotary_emb(F.rms_norm(x,(D,)),cos[:,-2:],sin[:,-2:])[0,:,h]
   routing[k,l]=(vv['c_k']@vv['c_q'][-1]/D)*(vv['c_k2']@vv['c_q2'][-1]/D)
 def metrics(z,k):
  nz=logits[k].to(z);lp=F.log_softmax(z,0);nlp=F.log_softmax(nz,0);e=examples[k];v=e['variant'];answer=e['groups'][v];dm=measure(z,e['groups'])['margin']-native[k]['margin']
  return dict(margin_delta=dm,donor_directed_delta=dm*(1 if v==1 else -1),kl=float((nlp.exp()*(nlp-lp)).sum()),group_nll_delta=float(torch.logsumexp(nlp[answer],0)-torch.logsumexp(lp[answer],0)))
 jobs=defaultdict(list)
 for k,e in examples.items():
  if k[1]>1:continue
  for op in OPS:
   for site in (-1,-2):
    for mode in ('self','counterfactual','nuisance'):jobs[len(e['ids']),tuple(op),site,mode].append(k)
 print('INTERVENTIONS',sum(map(len,jobs.values())),flush=True)
 for ji,((T,(kind,u,l,h),site,mode),kk) in enumerate(jobs.items()):
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];donors=[(k[0],k[1] if mode=='self' else 1-k[1] if mode=='counterfactual' else k[1]+2) for k in bb];don=torch.stack([capstore[k][kind,u][site] for k in donors]).cuda();nv=torch.stack([capstore[k][l,'c_v'][site,h] for k in bb]).cuda();vc={};handles=[]
   if kind=='attn':
    def wh(m,args,out):y,bus=out;y=y.clone();y[:,site]=don.to(y);return y,bus
   else:
    def wh(m,args,out):y=out.clone();y[:,site]=don.to(y);return y
   handles.append(getattr(model.blocks[u],kind).register_forward_hook(wh))
   def vh(m,args,out):vc['v']=out[:,site].view(len(bb),H,D)[:,h].clone()
   handles.append(model.blocks[l].attn.c_v.register_forward_hook(vh));ids=torch.tensor([examples[k]['ids'] for k in bb],device='cuda')
   try:full,_=run(model,ids)
   finally:
    for hook in handles:hook.remove()
   dv=vc['v']-nv;pat=torch.stack([routing[k,l][site] for k in bb]);delta=F.linear(dv*(1-model.blocks[l].attn.lamb)*pat[:,None],model.blocks[l].attn.c_proj.weight[:,h*D:(h+1)*D]);med,_=run(model,ids,patch=(l,delta))
   for ix,k in enumerate(bb):
    e=examples[k];result['interventions'].append(dict(id=k[0],variant=k[1],family=e['family'],template=e['template'],operation=f'{kind}_{u}',target_layer=l,target_head=h,site=site,mode=mode,writer_delta_norm=float((don[ix].cpu()-capstore[k][kind,u][site]).norm()),value_delta_norm=float(dv[ix].norm()),mediated_write_norm=float(delta[ix].norm()),full=metrics(full[ix],k),mediated=metrics(med[ix],k)))
  if ji%60==0:print('PROGRESS',ji,len(jobs),flush=True)
 result['summary']={}
 for fam in sorted({r['family'] for r in rows}):
  for t in (0,1):
   for kind,u,l,h in OPS:
    for site in (-1,-2):
     for mode in ('counterfactual','nuisance'):
      cc=[c for c in result['interventions'] if c['family']==fam and c['template']==t and c['operation']==f'{kind}_{u}' and c['site']==site and c['mode']==mode];key=f'{fam}_{t}_{kind}{u}_site{site}_{mode}'
      result['summary'][key]=dict(n=len(cc),**{n:float(np.mean([c[n] for c in cc])) for n in ['writer_delta_norm','value_delta_norm','mediated_write_norm']},**{f'{p}_{m}':float(np.mean([c[p][m] for c in cc])) for p in ('full','mediated') for m in ['donor_directed_delta','kl','group_nll_delta']})
 ss=[c for c in result['interventions'] if c['mode']=='self'];result['audits']=dict(self_full_margin_max_error=max(abs(c['full']['margin_delta']) for c in ss),self_mediated_margin_max_error=max(abs(c['mediated']['margin_delta']) for c in ss),finite=all(np.isfinite(c[p][m]) for c in result['interventions'] for p in ('full','mediated') for m in ('margin_delta','kl','group_nll_delta')),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),manifest_unchanged=mh==hashlib.sha256(manifest.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(captures=capstore,logits=logits,examples=examples),ART/f'{STEM}_cache.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
