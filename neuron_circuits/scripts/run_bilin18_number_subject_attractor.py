"""H586: frozen number direction under crossed subject/attractor number."""
import json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups,measure,continuation_id
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
from audit_bilin18_mlp_discriminators import TOKENIZER
STEM='bilin18_number_subject_attractor'
def suite():
 regular=[('artist','artists'),('sailor','sailors'),('baker','bakers'),('clerk','clerks'),('scientist','scientists'),('musician','musicians'),('waiter','waiters'),('engineer','engineers')]
 irregular=[('man','men'),('woman','women'),('child','children'),('person','people'),('mouse','mice'),('goose','geese'),('tooth','teeth'),('foot','feet')]
 attractors=[('guard','guards'),('visitor','visitors'),('worker','workers'),('tourist','tourists'),('keeper','keepers'),('farmer','farmers'),('doctor','doctors'),('athlete','athletes')];rows=[]
 for kind,ss in [('regular',regular),('irregular',irregular)]:
  for i,(subject,attractor) in enumerate(zip(ss,attractors)):
   for frame in ('preposition','relative_clause'):
    for sn in (0,1):
     for an in (0,1):
      text=f'The {subject[sn]} beside the {attractor[an]} usually' if frame=='preposition' else f'The {subject[sn]} that the {attractor[an]} can see usually'
      rows.append(dict(id=f'{kind}-{i}-{frame}-{sn}-{an}',kind=kind,item=i,frame=frame,subject_number=sn,attractor_number=an,prompt=text))
 return rows
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_agreement_subspace_scope.json';bp=ART/'bilin18_agreement_write_subspace_locked_bases.pt';b=torch.load(bp,weights_only=False);V=b['pca'];R=b['random'];N=b['nuisance'];axis=V[0]*torch.sign(V[0]@b['mean']);cfg=dict(experiment='H586',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),basis_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),design='Regular/irregular subject plurals x subject number x attractor number x prepositional/relative-clause frame. Eight lexical quadruples per type/frame. Final token usually fixed. Subject-only and attractor-only Attention5 final-write swaps; full/zero, frozen ranks1/4/8, nuisance and five norm-matched random rank1 controls. No refitting.',hypotheses=['Grammatical subject number: coefficient changes mainly with subject even under opposite attractor number.','Nearest-noun plurality: coefficient follows attractor at least as strongly.','Regular suffix proxy: effect weakens on irregular plurals.','Frame-dependent mixture: subject dominance in PP can fail in relative clauses.'],gate='Per type/frame and per attractor number: >=4 subject pairs, >=75% both-endpoint candidate correctness, >=50% full-vocabulary intended-group top1. Retain every case.',scope='New prompt combinations, lexical overlap with earlier studies. Different token lengths possible and recorded; frame changes position and syntax together. Group NLL is not single-token CE.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));rows=suite();mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);ex={};index={}
 for r in rows:
  gs=[[continuation_id(tok,r['prompt'],x) for x in g] for g in groups('agreement',['is','are'])];assert all(all(x is not None for x in g) for g in gs);e=dict(r,ids=tok.encode(r['prompt'],add_special_tokens=False),groups=gs);ex[r['id']]=e;index[r['kind'],r['item'],r['frame'],r['subject_number'],r['attractor_number']]=r['id']
 model=load_full_model('cuda')
 for m in (model.embedding,model.blocks,model.lm_head):
  m.float().eval()
  for p in m.parameters():p.requires_grad_(False)
 buckets=defaultdict(list);writes={};native={}
 for k,e in ex.items():buckets[len(e['ids'])].append(k)
 for T,kk in buckets.items():
  for st in range(0,len(kk),24):
   bb=kk[st:st+24];cap={}
   def hook(m,args,out):cap['w']=out[0][:,-1].cpu()
   handle=model.blocks[5].attn.register_forward_hook(hook)
   try:z,_=run(model,torch.tensor([ex[k]['ids'] for k in bb],device='cuda'))
   finally:handle.remove()
   for ix,k in enumerate(bb):writes[k]=cap['w'][ix].clone();native[k]=z[ix].cpu()
 result=dict(config=cfg,manifest_sha256=mh,native=[],geometry=[],competence={},summary={},interventions=[],status='running')
 for k,e in ex.items():result['native'].append(dict(id=k,kind=e['kind'],frame=e['frame'],subject_number=e['subject_number'],attractor_number=e['attractor_number'],length=len(e['ids']),coefficient=float(writes[k]@axis),**measure(native[k],e['groups'])))
 for kind in ('regular','irregular'):
  for frame in ('preposition','relative_clause'):
   for an in (0,1):
    kk=[index[kind,i,frame,sn,an] for i in range(8) for sn in (0,1)];mm={k:measure(native[k],ex[k]['groups']) for k in kk};both=float(np.mean([mm[index[kind,i,frame,0,an]]['margin']>0 and mm[index[kind,i,frame,1,an]]['margin']<0 for i in range(8)]));top=float(np.mean([mm[k]['top1'][ex[k]['subject_number']] for k in kk]));result['competence'][f'{kind}_{frame}_attractor{an}']=dict(pairs=8,both_endpoint_choice=both,full_vocab_top1=top,passes=both>=.75 and top>=.5)
   for i in range(8):
    ks={(s,a):index[kind,i,frame,s,a] for s in (0,1) for a in (0,1)};cc={p:float(writes[k]@axis) for p,k in ks.items()};se=[cc[1,a]-cc[0,a] for a in (0,1)];ae=[cc[s,1]-cc[s,0] for s in (0,1)];result['geometry'].append(dict(kind=kind,frame=frame,item=i,subject_effects=se,attractor_effects=ae,subject_main=float(np.mean(se)),attractor_main=float(np.mean(ae)),interaction=se[1]-se[0],subject_length_changes=[len(ex[ks[1,a]]['ids'])-len(ex[ks[0,a]]['ids']) for a in (0,1)],attractor_length_changes=[len(ex[ks[s,1]]['ids'])-len(ex[ks[s,0]]['ids']) for s in (0,1)]))
 print('COMPETENCE',json.dumps(result['competence']),flush=True);jobs=defaultdict(list);vectors={}
 for k,e in ex.items():
  for swap in ('subject','attractor'):
   sn=e['subject_number'];an=e['attractor_number'];dk=index[e['kind'],e['item'],e['frame'],1-sn if swap=='subject' else sn,1-an if swap=='attractor' else an];d=writes[dk]-writes[k];vv={'full':d,'zero':torch.zeros_like(d)}
   for rank in (1,4,8):vv[f'pca_{rank}']=V[:rank].T@(V[:rank]@d)
   target=vv['pca_1'].norm();nd=N[:1].T@(N[:1]@d);vv['nuisance_matched_1']=nd*target/nd.norm().clamp_min(1e-20)
   for seed,B in R.items():
    rd=B[:1].T@(B[:1]@d);vv[f'random{seed}_matched_1']=rd*target/rd.norm().clamp_min(1e-20)
   for mode,v in vv.items():jobs[len(e['ids'])].append((k,swap,mode,v));vectors[k,swap,mode]=v
 print('LOCKED',sum(map(len,jobs.values())),flush=True);out={}
 for T,jj in jobs.items():
  for st in range(0,len(jj),24):
   bb=jj[st:st+24];z,_=run(model,torch.tensor([ex[k]['ids'] for k,sw,m,v in bb],device='cuda'),patch=(5,torch.stack([v for k,sw,m,v in bb])))
   for ix,(k,sw,m,v) in enumerate(bb):out[k,sw,m]=z[ix].cpu()
 for (k,swap,mode),z in out.items():
  e=ex[k];ref=out[k,swap,'full'];rp=ref.log_softmax(0);lp=z.log_softmax(0);np_=native[k].log_softmax(0);gs=e['groups'];nm=measure(native[k],gs)['margin'];rm=measure(ref,gs)['margin'];margin=measure(z,gs)['margin'];d=vectors[k,swap,'full'];sgn=1 if (e['subject_number'] if swap=='subject' else e['attractor_number'])==1 else -1;result['interventions'].append(dict(id=k,kind=e['kind'],frame=e['frame'],subject_number=e['subject_number'],attractor_number=e['attractor_number'],swap=swap,mode=mode,internal_relative_error=float((vectors[k,swap,mode]-d).norm()/d.norm()),reference_kl=float((rp.exp()*(rp-lp)).sum()),full_native_kl=float((rp.exp()*(rp-np_)).sum()),margin_error=margin-rm,margin_delta=margin-nm,full_margin_delta=rm-nm,changed_noun_directed_delta=(margin-nm)*sgn,group_nll_error=float(torch.logsumexp(rp[gs[e['subject_number']]],0)-torch.logsumexp(lp[gs[e['subject_number']]],0))))
 for kind in ('regular','irregular'):
  for frame in ('preposition','relative_clause'):
   for swap in ('subject','attractor'):
    cc=[c for c in result['interventions'] if (c['kind'],c['frame'],c['swap'])==(kind,frame,swap)];den=np.mean([abs(c['full_margin_delta']) for c in cc if c['mode']=='full']);kd=np.mean([c['full_native_kl'] for c in cc if c['mode']=='full'])
    for mode in sorted({c['mode'] for c in cc}):
     vv=[c for c in cc if c['mode']==mode];result['summary'][f'{kind}_{frame}_{swap}_{mode}']=dict(n=len(vv),internal_relative_error=float(np.mean([c['internal_relative_error'] for c in vv])),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/den),changed_noun_directed_delta=float(np.mean([c['changed_noun_directed_delta'] for c in vv])),group_nll_abs_error=float(np.mean([abs(c['group_nll_error']) for c in vv])))
 result['audits']=dict(zero_margin_max_error=max(abs(c['margin_delta']) for c in result['interventions'] if c['mode']=='zero'),finite=all(np.isfinite(c['reference_kl']) and np.isfinite(c['internal_relative_error']) for c in result['interventions']),source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),basis_unchanged=cfg['basis_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest(),manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest());result['status']='complete';result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2));torch.save(dict(examples=ex,writes=writes,native=native),ART/f'{STEM}_cache.pt');print(result['audits'],flush=True)
if __name__=='__main__':main()
