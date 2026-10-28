"""H606 separate item/task/wording transfer and predictor state."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure,groups,continuation_id
from run_bilin18_diverse_writer_transfer import suite as old_suite
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_predictor_controls'
def suite():
 rows=[]
 for r in old_suite():
  if r['template']!=0:continue
  fam=r['family'];a,b=r['answers'];i=r['item']
  for wording in [0,1]:
   if wording==0:base=r['prompts'][:2];edit=r['prompts'][2:]
   else:
    if fam=='sequence_copy':
     def make(x,obj):return f'A {obj} was delivered to {x}. A {obj} was delivered to {x}. A {obj} was delivered to'
     base=[make(x,'letter') for x in [a,b]];edit=[make(x,'message') for x in [a,b]]
    elif fam=='entity_binding':
     def make(x,obj):return f'{b} and {a} met at the museum. {x} offered the {obj} to'
     base=[make(x,'folder') for x in [b,a]];edit=[make(x,'notebook') for x in [b,a]]
    elif fam=='agreement':
     ns=[('actor','actors'),('pilot','pilots'),('nurse','nurses'),('lawyer','lawyers'),('chef','chefs'),('guard','guards'),('dancer','dancers'),('painter','painters')][i]
     base=[f'The {x} close to the window usually' for x in ns];edit=[f'The {x} close to the door usually' for x in ns]
    elif fam=='capital_recall':
     cs=['France','Germany','Italy','Japan','Spain','China','Russia','England'];base=[f'The city that serves as the capital of {cs[ii]} is' for ii in [i,(i+1)%8]];edit=['Fact: '+x for x in base]
    else:
     adjs=[('wonderful','terrible'),('excellent','awful'),('delightful','dreadful'),('enjoyable','boring'),('great','bad'),('amazing','horrible'),('superb','poor'),('fantastic','disappointing')][i]
     base=[f'Review: The performance felt {x}. Sentiment:' for x in adjs];edit=[f'Review: The show felt {x}. Sentiment:' for x in adjs]
   for edit_type in [0,1]:
    nuisance=edit if edit_type==0 else ['For the record, '+x[0].lower()+x[1:] for x in base]
    rows.append(dict(r,id=f'factorial-{fam}-{i}-{wording}-{edit_type}',template=wording*2+edit_type,wording=wording,edit_type=edit_type,prompts=base+nuisance))
 return rows

@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_metric_wording_edit_factorial_captures.pt';d=torch.load(src,weights_only=False);ex=d['examples'];families=sorted({e['family'] for e in ex.values()});cfg=dict(experiment='H606',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='All five tasks, item0-3 training only; item4-7/new wording evaluation. Four training regimes: same task/new wording, same task/old wording, other tasks/new wording, other tasks/old wording. Every task retains its own sentence format; new-wording label does not mean a common cross-task shell. Predict prefix dc from MLP11 normalized input, MLP5 normalized input, or Attention5 final write. Fixed ridge regularization .1 times mean Gram diagonal. Include per-regime constant mean. No evaluation dc used for predictions. Earlier-state models see final position only.',scope='Adaptive reused evaluation panel; new native captures extend training items and add predictor states. Main split is by item ID, not guaranteed lexical disjointness. Other-task models get32 contexts versus8 same-task. Test exactly same recipients and signals in every mode. No model selection on evaluation.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));rows=[r for r in suite() if r['edit_type']==1];mp=ART/f'{STEM}_manifest.json';mp.write_text(json.dumps(rows,indent=2));mh=hashlib.sha256(mp.read_bytes()).hexdigest();tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);all_ex={}
 for r in rows:
  for v,text in enumerate(r['prompts']):all_ex[r['id'],v]=dict(r,variant=v,ids=tok.encode(text,add_special_tokens=False))

 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 m=model.blocks[11].mlp;features={};native_logits={};batches=defaultdict(list)
 for k,e in all_ex.items():batches[len(e['ids'])].append(k)
 for kk in batches.values():
  for st in range(0,len(kk),16):
   bb=kk[st:st+16];cap={}
   def p11(mod,args):cap['mlp11']=args[0][:,-1].cpu()
   def p5(mod,args):cap['mlp5']=args[0][:,-1].cpu()
   def ah(mod,args,out):cap['attn5']=out[0][:,-1].cpu()
   hs=[m.register_forward_pre_hook(p11),model.blocks[5].mlp.register_forward_pre_hook(p5),model.blocks[5].attn.register_forward_hook(ah)]
   try:z,_=run(model,torch.tensor([all_ex[k]['ids'] for k in bb],device='cuda'))
   finally:
    for h in hs:h.remove()
   for i,k in enumerate(bb):features[k]={name:v[i].clone() for name,v in cap.items()};native_logits[k]=z[i].cpu()
 regimes=['same_new','same_old','other_new','other_old'];feature_names=['mlp11','mlp5','attn5'];folds={};fit_audits=[];overlap_audits=[]
 for fam in families:
  fold={};test=[k for k,e in all_ex.items() if k[1]<2 and e['family']==fam and e['template']==3 and e['item']>=4]
  for regime in regimes:
   same=regime.startswith('same');template=3 if regime.endswith('new') else 1;train=[k for k,e in all_ex.items() if k[1]<2 and e['item']<4 and e['template']==template and ((e['family']==fam)==same)];assert len(train)==(8 if same else 32) and len(test)==8
   Y=torch.stack([features[k[0],k[1]+2]['mlp11']-features[k]['mlp11'] for k in train]).double();ym=Y.mean(0);params={}
   for feature in feature_names:
    X=torch.stack([features[k][feature] for k in train]).double();xm=X.mean(0);X=X-xm;yc=Y-ym;G=X@X.T;lam=.1*G.trace()/len(X);H=G+lam*torch.eye(len(X),dtype=torch.float64);coef=torch.linalg.solve(H,yc);params[feature]=dict(xmean=xm,X=X,coef=coef);fit_audits.append(dict(family=fam,regime=regime,feature=feature,normal_error=float((H@coef-yc).norm()/yc.norm()),training_error=float((ym+G@coef-Y).norm()/Y.norm())))
   # Report exact text overlap separately from disjoint item IDs.
   traintexts={tuple(all_ex[k]['ids']) for k in train};overlap_audits.append(dict(family=fam,regime=regime,eval_exact_prompt_overlap=sum(tuple(all_ex[k]['ids']) in traintexts for k in test)))
   fold[regime]=dict(train=train,test=test,mean=ym,params=params)
  folds[fam]=fold
 fitted=ART/f'{STEM}_locked_folds.pt';torch.save(dict(folds=folds,features=features,examples=all_ex,config=cfg),fitted);fh=hashlib.sha256(fitted.read_bytes()).hexdigest();print('FROZEN',len(folds),'folds',flush=True)
 def T(c,e):
  c=c.cuda();e=e.cuda();v=m.Left(c)*m.Right(e)+m.Left(e)*m.Right(c);return (m.Down(v)-m.Down(torch.zeros_like(v))).cpu()
 modes=['full','native']+[r+'_'+v for r in regimes for v in ['mean']+feature_names];vectors={};geometry=[];buckets=defaultdict(list)
 for job,c in d['cache'].items():
  k,signal,sign=job;e=ex[k]
  if e['template']!=3:continue
  assert e['ids']==all_ex[k]['ids'];dc=c['dc'];de=c['de'];full=d['vectors'][job]['full'];vv={'full':full,'native':torch.zeros_like(full)};cg={}
  for regime in regimes:
   fit=folds[e['family']][regime];assert k in fit['test']
   for feature in ['mean']+feature_names:
    pred=fit['mean']
    if feature!='mean':
     p=fit['params'][feature];pred=pred+(features[k][feature].double()-p['xmean'])@p['X'].T@p['coef']
    mode=regime+'_'+feature;pred=pred.float();vv[mode]=T(pred,de);cg[mode]=float((pred-dc).norm()/dc.norm())
  geometry.append(dict(id=k[0],variant=k[1],family=e['family'],signal=signal,sign=sign,context_errors=cg));vectors[job]=vv;buckets[len(e['ids'])].append(job)
 sourcej=json.loads((ART/'bilin18_metric_wording_edit_factorial.json').read_text());competence={fam:sourcej['competence'][fam+'_3'] for fam in families};result=dict(config=cfg,competence=competence,fitting_audits=fit_audits,prompt_overlap=overlap_audits,geometry=geometry,interventions=[],summary={},status='running');refs={};replay=[]
 nb=torch.load(ART/'bilin18_agreement_write_subspace_locked_bases.pt',weights_only=False);axis=nb['pca'][0]*torch.sign(nb['pca'][0]@nb['mean']);size=float(axis@nb['mean'])*.0625;rand=torch.randn(1152,generator=torch.Generator().manual_seed(600));rand/=rand.norm()
 def delta(j):
  k,s,sgn=j;v=axis if s=='number' else rand if s=='random' else d['attn'][k[0],1-k[1]]-d['attn'][k]
  if s=='task':v=v/v.norm()
  return v*size*sgn
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save();print('INTERVENTIONS',len(vectors)*len(modes),flush=True)
 for mode in modes:
  for jj in buckets.values():
   for st in range(0,len(jj),16):
    bb=jj[st:st+16];rep=torch.stack([d['cache'][j]['write']+vectors[j][mode] for j in bb]).cuda()
    def hook(m,args,out):y=out.clone();y[:,-1]=rep.to(y);return y
    h=m.register_forward_hook(hook)
    try:z,_=run(model,torch.tensor([ex[j[0]]['ids'] for j in bb],device='cuda'),patch=(5,torch.stack([delta(j) for j in bb])))
    finally:h.remove()
    for i,j in enumerate(bb):
     k,signal,sign=j;e=ex[k];zz=z[i].cpu()
     if mode=='full':refs[j]=zz
     if mode=='native':replay.append(float((zz-d['cache'][j]['logits']).abs().max()))
     rp=refs[j].double().log_softmax(0);lp=zz.double().log_softmax(0);f=vectors[j]['full'];result['interventions'].append(dict(id=k[0],variant=k[1],family=e['family'],signal=signal,sign=sign,mode=mode,margin_error=measure(zz,e['groups'])['margin']-measure(refs[j],e['groups'])['margin'],reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce_error=float(rp[e['target']]-lp[e['target']]),internal_response_error=float((vectors[j][mode]-f).norm()/f.norm())))
  save();print('COMPLETE',mode,flush=True)
 for fam in families:
  for signal in ['number','task','random']:
   cc=[c for c in result['interventions'] if c['family']==fam and c['signal']==signal];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
   for mode in modes:
    vv=[c for c in cc if c['mode']==mode];result['summary'][f'{fam}_{signal}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd))
 result['audits']=dict(source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),manifest_unchanged=mh==hashlib.sha256(mp.read_bytes()).hexdigest(),fitted_folds_unchanged=fh==hashlib.sha256(fitted.read_bytes()).hexdigest(),training_item_overlap=0,native_capture_count=len(features),case_count=len(vectors),intervention_count=len(result['interventions']),replay_max_logit_error=max(replay),ridge_normal_error=max(q['normal_error'] for q in fit_audits),matched_native_replay_max_logit_error=max(float((native_logits[k]-d['native'][k]).abs().max()) for k in native_logits if k in d['native']),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['margin_error','reference_kl','target_ce_error','internal_response_error']));torch.save(dict(vectors=vectors,folds=folds),ART/f'{STEM}_vectors.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
