"""H739: zero-fit static bus map predictions and absent-token substitutions."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_static_bus_map'
@torch.no_grad()
def static_bus(model,token_ids,head=0):
 ids=torch.tensor(token_ids,device='cuda');x0=F.rms_norm(model.embedding(ids),(1152,));lam=model.blocks[0].lambdas;x=F.rms_norm(lam[0]*x0+lam[1]*x0,(1152,));return model.blocks[0].attn.c_v(x)[:,head*128:(head+1)*128]
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'fresh':ART/'bilin18_fresh_head_tasks_captures.pt','manifest':ART/'bilin18_fresh_head_tasks_manifest.json','factors':ART/'bilin18_head_routing_values_captures.pt','factor_results':ART/'bilin18_head_routing_values.json','model':SNAPSHOT/'pytorch_model.bin','model_source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py'),'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};old=torch.load(paths['fresh'],weights_only=False);factors=torch.load(paths['factors'],weights_only=False);prior=json.loads(paths['factor_results'].read_text());examples=old['examples'];rows={r['id']:r for r in json.loads(paths['manifest'].read_text())};tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True)
 cfg=dict(experiment='H739',hashes=hashes,design='FixedL13H0 onH732capital/binding/copy taskdirections eligibleH738. Require allchangedpositions shareexact same(beforetoken,aftertoken)pair; nooutcomeexclusions. StaticB(token)=layer0c_v(RMS(lambda00*x0+lambda01*x0)),x0=RMSembedding. Factorbusdelta asalpha*(B(after)-B(before)),alpha=lambda13*sum(nativepattern atchangedpositions). Rawprediction=(unembeddingtargetdifference) dot W_O_head dot busdelta, nofittedsigns/scales; notclaimedexactdownstreamlogitprediction. Contrast identityvsdeclaredtaskanswers.',absent_design='Same3predeclared sourcepairs acrosseach eligiblecontext:Kevin/Brian,stone/glass,Canada/Australia. Injecteach intochangedsourcepositions onlythroughL13H0bus contribution, routingandlocalvaluesfixed, downstreamlive. Exclude tokenpairs presentinprompt/answers ornotcontext-validsingletokens beforeoutputs. Compareoutputidentitycontrast for allpairs, andOttawa/Canberraassociation forcountry pair iftokenvalid. PredictionsusefixedW_O/unembedding mapandnativealpha; nofit. Taskcorrectnessnotexpectedforabsenttokens.',limitations='Rawunembeddingalignment omits downstreamtransformations,RMSnorm andlogitcap. Signagreement doesnotprove causalnecessity oruniversalidentity/association map. Tokenpairsabsentfromcurrentprompt, notnewtoallresearch. Nativecompetencefailuresretainedasdiagnostic. No semantic feature ormanifoldclaim.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 W=model.blocks[13].attn.c_proj.weight[:,:128];U=model.lm_head.weight;lam=float(model.blocks[13].attn.lamb);valid=[];exclusions=[];staticerrors=[];factorerrors=[];natural=[];jobs=[];jobmeta={};site='capital_head';pairs=[('names',[' Kevin',' Brian'],None),('objects',[' stone',' glass'],None),('countries',[' Canada',' Australia'],[' Ottawa',' Canberra'])]
 observed={(r['id'],r['receiver'],r['donor']):r['donor_directed_margin_change'] for r in prior['records'] if r['site']==site and r['mode']=='bus'}
 for r in prior['valid_directions']:
  if r['family'] not in ['capital_recall','entity_binding','sequence_copy'] or r['donor']==2:continue
  key=(r['id'],r['receiver']);er=examples[key];ed=examples[r['id'],r['donor']];positions=r['changed_positions'];sourcepairs={(er['ids'][i],ed['ids'][i]) for i in positions}
  if len(sourcepairs)!=1 or not positions:exclusions.append(dict(**r,reason='source substitutions not a single repeated token pair'));continue
  before,after=next(iter(sourcepairs));c=factors['components'][site,r['id'],r['receiver']];d=factors['components'][site,r['id'],r['donor']];vb=static_bus(model,[before,after]);alpha=lam*float(c['pattern'][positions].double().sum());delta=alpha*(vb[1]-vb[0]);expected=factors['deltas'][site,r['id'],r['receiver'],r['donor']]['bus'];factorerrors.append(float((delta.cpu().double()-expected).norm()/expected.norm().clamp_min(1e-12)))
  for i in positions:
   for v,actual in [(vb[0],c['bus'][i]),(vb[1],d['bus'][i])]:staticerrors.append(float((v.cpu().double()-actual.double()).norm()/actual.double().norm().clamp_min(1e-12)))
  gs=er['groups'];assert all(len(g)==1 for g in gs);u=U[gs[r['donor']][0]]-U[gs[r['receiver']][0]];core=W@(vb[1]-vb[0]);identity_core=float((U[after]-U[before])@core);task_core=float(u@core);rid=(r['id'],r['receiver'],r['donor']);natural.append(dict(**r,source_before=tok.decode([before]),source_after=tok.decode([after]),alpha=alpha,identity_core_alignment=identity_core,task_core_alignment=task_core,predicted_task_alignment=alpha*task_core,observed_bus_effect=observed[rid]));valid.append(r)
  for label,value in [('replay',vb[1]),('identity',vb[0])]:jobs.append(dict(rid=rid,condition=label,pair=None,delta=(alpha*(value-vb[0])).cpu()))
  text=(rows[r['id']]['prompts']+[rows[r['id']]['nuisance']])[r['receiver']]
  for pair,source_words,target_words in pairs:
   ids=[continuation_id(tok,text,w) for w in source_words];targets=[continuation_id(tok,text,w) for w in target_words] if target_words else None
   if any(t is None for t in ids) or any(t in er['ids'] or any(t in g for g in gs) for t in ids):exclusions.append(dict(**r,pair=pair,reason='absent source pair token validity/absence'));continue
   vals=static_bus(model,ids);core=W@(vals[0]-vals[1]);meta=dict(**r,pair=pair,alpha=alpha,identity_tokens=ids,predicted_identity_alignment=alpha*float((U[ids[0]]-U[ids[1]])@core))
   if targets is not None and all(t is not None for t in targets):meta.update(association_tokens=targets,predicted_association_alignment=alpha*float((U[targets[0]]-U[targets[1]])@core))
   elif targets is not None:meta['association_unavailable']='not single-token continuations'
   jobmeta[rid,pair]=meta
   for n in [0,1]:jobs.append(dict(rid=rid,condition=str(n),pair=pair,delta=(alpha*(vals[n]-vb[0])).cpu()))
 assert max(staticerrors)<1e-5 and max(factorerrors)<1e-4
 buckets=defaultdict(list)
 for job in jobs:buckets[len(examples[job['rid'][0],job['rid'][1]]['ids'])].append(job)
 results={};replay=[];identity=[];forwards=0
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];delta=torch.stack([q['delta'] for q in bb]).cuda()
   def edit(m,args):
    y=args[0].clone();y[:,-1,:128]+=delta.to(y);return (y,)
   hook=model.blocks[13].attn.c_proj.register_forward_pre_hook(edit)
   try:logits,_=run(model,torch.tensor([examples[q['rid'][0],q['rid'][1]]['ids'] for q in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for n,job in enumerate(bb):
    rid=job['rid'];key=(rid[0],rid[1]);gs=examples[key]['groups'];sign=1 if rid[2]==0 else -1;effect=sign*(measure(logits[n],gs)['margin']-measure(old['native_logits'][key],gs)['margin'])
    if job['condition']=='replay':replay.append(abs(effect-observed[rid]))
    elif job['condition']=='identity':identity.append(abs(effect))
    else:
     meta=jobmeta[rid,job['pair']];scores={}
     for typ in ['identity','association']:
      if typ+'_tokens' in meta:
       a,b=meta[typ+'_tokens'];scores[typ]=float(logits[n,a]-logits[n,b])
     results[rid,job['pair'],job['condition']]=scores
 absent=[]
 for (rid,pair),meta in jobmeta.items():
  a=results[rid,pair,'0'];b=results[rid,pair,'1'];absent.append(dict(**meta,**{'observed_'+typ+'_effect':a[typ]-b[typ] for typ in a}))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},static_bus_replay_max_relative=max(staticerrors),factorized_bus_delta_max_relative=max(factorerrors),bus_task_effect_replay_max_abs=max(replay),identity_effect_max_abs=max(identity),finite=all(np.isfinite(r['observed_identity_effect']) for r in absent),whole_model_forwards=forwards,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(replay)<1e-4 and max(identity)<1e-4;(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,natural=natural,absent=absent,exclusions=exclusions,competence=prior['competence'],audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
