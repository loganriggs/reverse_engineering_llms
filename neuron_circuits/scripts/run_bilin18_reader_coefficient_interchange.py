"""H872 frozen matched-token reader value interventions and signed-read algebra audit."""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_broad_task_competence import continuation_id,final_logits
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_reader_coefficient_interchange';TARGETS=[(13,0),(14,4),(8,3)]
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'panel':ART/'bilin18_aligned_lexical_continuations_captures.pt','screen':ART/'bilin18_aligned_lexical_continuations.json','previous':ART/'bilin18_matched_reader_values.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H872','hashes':hashes,'design':'Frozen12matched-country pairs,2formats,3families,144endpoints. Copy/capital competent; faileddiplomaticbinding retained separately. TargetsL13H0,L14H4,L8H3 frozen beforepanel. Native signed routing andlocal values preserved ateditedreader.','arms':'Final-query counterfactual head-write perturbations gamma*staticcountrycontrast, not replacement of nativeattention patterns. Identity,nativegamma,matchedforeign-task gamma,foreignsign/nativeamplitude,nativesign/foreignamplitude,signflip,plus/minus common geometricmean amplitude. Commonamplitude sqrt(abs(gamma_copy*gamma_capital)) is equal across matchedcontexts but pair-conditioned. Binding failedcontrol usescopy asforeign task.','metrics':'Native/edited own-task,countryidentity,capitalassociation margins; correctness,KL. Exactheadperturbation and final-logit replay between directvalueedit and predictedheadedit. Reader effect is not wholemodel necessity. Adaptive reusedpanel; nofeaturizer training ornew readerselection.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['panel'],weights_only=False);ex=data['examples'];screen=json.loads(paths['screen'].read_text());pairs=screen['config']['pairs'];idx={(e['id'],e['order']):i for i,e in enumerate(ex)};tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);groups={}
 for i,e in enumerate(ex):
  groups[i]={}
  for typ,part in [('country',0),('capital',1)]:
   ids=[continuation_id(tok,e['text'],' '+p[part]) for p in pairs[e['pair']]];assert all(v is not None for v in ids);groups[i][typ]=[[v] for v in ids]
 prior=json.loads(paths['previous'].read_text());old={(r['id'],r['order'],r['layer'],r['head']):r for r in prior['records'] if r['mode']=='predicted_final'};matched={(e['pair'],e['format'],e['order'],e['family']):i for i,e in enumerate(ex)};native_replay=0.;crosscontrast=0.
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 buckets=defaultdict(list)
 for i,e in enumerate(ex):buckets[len(e['ids'])].append(i)
 records=[];geometry=[];actual=0;ident=0.;headerr=0.;predlogiterr=0.;staticerr=0.;routingerr=0.;callcounts=set()
 for length,ii in sorted(buckets.items()):
  for st in range(0,len(ii),12):
   ix=ii[st:st+12];ids=torch.tensor([ex[i]['ids'] for i in ix],device='cuda');donors=[idx[ex[i]['id'],1-ex[i]['order']] for i in ix]
   for l,h in TARGETS:
    att=model.blocks[l].attn;original=att.squared_attention;lam=float(att.lamb);delta=torch.zeros(len(ix),length,128,device='cuda');one=[]
    for k,(i,j) in enumerate(zip(ix,donors)):
     pp=ex[i]['source_positions'];assert pp==ex[j]['source_positions'];db=(data['bus'][j][pp,h]-data['bus'][i][pp,h]).double();staticerr=max(staticerr,float((db-db[:1]).abs().max()));delta[k,pp]=(.5*lam*db).float().cuda();one.append(db[0])
    nativecoef=torch.tensor([old[ex[i]['id'],ex[i]['order'],l,h]['routing_coefficient'] for i in ix],device='cuda',dtype=torch.double);foreign=[];copycoef=[];capcoef=[]
    for i in ix:
     e=ex[i];fc=[]
     for fam in ['copy','capital']:
      j=matched[e['pair'],e['format'],e['order'],fam];fc.append(old[ex[j]['id'],ex[j]['order'],l,h]['routing_coefficient'])
     copycoef.append(fc[0]);capcoef.append(fc[1]);foreign.append(fc[1] if e['family']=='copy' else fc[0])
     if e['family'] in ['copy','capital']:
      other=matched[e['pair'],e['format'],e['order'],'capital' if e['family']=='copy' else 'copy'];od=idx[ex[other]['id'],1-e['order']];p=ex[other]['source_positions'][0];otherdelta=data['bus'][od][p,h].double()-data['bus'][other][p,h].double();ownp=e['source_positions'][0];ownj=idx[e['id'],1-e['order']];owndelta=data['bus'][ownj][ownp,h].double()-data['bus'][i][ownp,h].double();crosscontrast=max(crosscontrast,float((otherdelta-owndelta).abs().max()))
    foreign=torch.tensor(foreign,device='cuda',dtype=torch.double);common=(torch.tensor(copycoef,device='cuda',dtype=torch.double)*torch.tensor(capcoef,device='cuda',dtype=torch.double)).abs().sqrt();assert bool((nativecoef.abs()>1e-10).all()) and bool((foreign.abs()>1e-10).all())
    coefficients_by_mode={'identity':nativecoef*0,'native':nativecoef,'foreign':foreign,'foreign_sign':foreign.sign()*nativecoef.abs(),'foreign_magnitude':nativecoef.sign()*foreign.abs(),'sign_flip':-nativecoef,'common_positive':common,'common_negative':-common}
    for mode,coeff in coefficients_by_mode.items():
     calls=[0];saved={}
     def edit(q,k,v,q2,k2):
      nonlocal routingerr
      calls[0]+=1;base=original(q,k,v,q2,k2);A=((q[:,-1,h,None,:]*k[:,:,h]).sum(-1)/128)*((q2[:,-1,h,None,:]*k2[:,:,h]).sum(-1)/128);pred=coeff[:,None]*torch.stack(one).cuda();saved['coefficients']=coeff.cpu();saved['prednorm']=pred.norm(dim=1).cpu()
      for b,i in enumerate(ix):routingerr=max(routingerr,float((A[b].cpu()-data['captures'][i][l,h]['routing']).abs().max()))
      if mode=='identity':return base
      out=base.clone();out[:,h,-1]+=pred.to(out);return out
     att.squared_attention=edit
     try:y=final_logits(model,ids)
     finally:att.squared_attention=original
     actual+=len(ix);callcounts.add(calls[0]);assert calls[0]==1
     for k,i in enumerate(ix):
      native=data['native'][i];e=ex[i]
      if mode=='identity':ident=max(ident,float((y[k].double().cpu()-native).abs().max()))
      m=measure(y[k],e['groups'])
      if mode=='native':native_replay=max(native_replay,abs(m['margin']-old[e['id'],e['order'],l,h]['edited_margin']))
      n=measure(native,e['groups']);lp=native.log_softmax(0);rec={z:e[z] for z in ['pair','split','format','family','order','expected','id']};rec.update(layer=l,head=h,mode=mode,native_margin=n['margin'],edited_margin=m['margin'],native_top1_correct=n['top1'][e['expected']],edited_top1_correct=m['top1'][e['expected']],KL=float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum()),routing_coefficient=float(saved['coefficients'][k]),native_coefficient=float(nativecoef[k]),foreign_coefficient=float(foreign[k]),predicted_head_edit_norm=float(saved['prednorm'][k]))
      for typ,gg in groups[i].items():rec['native_'+typ+'_margin']=measure(native,gg)['margin'];rec['edited_'+typ+'_margin']=measure(y[k],gg)['margin']
      records.append(rec)
  print('DONE',length,actual,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_logits_max_abs':ident,'prior_native_edit_margin_error':native_replay,'matched_context_sourcecontrast_max_abs':crosscontrast,'same_contrast_across_sources_max_abs':staticerr,'native_routing_replay_max_abs':routingerr,'reader_call_counts':sorted(callcounts),'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL']) for r in records)};assert audits['hashes_unchanged'] and max(ident,native_replay,crosscontrast)<1e-4 and staticerr<1e-4 and audits['finite'] and actual==len(ex)*3*8;(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
