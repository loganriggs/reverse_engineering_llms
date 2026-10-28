"""H827 fixed directions crossed across unused same-token/position recipient documents."""
import json,time,hashlib,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
from run_bilin18_early_local_geometry import sha
S='bilin18_fixed_direction_recipients'
def cosine(a,b):return float(a@b/(a.norm()*b.norm()).clamp_min(1e-30))
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'panel':ART/'bilin18_fresh_early_geometry_panel_captures.pt','prior':ART/'bilin18_fresh_geometry_causal_captures.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};d=torch.load(paths['panel'],weights_only=False);prior=torch.load(paths['prior'],weights_only=False);excluded={e//16 for e in prior['native']};groups=[];coverage=[];used=set()
 for site,block in sorted({(p['site'],p['block']) for p in d['pairs']}):
  c=d['codes'][site];bi=c['blocks'].index(block);found=None
  for i,p in enumerate(d['pairs']):
   if p['site']!=site or p['block']!=block:continue
   pos=p['position'];pi=d['positions'].index(pos);candidates=[doc*16+pi for doc in range(1024) if doc not in excluded and doc not in used and c['mask'][doc*16+pi,bi] and int(d['all_token_ids'][doc,pos])==p['token_id']];candidates.sort(key=lambda ep:hashlib.sha256(f'H827:{site}:{block}:{ep}'.encode()).hexdigest())
   if len(candidates)>=4:found=(i,p,candidates);break
  if found is None:coverage.append({'site':site,'block':block,'status':'insufficient_unused_documents'});continue
  i,p,cc=found;rec=cc[:4];used.update(e//16 for e in rec);g={'site':site,'block':block,'source_pair':i,'token_id':p['token_id'],'position':p['position'],'recipients':rec,'directions':{mode:prior['patches'][i,'near',mode] for mode in ['block','random']}};groups.append(g);coverage.append({'site':site,'block':block,'status':'selected','candidates':len(cc),'recipients':rec})
 lock=[{k:v for k,v in g.items() if k!='directions'} for g in groups];cfg={'experiment':'H827','hashes':hashes,'groups':lock,'coverage':coverage,'selection':'For each H824 block, first frozen pair admitting4 new recipient documents. Same token/absolute position/block active. Exclude all H825 native endpoint documents and all already selected recipient documents. Hash-order selection; no output inspected.','directions':'Identical saved H825 near-block and near-random edit vectors used at all4 recipients per group, no recipient-specific rescaling.','doses':[-.25,-.1,0,.1,.25],'analysis':'Central finite differences at .1/.25; compare recipient responses with uniform-centered logits and common metric using average native probabilities across4 recipients. Own-recipient weighting diagnostic only.','limitations':'New recipient documents within existing H824 activation panel, no new corpus. Previously evaluated directions; exploratory, not semantic recovery.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');assert groups
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 native={};replay=[];forwards=0
 for gi,g in enumerate(groups):
  rec=g['recipients'];pos=g['position'];layer=int(g['site'][-1]);capt={}
  def hook(m,args,out):capt['x']=out[0][:,-1].cpu()
  h=model.blocks[layer].register_forward_hook(hook)
  try:z,_=run(model,torch.stack([d['all_token_ids'][e//16,:pos+1] for e in rec]).cuda())
  finally:h.remove()
  for n,e in enumerate(rec):native[gi,e]=z[n].cpu();truth=d['activations'][g['site']][e//16,e%16];replay.append(float((capt['x'][n]-truth).norm()/truth.norm()))
  forwards+=len(rec)
 outputs={};zeroerr=[];patchrelative=[]
 for gi,g in enumerate(groups):
  pos=g['position'];rec=g['recipients'];layer=int(g['site'][-1])
  for mode in ['zero','block','random']:
   for amp in ([0.] if mode=='zero' else [-.25,-.1,.1,.25]):
    v=torch.zeros(1152) if mode=='zero' else g['directions'][mode]*amp
    def edit(m,args,out):
     x,bus=out;x=x.clone();x[:,-1]+=v.to(x);return x,bus
    h=model.blocks[layer].register_forward_hook(edit)
    try:z,_=run(model,torch.stack([d['all_token_ids'][e//16,:pos+1] for e in rec]).cuda())
    finally:h.remove()
    for n,e in enumerate(rec):
     outputs[gi,e,mode,amp]=z[n].cpu()
     if mode=='zero':zeroerr.append(float((z[n].cpu()-native[gi,e]).abs().max()))
     else:patchrelative.append(float(v.norm()/d['activations'][g['site']][e//16,e%16].norm()))
    forwards+=len(rec)
  print('DONE',gi,g['site'],g['block'],flush=True)
 records=[];linearity=[];noise=[]
 for gi,g in enumerate(groups):
  prob=torch.stack([native[gi,e].double().softmax(0) for e in g['recipients']]).mean(0)
  def transform(v,metric,p=prob):return v-v.mean() if metric=='uniform' else (v-(p*v).sum())*p.sqrt()
  for mode in ['block','random']:
   for metric in ['uniform','common_fisher']:
    responses={}
    for e in g['recipients']:
     zero=outputs[gi,e,'zero',0.].double();aa={}
     for amp in [.1,.25]:
      plus=outputs[gi,e,mode,amp].double();minus=outputs[gi,e,mode,-amp].double();der=transform((plus-minus)/(2*amp),metric);aa[amp]=der;responses[e,amp]=der
      even=transform((plus+minus)/2-zero,metric);odd=transform((plus-minus)/2,metric);linearity.append({'group':gi,'site':g['site'],'block':g['block'],'recipient':e,'mode':mode,'metric':metric,'amp':amp,'even_over_odd':float(even.norm()/odd.norm().clamp_min(1e-30))})
      if metric=='common_fisher':
       err=transform(zero-native[gi,e].double(),metric).norm();noise.append(float(odd.norm()/err.clamp_min(1e-30)))
     linearity.append({'group':gi,'site':g['site'],'block':g['block'],'recipient':e,'mode':mode,'metric':metric,'amp':'cross_dose','derivative_relative_change':float((aa[.25]-aa[.1]).norm()/aa[.1].norm().clamp_min(1e-30)),'derivative_cosine':cosine(aa[.25],aa[.1])})
    for amp in [.1,.25]:
     for a,b in itertools.combinations(g['recipients'],2):records.append({'group':gi,'site':g['site'],'block':g['block'],'mode':mode,'metric':metric,'amp':amp,'a':a,'b':b,'response_cosine':cosine(responses[a,amp],responses[b,amp]),'norm_ratio':float(responses[a,amp].norm()/responses[b,amp].norm().clamp_min(1e-30))})
 summaries=[]
 for site in ['residual0','residual1']:
  for mode in ['block','random']:
   for metric in ['uniform','common_fisher']:
    rr=[r for r in records if r['site']==site and r['mode']==mode and r['metric']==metric and r['amp']==.1];ll=[r for r in linearity if r['site']==site and r['mode']==mode and r['metric']==metric and r['amp']=='cross_dose'];ee=[r for r in linearity if r['site']==site and r['mode']==mode and r['metric']==metric and r['amp']==.1]
    if rr:summaries.append({'site':site,'mode':mode,'metric':metric,'groups':len({r['group'] for r in rr}),'recipient_pairs':len(rr),'median_cross_recipient_cosine':float(np.median([r['response_cosine'] for r in rr])),'negative_pairs':sum(r['response_cosine']<0 for r in rr),'median_dose_derivative_change':float(np.median([r['derivative_relative_change'] for r in ll])),'median_dose_derivative_cosine':float(np.median([r['derivative_cosine'] for r in ll])),'median_small_dose_even_odd':float(np.median([r['even_over_odd'] for r in ee]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'recipient_documents_disjoint_prior':not bool(used&excluded),'recipient_documents_unique':len(used)==4*len(groups),'groups':len(groups),'recipient_count':len(used),'native_replay_max_relative':max(replay),'zero_edit_max_abs':max(zeroerr),'max_patch_relative_to_residual':max(patchrelative),'minimum_odd_to_replay_ratio':min(noise),'whole_model_forwards':forwards,'test_evaluations':0,'finite':all(np.isfinite(r['response_cosine']) for r in records)};assert audits['hashes_unchanged'] and audits['recipient_documents_unique'] and audits['recipient_documents_disjoint_prior'] and max(replay)<1e-4 and max(zeroerr)<1e-4 and audits['finite']
 torch.save({'groups':groups,'native':native,'outputs':outputs},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'linearity':linearity,'summary':summaries,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summaries,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()
