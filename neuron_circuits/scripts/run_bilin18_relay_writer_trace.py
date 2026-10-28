"""H850 exact native writer decomposition and discovery-frozen writer interventions."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_writer_trace'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'fits':ART/'bilin18_relay_contrast_subspaces_fits.pt','old_examples':ART/'bilin18_candidate_task_screen_captures.pt','old_positions':ART/'bilin18_context_relay_positions.json','fresh':ART/'bilin18_relay_fresh_distance_captures.pt','fresh_results':ART/'bilin18_relay_fresh_distance.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H850','hashes':hashes,'design':'Exact weighted residual-write decomposition of frozen preMLP4 rank1 coordinate. At matching relay token: attention head outputs atL0..4 plusMLP outputsL0..3, weighted by subsequent residual lambdas. Same-token embedding difference cancels. Rank49 writers by discovery8pairs/task RMS signed contribution; freeze top3. Evaluate all freshH849 cases, both directions, without reselecting.','causal':'Each top3 writer full-vector donor delta or coordinate-only donor delta at original write site; top3 joint full and joint axial. Axial uses upstream dot(q)*q, not transport-weight multiplied twice. Measure observed preMLP4 coordinate change and final answer change; native accounting is not causal attribution.','numerical_audit':'Head/full-write summation on rawstates of norm thousands uses relative1e-5 threshold plus coordinate-specific relative1e-5; zero-edit logits absolute1e-4. Original absolute1e-4 decomposition audit failed despite <1e-6 relative error; diagnostics preserved.', 'scope':'Exploratory writer selection, frozen before fresh writer tests, but fresh panel already used for direction validation. Not independently fresh for overall research. Task-specific rank1 directions retained. All gates/alignment inherited, no exclusions by effects.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');fits=torch.load(paths['fits'],weights_only=False);oldex=torch.load(paths['old_examples'],weights_only=False)['examples'];oldrows=json.loads(paths['old_positions'].read_text())['records'];fresh=torch.load(paths['fresh'],weights_only=False);freshr=json.loads(paths['fresh_results'].read_text());model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 samples={};pairs=[]
 for fam,offset in [('agreement',-3),('sentiment',-4)]:
  rr=[r for r in oldrows if r['family']==fam and r['from_end']==offset and r['split']=='discovery' and r['recipient']==0]
  for r in rr:
   keys=[]
   for i in [r['i'],r['j']]:
    key=f'old-{i}';samples[key]={**oldex[i],'position':r['position']};keys.append(key)
   pairs.append({'id':r['id'],'family':fam,'cohort':'discovery','a':keys[0],'b':keys[1]})
 fr=[r for r in freshr['records'] if r['kind']=='full' and r['recipient']==0];findex={(e['id'],e['variant']):i for i,e in enumerate(fresh['examples'])}
 for r in fr:
  keys=[]
  for v in [0,1]:
   i=findex[r['id'],v];key=f'fresh-{i}';samples[key]=fresh['examples'][i];keys.append(key)
  pairs.append({k:r[k] for k in ['id','family','format','gap','source_pair']}|{'cohort':'fresh','a':keys[0],'b':keys[1]})
 transport={l:float(torch.stack([model.blocks[k].lambdas[0] for k in range(l+1,5)]).prod()) if l<4 else 1. for l in range(5)};caps={};native={};actual=0;headerr=0.;headrel=0.;traceerr=0.;tracerel=0.;axialrel=0.;ident=0.;records=[]
 def run(batch,patches=None,capture=False):
  nonlocal headerr,headrel
  ids=torch.tensor([samples[key]['ids'] for key in batch],device='cuda');pos=torch.tensor([samples[key]['position'] for key in batch],device='cuda');ix=torch.arange(len(batch),device='cuda');x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None;writes={};pre4=None
  for l,block in enumerate(model.blocks):
   raw=block.lambdas[0]*x+block.lambdas[1]*x0;hh={};hook=None
   if capture and l<=4:
    def grab(mod,args):hh['z']=args[0][ix,pos].clone()
    hook=block.attn.c_proj.register_forward_pre_hook(grab)
   try:att,bus=block.attn(F.rms_norm(raw,(1152,)),bus)
   finally:
    if hook:hook.remove()
   if capture and l<=4:
    z=hh['z'];ws=[]
    for h in range(9):
     w=F.linear(z[:,h*128:(h+1)*128],block.attn.c_proj.weight[:,h*128:(h+1)*128]);writes[l,'head',h]=w.cpu();ws.append(w)
    headerr=max(headerr,float((torch.stack(ws).sum(0)-att[ix,pos]).abs().max()));headrel=max(headrel,float(((torch.stack(ws).sum(0)-att[ix,pos]).norm(dim=-1)/att[ix,pos].norm(dim=-1).clamp_min(1e-30)).max()))
   if patches:
    for (pl,kind,h),edit in patches.items():
     if pl==l and kind=='head':att=att.clone();att[ix,pos]+=edit.to(att)
   pre=raw+att
   if l==4:pre4=pre[ix,pos].cpu()
   mlp=block.mlp(F.rms_norm(pre,(1152,)))
   if capture and l<4:writes[l,'mlp',-1]=mlp[ix,pos].cpu()
   if patches:
    for (pl,kind,h),edit in patches.items():
     if pl==l and kind=='mlp':mlp=mlp.clone();mlp[ix,pos]+=edit.to(mlp)
   x=pre+mlp
  logits=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30)
  return logits,pre4,writes
 for length in sorted({len(e['ids']) for e in samples.values()}):
  keys=[key for key,e in samples.items() if len(e['ids'])==length]
  for st in range(0,len(keys),16):
   batch=keys[st:st+16];y,p,w=run(batch,capture=True);actual+=len(batch)
   for k,key in enumerate(batch):native[key]=y[k].double().cpu();caps[key]={'pre':p[k],'writes':{unit:value[k] for unit,value in w.items()}}
 units=list(next(iter(caps.values()))['writes']);traces=[];selected={}
 for pair in pairs:
  a,b=pair['a'],pair['b'];q=fits['bases'][pair['family']][:,0];delta=caps[b]['pre'].double()-caps[a]['pre'].double();total=torch.zeros_like(delta);contributions=[]
  for unit in units:
   diff=(caps[b]['writes'][unit].double()-caps[a]['writes'][unit].double())*transport[unit[0]];total+=diff;contributions.append(float(diff@q))
  traceerr=max(traceerr,float((delta-total).abs().max()));tracerel=max(tracerel,float((delta-total).norm()/delta.norm()));axialrel=max(axialrel,float(abs((delta-total)@q)/abs(delta@q).clamp_min(1e-30)));traces.append(pair|{'coordinate_difference':float(delta@q),'contributions':contributions})
 for fam in ['agreement','sentiment']:
  tr=[r for r in traces if r['family']==fam and r['cohort']=='discovery'];rms=np.sqrt(np.mean(np.square([r['contributions'] for r in tr]),axis=0));order=np.argsort(-rms);selected[fam]=[units[int(k)] for k in order[:3]]
 (ART/f'{S}_selection.json').write_text(json.dumps({'selected':selected,'transport':transport,'units':units,'selection':'Discovery only RMS projected contributions; not fresh outcomes.'},indent=2)+'\n');print('SELECTED',selected,flush=True)
 freshpairs=[p for p in pairs if p['cohort']=='fresh'];jobs=[]
 for p in freshpairs:
  for rec in [0,1]:
   a,b=(p['a'],p['b']) if rec==0 else (p['b'],p['a']);u=selected[p['family']]
   for which in range(4):
    for mode in ['full','axial']:jobs.append((p,rec,a,b,which,mode,u if which==3 else [u[which]]))
 for length in sorted({len(samples[j[2]]['ids']) for j in jobs}):
  jj=[j for j in jobs if len(samples[j[2]]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];keys=[j[2] for j in batch];patch={}
   for k,(p,rec,a,b,which,mode,us) in enumerate(batch):
    q=fits['bases'][p['family']][:,0]
    for unit in us:
     diff=(caps[b]['writes'][unit]-caps[a]['writes'][unit]).double()
     if mode=='axial':diff=(diff@q)*q
     if unit not in patch:patch[unit]=torch.zeros(len(batch),1152,dtype=torch.float64)
     patch[unit][k]=diff
   y,pre,_=run(keys,patch);actual+=len(batch)
   for k,(p,rec,a,b,which,mode,us) in enumerate(batch):
    q=fits['bases'][p['family']][:,0];base=measure(native[a],samples[a]['groups'])['margin'];v=measure(y[k],samples[a]['groups'])['margin'];don=measure(native[b],samples[b]['groups'])['margin'];sgn=1 if rec==1 else -1;coord=float((pre[k].double()-caps[a]['pre'].double())@q);target=float((caps[b]['pre'].double()-caps[a]['pre'].double())@q);records.append({k:p[k] for k in ['id','family','format','gap','source_pair']}|{'recipient':rec,'which':which,'mode':mode,'writers':us,'donor_directed_delta':sgn*(v-base),'native_answer_gap':sgn*(don-base),'coordinate_change':coord,'native_coordinate_difference':target})
  print('DONE',length,flush=True)
 # Exact zero edits through allselected sites audit on one sample per length.
 for length in sorted({len(e['ids']) for e in samples.values()}):
  key=next(k for k,e in samples.items() if len(e['ids'])==length);patch={u:torch.zeros(1,1152) for us in selected.values() for u in us};y,_,_=run([key],patch);actual+=1;ident=max(ident,float((y[0].double().cpu()-native[key]).abs().max()))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'head_sum_max_abs':headerr,'head_sum_max_relative':headrel,'weighted_difference_max_relative':tracerel,'coordinate_decomposition_max_relative':axialrel,'weighted_native_difference_reconstruction_max_abs':traceerr,'zero_edit_logits_max_abs':ident,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['donor_directed_delta']) for r in records)};print('AUDIT',audits,flush=True);torch.save({'samples':samples,'caps':caps,'native':native},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'units':units,'selected':selected,'traces':traces,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True);assert audits['hashes_unchanged'] and ident<1e-4 and max(headrel,tracerel,axialrel)<1e-5 and audits['finite']
if __name__=='__main__':main()
