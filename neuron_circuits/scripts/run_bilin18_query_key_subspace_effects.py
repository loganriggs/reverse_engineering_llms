"""H743: downstream tests of all H742 held-out-name edit projections."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_static_bus_map import static_bus
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_query_key_subspace_effects'
@torch.no_grad()
def main():
 torch.set_num_threads(6);start=time.time();paths={'projections':ART/'bilin18_query_key_edit_subspaces.json','native':ART/'bilin18_recall_operation_cross_captures.pt','native_results':ART/'bilin18_recall_operation_cross.json','target':ART/'bilin18_recall_query_key_cross.json','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'static_bus':ROOT/'scripts/run_bilin18_static_bus_map.py','forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};p=json.loads(paths['projections'].read_text());data=torch.load(paths['native'],weights_only=False);old=json.loads(paths['native_results'].read_text());target=json.loads(paths['target'].read_text());examples=data['examples'];native=data['native_logits'];rows=[r for r in p['records'] if r['split']=='heldout_names'];cfg=dict(experiment='H743',hashes=hashes,design='AllH742heldout-name rows, ranks/scopes/randomseeds andraw/normpreservingvariants unchanged. ReplaceonlydesignatedL13H0bus sourcecontribution usingprojectedcoefficient, nativefactor1 andallotheredges/localvaluesfixed. Testoriginalsourcevalue andKevin/Brian absentcontrast. Includeexactfullrankfactor2target,nativecoefficient,andzerocoefficient. ComparetoH740/H741replays. No candidate/rankselection inthisrun.',limitations='Reusedpanel, fitnamesheldoutonlyfrombasisfitting. Single-edgecounterfactual, notglobalquery/keyreplacement. Effects ofprojection determinedbycoefficient inthisfixed-map assay; independentfreshcontexts stillrequired foruniversal/unit/manifoldclaims.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 oldabs={(r['id'],r['receiver']):r for r in old['absent'] if r['pair']=='names' and r['gate']==1};exactabs={(r['id'],r['receiver']):r for r in target['absent'] if r['pair']=='names' and r['mode']==12};jobs=[];trialmeta={};keys=sorted({(r['id'],r['receiver']) for r in rows});basis_native={(r['id'],r['receiver']):r['native_alpha'] for r in rows}
 for key in keys:
  for scope,a in [('native',basis_native[key]),('zero',0.)]:rows.append(dict(id=key[0],receiver=key[1],family=examples[key]['family'],format=examples[key]['format'],scope=scope,rank=0,seed=None,renorm=False,native_alpha=basis_native[key],target_alpha=None,predicted_alpha=a))
 for tid,r in enumerate(rows):
  key=(r['id'],r['receiver']);e=examples[key];B=static_bus(model,[e['ids'][e['source_position']]])[0];tokens=oldabs[key]['tokens'];V=static_bus(model,tokens);a=r['predicted_alpha'];base=r['native_alpha'];trialmeta[tid]=r
  for kind,v,delta in [('native_value',None,(a-base)*B),('absent',0,a*V[0]-base*B),('absent',1,a*V[1]-base*B)]:jobs.append(dict(trial=tid,key=key,kind=kind,variant=v,delta=delta.cpu(),tokens=tokens))
 buckets=defaultdict(list)
 for job in jobs:buckets[len(examples[job['key']]['ids'])].append(job)
 result={};nativeeffects={};forwards=0
 for length,jj in buckets.items():
  for lo in range(0,len(jj),12):
   bb=jj[lo:lo+12];delta=torch.stack([r['delta'] for r in bb]).cuda()
   def edit(m,args):
    y=args[0].clone();y[:,-1,:128]+=delta.to(y);return(y,)
   hook=model.blocks[13].attn.c_proj.register_forward_pre_hook(edit)
   try:logits,_=run(model,torch.tensor([examples[r['key']]['ids'] for r in bb],device='cuda'))
   finally:hook.remove()
   forwards+=len(bb)
   for n,r in enumerate(bb):
    if r['kind']=='absent':a,b=r['tokens'];result[r['trial'],r['variant']]=float(logits[n,a]-logits[n,b])
    else:
     key=r['key'];gs=examples[key]['groups'];change=measure(logits[n],gs)['margin']-measure(native[key],gs)['margin'];nativeeffects[r['trial']]=change*(1 if key[1]==0 else -1)
 records=[];replay=[];native_replay=[];zeros=[];identities=[]
 for tid,r in trialmeta.items():
  key=(r['id'],r['receiver']);effect=result[tid,0]-result[tid,1];records.append(dict(**r,absent_identity_effect=effect,native_value_intended_margin_change=nativeeffects[tid],exact_factor2_absent_effect=exactabs[key]['observed_identity_effect'],native_absent_effect=oldabs[key]['observed_identity_effect']))
  if r['scope']=='identity' and not r['renorm']:replay.append(abs(effect-exactabs[key]['observed_identity_effect']))
  if r['scope']=='native':native_replay.append(abs(effect-oldabs[key]['observed_identity_effect']));identities.append(abs(nativeeffects[tid]))
  if r['scope']=='zero':zeros.append(abs(effect))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},full_rank_h741_effect_replay_max_abs=max(replay),native_h740_effect_replay_max_abs=max(native_replay),native_value_identity_max_abs=max(identities),zero_contrast_max_abs=max(zeros),finite=all(np.isfinite(r['absent_identity_effect']) and np.isfinite(r['native_value_intended_margin_change']) for r in records),whole_model_forwards=forwards,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(replay)<1e-4 and max(native_replay)<1e-4 and max(identities)<1e-4 and max(zeros)<1e-4;(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
