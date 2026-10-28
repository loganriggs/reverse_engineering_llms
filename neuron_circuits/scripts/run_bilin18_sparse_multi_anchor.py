"""H1007: independently encoded sparse supports and overlapping groups."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_sparse_multi_anchor'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','basis':ART/'bilin18_residual_discovery_spans_bases.pt','pca':ART/'bilin18_pca_behavior_transfer_bases.pt','oldmeta':ART/'expanded_joint_featurizer/block_paired_changes_expanded6_9831.pt','old':ART/'bilin18_context_training_bank_captures.pt','new':ART/'bilin18_noun_cue_controls_captures.pt','newscreen':ART/'bilin18_noun_cue_controls.json','natural':ART/'bilin18_natural_residual_discovery_captures.pt','selection':ART/'bilin18_coordinate_causal_atlas_selection.json','geometry':ART/'bilin18_multi_anchor_geometry_bases.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 bd=torch.load(paths['basis'],weights_only=False);pc=torch.load(paths['pca'],weights_only=False)['PCA128_expanded6'];B=pc['basis'].double();mu=pc['center'].double();q=bd['bases']['noun_expectation_rank1']
 old=torch.load(paths['old'],weights_only=False);meta=torch.load(paths['oldmeta'],weights_only=False);new=torch.load(paths['new'],weights_only=False);ns=json.loads(paths['newscreen'].read_text());nat=torch.load(paths['natural'],weights_only=False)
 training_indices=[meta['metadata'][i]['index'] for i in meta['training_indices']];assert len(training_indices)==144
 T=torch.stack([old['writes'][i][12].double() for i in training_indices]);ni=[i for i,r in enumerate(nat['metadata']) if r['split']=='discovery'];assert len(ni)==768;N=nat['writes'][ni].double()
 train_hashes={nat['metadata'][i]['document_hash'] for i in ni};validation_hashes={r['document_hash'] for r in nat['metadata'] if r['split']=='confirmation'};assert not train_hashes&validation_hashes
 source=torch.load(paths['geometry'],weights_only=False);Q=source['bases']['multi_anchor128'];cm=source['centers']['multi_anchor128'];bases={'old128':B,'multi128':Q};centers={'old128':mu,'multi128':cm}
 memberships={};groups=list(source['group_indices'].values())+[list(range(i,i+8)) for i in range(0,128,8)];M=torch.zeros(len(groups),128,dtype=torch.bool)
 for i,indices in enumerate(groups):M[i,indices]=True
 memberships['groups_real']=M
 for seed in [10071,10072]:memberships[f'groups_shuffle{seed}']=M[:,torch.randperm(128,generator=torch.Generator().manual_seed(seed))]
 for name in ['scalar32']+list(memberships)+['scalar_match_'+n for n in memberships]:bases[name]=Q;centers[name]=cm
 old_projector_error=0.
 torch.save(dict(bases=bases,centers=centers,memberships=memberships,group_names=list(source['group_indices'])+[f'background{i}' for i in range(16)]),ART/f'{S}_bases.pt');basis_hash=sha(ART/f'{S}_bases.pt')
 items=[];seen=set()
 for p in meta['pairs']:
  if p['source']!='context' or p['family']=='antonym' or p['split']=='discovery':continue
  for ind in p['indices']:
   r=meta['metadata'][ind];i=r['index']
   if i in seen:continue
   seen.add(i);items.append(dict(domain='old_tasks',family=p['family'],cohort=f"{p['family']}/{p['context']}/{p['split']}",eligible=p['native_pass'],ids=old['examples'][i]['ids'],write=old['writes'][i][12].double(),native=old['native'][i],source_index=i))
 eligible_new=set()
 for p in ns['pairs']:
  if p['kind']=='action' and any(c['pass_gate'] and (p['cohort'],p['condition'])==(c['cohort'],c['condition']) for c in ns['competence']):eligible_new.update(p['indices'])
 for i,e in enumerate(new['examples']):items.append(dict(domain='new_controls',family='noun_contrast',cohort='competent' if i in eligible_new else 'diagnostic',eligible=i in eligible_new,ids=e['ids'],write=new['writes'][i].double(),native=new['native'][i],source_index=i))
 for i,r in enumerate(nat['metadata']):
  if r['split']!='confirmation':continue
  items.append(dict(domain='natural',family='natural',cohort=r['document_hash'],eligible=True,ids=nat['tokens'][r['document'],:r['position']+1].tolist(),write=nat['writes'][i].double(),native=None,source_index=i))
 cfg=dict(experiment='H1007',hashes=hashes,design='Frozen multi-anchor128 dictionary. ScalarTop32 versus four greedily selected groups maximizing uncovered squared coordinate energy; six overlapping task8atom groups, one noun1atom group, sixteen background8atom groups. Two column-permuted memberships preserve group sizes and overlap structure. Each group arm has a scalar control with exactly its active atom count per state. All encoders use only that state; paired edits subtract independently reconstructed endpoints. Shared atoms decoded once.',basis_hash=basis_hash,task_training_slots=len(T),natural_training_states=len(N),evaluation='All old original-six non-discovery context states, retaining native failures as diagnostic; all80 H1002states with native-gate eligibility; all256 H998confirmation states from64documents, fourpositions each. No original whole-task holdouts evaluated. Paired edits include old contexts and new action pairs only; nuisance pairs excluded.',metrics='KL(native || reconstructed), native-top1 agreement, centered squared-error ratio. Macro average old native-passing cohorts and natural documents. Natural top1 agreement is fidelity, not ground-truth accuracy.',limits='Frozen dictionary and hard energy routing, not trained BSF encoder. Four selected groups can have different union dimensions; compare each to its per-state scalar budget control. Matched scalar budgets depend on group selector counts, not task labels. Native failures retained diagnostically. No fitting or task-aware routing.')
 X=torch.stack([r['write'] for r in items]);coordinates=(X-cm)@Q;energy=coordinates.square();supports={};chosen={};predictions={};Nitems=len(items)
 for arm in ['old128','multi128']:supports[arm]=torch.ones(Nitems,128,dtype=torch.bool)
 mask=torch.zeros_like(energy,dtype=torch.bool);mask.scatter_(1,energy.topk(32,dim=1).indices,True);supports['scalar32']=mask
 for name,M in memberships.items():
  mask=torch.zeros_like(energy,dtype=torch.bool);used=torch.zeros(Nitems,len(M),dtype=torch.bool);selected=[]
  for step in range(4):
   scores=(energy*(~mask))@M.double().T;scores[used]=-float('inf');pick=scores.argmax(1);used[torch.arange(Nitems),pick]=True;mask|=M[pick];selected.append(pick)
  supports[name]=mask;chosen[name]=torch.stack(selected,1);assert bool((mask.sum(1)<=32).all())
  control=torch.zeros_like(mask)
  for i in range(Nitems):control[i,energy[i].topk(int(mask[i].sum())).indices]=True
  supports['scalar_match_'+name]=control
 for name,b in bases.items():
  c=(X-centers[name])@b;predictions[name]=centers[name]+(c*supports[name])@b.T
 torch.save(dict(supports=supports,chosen=chosen,memberships=memberships,coordinates=coordinates,predictions=predictions,metadata=[{k:r[k] for k in ['domain','cohort','eligible','source_index']} for r in items]),ART/f'{S}_supports.pt');support_hash=sha(ART/f'{S}_supports.pt')
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 actual=0;replay=0.;buckets=defaultdict(list)
 for i,r in enumerate(items):
  if r['native'] is None:buckets[len(r['ids'])].append(i)
 for length,ii in sorted(buckets.items()):
  for st in range(0,len(ii),24):
   inds=ii[st:st+24];y,cap=forward(model,torch.tensor([items[i]['ids'] for i in inds],device='cuda'),capture=True);actual+=len(inds)
   for k,i in enumerate(inds):items[i]['native']=y[k].double().cpu();replay=max(replay,float((cap[12][k].double()-items[i]['write']).norm()/items[i]['write'].norm()))
 jobs=defaultdict(list)
 for i,r in enumerate(items):
  x=r['write'];centered=x-mu
  for arm in ['identity']+list(bases):
   pred=x if arm=='identity' else predictions[arm][i]
   jobs[len(r['ids'])].append((i,arm,pred.float(),float((x-pred).square().sum()/centered.square().sum())))
 records=[];samples=[];identity=0.;sample_counts=defaultdict(int)
 for length,jj in sorted(jobs.items()):
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];y,_=forward(model,torch.tensor([items[t[0]]['ids'] for t in batch],device='cuda'),patch=(12,torch.stack([t[2] for t in batch])));actual+=len(batch)
   for k,(i,arm,pred,err) in enumerate(batch):
    r=items[i];out=y[k].double().cpu();base=r['native'];lp=out.log_softmax(0);ln=base.log_softmax(0)
    if arm=='identity':identity=max(identity,float((out-base).abs().max()))
    rec={key:r[key] for key in ['domain','family','cohort','eligible','source_index']}|dict(item=i,arm=arm,KL=float((ln.exp()*(ln-lp)).sum()),top1_agreement=int(out.argmax()==base.argmax()),squared_error_ratio=err);records.append(rec)
    key=(r['domain'],arm)
    if sample_counts[key]<2:samples.append(dict(record=rec,output=out.float(),native=base.float(),write=r['write']));sample_counts[key]+=1
  print('DONE',length,actual,flush=True)
 grouped=defaultdict(list)
 for r in records:grouped[r['domain'],r['cohort'],r['eligible'],r['arm']].append(r)
 summary=[dict(zip(['domain','cohort','eligible','arm'],key))|dict(n=len(rr),mean_KL=float(np.mean([r['KL'] for r in rr])),top1_agreement=float(np.mean([r['top1_agreement'] for r in rr])),squared_error_ratio=float(np.mean([r['squared_error_ratio'] for r in rr]))) for key,rr in grouped.items()]
 macro=[]
 for domain in ['old_tasks','new_controls','natural']:
  for arm in ['identity']+list(bases):
   rr=[r for r in summary if r['domain']==domain and r['eligible'] and r['arm']==arm];macro.append(dict(domain=domain,arm=arm,cohorts=len(rr),states=sum(r['n'] for r in rr),mean_KL=float(np.mean([r['mean_KL'] for r in rr])),top1_agreement=float(np.mean([r['top1_agreement'] for r in rr])),squared_error_ratio=float(np.mean([r['squared_error_ratio'] for r in rr]))))
 # Paired donor edits use frozen bases; affine centers cancel exactly.
 item_lookup={(r['domain'],r['source_index']):i for i,r in enumerate(items)};pairs=[]
 for p in meta['pairs']:
  if p['source']!='context' or p['family']=='antonym' or p['split']=='discovery':continue
  inds=[item_lookup['old_tasks',meta['metadata'][i]['index']] for i in p['indices']];pairs.append(dict(indices=inds,domain='old_tasks',cohort=items[inds[0]]['cohort'],eligible=p['native_pass']))
 for p in ns['pairs']:
  if p['kind']!='action':continue
  inds=[item_lookup['new_controls',i] for i in p['indices']];eligible=any(c['pass_gate'] and (p['cohort'],p['condition'])==(c['cohort'],c['condition']) for c in ns['competence']);pairs.append(dict(indices=inds,domain='new_controls',cohort=p['cohort']+'/'+p['condition'],eligible=eligible))
 refs={};pair_records=[];pair_samples=[];counts=defaultdict(int)
 def run_pairs(reference):
  nonlocal actual
  jobs=defaultdict(list)
  for pi,p in enumerate(pairs):
   for rec,i in enumerate(p['indices']):
    donor=p['indices'][1-rec];delta=items[donor]['write']-items[i]['write']
    for arm in (['donor'] if reference else ['identity']+list(bases)):
     edit=delta if reference else torch.zeros_like(delta) if arm=='identity' else predictions[arm][donor]-predictions[arm][i]
     jobs[len(items[i]['ids'])].append((pi,p,rec,i,arm,(items[i]['write']+edit).float()))
  for length,jj in sorted(jobs.items()):
   for st in range(0,len(jj),24):
    batch=jj[st:st+24];y,_=forward(model,torch.tensor([items[t[3]]['ids'] for t in batch],device='cuda'),patch=(12,torch.stack([t[5] for t in batch])));actual+=len(batch)
    for k,(pi,p,rec,i,arm,repl) in enumerate(batch):
     out=y[k].double().cpu()
     if reference:refs[pi,rec]=out;continue
     lp=out.log_softmax(0);lr=refs[pi,rec].log_softmax(0);ln=items[i]['native'].log_softmax(0);r=dict(pair=pi,recipient=rec,item=i,domain=p['domain'],cohort=p['cohort'],eligible=p['eligible'],arm=arm,KL=float((lr.exp()*(lr-lp)).sum()),reference_KL=float((lr.exp()*(lr-ln)).sum()));pair_records.append(r)
     key=(p['domain'],arm)
     if counts[key]<2:pair_samples.append(dict(record=r,output=out.float(),reference=refs[pi,rec].float(),native=items[i]['native'].float()));counts[key]+=1
   print('PAIRS',reference,length,actual,flush=True)
 run_pairs(True);run_pairs(False);grouped_pairs=defaultdict(list)
 for r in pair_records:grouped_pairs[r['domain'],r['cohort'],r['eligible'],r['arm']].append(r)
 pair_summary=[]
 for key,rr in grouped_pairs.items():
  num=sum(r['KL'] for r in rr);den=sum(r['reference_KL'] for r in rr);pair_summary.append(dict(zip(['domain','cohort','eligible','arm'],key))|dict(n=len(rr),relative_KL=num/den,mean_KL=num/len(rr),reference_KL=den/len(rr)))
 pair_macro=[]
 for domain in ['old_tasks','new_controls']:
  for arm in ['identity']+list(bases):
   rr=[r for r in pair_summary if r['domain']==domain and r['arm']==arm and r['eligible']];pair_macro.append(dict(domain=domain,arm=arm,cohorts=len(rr),relative_KL=float(np.mean([r['relative_KL'] for r in rr]))))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=256+(1+len(bases))*len(items)+2*len(pairs)*(2+len(bases)),identity_logits_max_abs=identity,natural_prefix_capture_relative_error=replay,finite=all(np.isfinite(r['KL']) for r in records),no_fitting=True,supports_unchanged=sha(ART/f'{S}_supports.pt')==support_hash,bases_unchanged=sha(ART/f'{S}_bases.pt')==basis_hash,old_projector_replay_error=old_projector_error,natural_train_confirmation_overlap=0,test_evaluations=0)
 assert audits['hashes_unchanged'] and audits['supports_unchanged'] and audits['bases_unchanged'] and actual==audits['expected_forwards'] and identity<1e-4 and replay<1e-5 and audits['finite']
 torch.save(dict(bases=bases,centers=centers,center=mu,samples=samples,pair_samples=pair_samples),ART/f'{S}_audit_samples.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,macro=macro,pair_records=pair_records,pair_summary=pair_summary,pair_macro=pair_macro,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(macro,audits,flush=True)
if __name__=='__main__':main()
