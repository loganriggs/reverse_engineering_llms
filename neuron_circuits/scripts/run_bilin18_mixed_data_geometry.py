"""H1004: matched-rank task, natural and mixed discovery representations."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_mixed_data_geometry'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','basis':ART/'bilin18_residual_discovery_spans_bases.pt','pca':ART/'bilin18_pca_behavior_transfer_bases.pt','oldmeta':ART/'expanded_joint_featurizer/block_paired_changes_expanded6_9831.pt','old':ART/'bilin18_context_training_bank_captures.pt','new':ART/'bilin18_noun_cue_controls_captures.pt','newscreen':ART/'bilin18_noun_cue_controls.json','natural':ART/'bilin18_natural_residual_discovery_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 bd=torch.load(paths['basis'],weights_only=False);pc=torch.load(paths['pca'],weights_only=False)['PCA128_expanded6'];B=pc['basis'].double();mu=pc['center'].double();q=bd['bases']['noun_expectation_rank1']
 old=torch.load(paths['old'],weights_only=False);meta=torch.load(paths['oldmeta'],weights_only=False);new=torch.load(paths['new'],weights_only=False);ns=json.loads(paths['newscreen'].read_text());nat=torch.load(paths['natural'],weights_only=False)
 training_indices=[meta['metadata'][i]['index'] for i in meta['training_indices']];assert len(training_indices)==144
 T=torch.stack([old['writes'][i][12].double() for i in training_indices]);ni=[i for i,r in enumerate(nat['metadata']) if r['split']=='discovery'];assert len(ni)==768;N=nat['writes'][ni].double()
 train_hashes={nat['metadata'][i]['document_hash'] for i in ni};validation_hashes={r['document_hash'] for r in nat['metadata'] if r['split']=='confirmation'};assert not train_hashes&validation_hashes
 centers={};bases={};fit_matrices={};singular={}
 for domain in ['task','natural','mixed']:
  center=T.mean(0) if domain=='task' else N.mean(0) if domain=='natural' else (T.mean(0)+N.mean(0))/2
  x=(T-center)/len(T)**.5 if domain=='task' else (N-center)/len(N)**.5 if domain=='natural' else torch.cat([(T-center)/(2*len(T))**.5,(N-center)/(2*len(N))**.5])
  fit_matrices[domain]=x;u,sv,vh=torch.linalg.svd(x,full_matrices=False);assert (sv>sv.max()*1e-8).sum()>=128
  name='old128' if domain=='task' else domain+'128';bases[name]=vh[:128].T;centers[name]=center;singular[domain]=sv[:140].tolist()
  residual=x-(x@q)@q.T;_,ss,vv=torch.linalg.svd(residual,full_matrices=False);basis=torch.cat([q,vv[:127].T],1);bases[domain+'_anchor128']=basis;centers[domain+'_anchor128']=center
 for domain in ['natural','mixed']:
  name=domain+'128_oldcenter';bases[name]=bases[domain+'128'];centers[name]=mu
 old_projector_error=float((bases['old128']@bases['old128'].T-B@B.T).abs().max());assert old_projector_error<1e-8
 assert max(float((b.T@b-torch.eye(128)).abs().max()) for b in bases.values())<1e-8
 torch.save(dict(bases=bases,centers=centers,task_training_indices=training_indices,natural_training_indices=ni,singular=singular),ART/f'{S}_bases.pt');basis_hash=sha(ART/f'{S}_bases.pt')
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
 cfg=dict(experiment='H1004',hashes=hashes,design='All arms rank128. Task PCA from144original training slots; natural PCA from768discovery states; mixed PCA minimizes equal-domain mean squared error with equal-domain center. Anchor arms constrain inclusion of frozen nounrank1 and use127leading directions of training data projected orthogonal to it. Center-swap diagnostics use oldtaskcenter with unchanged natural/mixed bases. Freeze before evaluation.',basis_hash=basis_hash,task_training_slots=len(T),natural_training_states=len(N),evaluation='All old original-six non-discovery context states, retaining native failures as diagnostic; all80 H1002states with native-gate eligibility; all256 H998confirmation states from64documents, fourpositions each. No original whole-task holdouts evaluated.',metrics='KL(native || reconstructed), native-top1 agreement, centered squared-error ratio. Macro average old native-passing cohorts and natural documents. Natural top1 agreement is fidelity, not ground-truth accuracy.',limits='Dense PCA geometry, not sparse features. Mixed objective equalizes domain sample counts, not activation energies. All arms rank128; own-domain centers differ except explicit old-center diagnostics. Natural confirmation reused for development; no final-test claim. Reconstruction denominator uses fixed oldtaskcenter for comparability.')
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
   pred=x if arm=='identity' else centers[arm]+((x-centers[arm])@bases[arm])@bases[arm].T
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
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=256+(1+len(bases))*len(items),identity_logits_max_abs=identity,natural_prefix_capture_relative_error=replay,finite=all(np.isfinite(r['KL']) for r in records),discovery_only_fit=True,bases_unchanged=sha(ART/f'{S}_bases.pt')==basis_hash,old_projector_replay_error=old_projector_error,natural_train_confirmation_overlap=0,test_evaluations=0)
 assert audits['hashes_unchanged'] and audits['bases_unchanged'] and actual==audits['expected_forwards'] and identity<1e-4 and replay<1e-5 and audits['finite']
 torch.save(dict(bases=bases,centers=centers,center=mu,samples=samples),ART/f'{S}_audit_samples.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,macro=macro,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(macro,audits,flush=True)
if __name__=='__main__':main()
