"""H1003: whole-state fidelity of frozen single-direction extension."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_joint_anchor_extension'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','basis':ART/'bilin18_residual_discovery_spans_bases.pt','pca':ART/'bilin18_pca_behavior_transfer_bases.pt','oldmeta':ART/'expanded_joint_featurizer/block_paired_changes_expanded6_9831.pt','old':ART/'bilin18_context_training_bank_captures.pt','new':ART/'bilin18_noun_cue_controls_captures.pt','newscreen':ART/'bilin18_noun_cue_controls.json','natural':ART/'bilin18_natural_residual_discovery_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()}
 bd=torch.load(paths['basis'],weights_only=False);pc=torch.load(paths['pca'],weights_only=False)['PCA128_expanded6'];B=pc['basis'].double();mu=pc['center'].double();q=bd['bases']['noun_expectation_rank1'];bases={'old128':B,'anchor129':torch.cat([B,q],1)}
 for seed in [10031,10032]:
  z=torch.randn(1152,1,generator=torch.Generator().manual_seed(seed),dtype=torch.double);z-=B@(B.T@z);z/=z.norm();bases[f'random129_{seed}']=torch.cat([B,z],1)
 old=torch.load(paths['old'],weights_only=False);meta=torch.load(paths['oldmeta'],weights_only=False);new=torch.load(paths['new'],weights_only=False);ns=json.loads(paths['newscreen'].read_text());nat=torch.load(paths['natural'],weights_only=False)
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
 cfg=dict(experiment='H1003',hashes=hashes,design='Whole-state MLP12 replacement mu+P(x-mu), not donor edits. Frozen oldtask128 versus129dimensions with noun anchor or either random residual direction. Fixed old training center for every arm. Identity control. No fitting.',evaluation='All old original-six non-discovery context states, retaining native failures as diagnostic; all80 H1002states with native-gate eligibility; all256 H998confirmation states from64documents, fourpositions each. No original whole-task holdouts evaluated.',metrics='KL(native || reconstructed), native-top1 agreement, centered squared-error ratio. Macro average old native-passing cohorts and natural documents. Natural top1 agreement is fidelity, not ground-truth accuracy.',limits='Dense linear representation; no learned sparse gates. Added dimension and random extensions have matched129dimension size. Reused development panels and natural confirmation are not untouched final tests. Stronger reconstruction geometry does not guarantee lower downstream KL.')
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
   pred=x if arm=='identity' else mu+(centered@bases[arm])@bases[arm].T
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
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),whole_model_forwards=actual,expected_forwards=256+5*len(items),identity_logits_max_abs=identity,natural_prefix_capture_relative_error=replay,finite=all(np.isfinite(r['KL']) for r in records),no_training=True,test_evaluations=0)
 assert audits['hashes_unchanged'] and actual==audits['expected_forwards'] and identity<1e-4 and replay<1e-5 and audits['finite']
 torch.save(dict(bases=bases,center=mu,samples=samples),ART/f'{S}_audit_samples.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,macro=macro,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(macro,audits,flush=True)
if __name__=='__main__':main()
