"""Frozen H983/H984 joint representations: finite downstream edit fidelity."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_joint_featurizer_behavior'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 files=list(sorted((ART/'joint_change_featurizer').glob('*.pt')))+list(sorted((ART/'joint_anchor_featurizer').glob('*_true_labels_*.pt')))
 assert len(files)==20
 paths=[Path(__file__),ROOT/'scripts/run_bilin18_diverse_causal_tasks.py',ART/'bilin18_expanded_task_bank.json',ART/'bilin18_expanded_task_bank_captures.pt',ART/'bilin18_task_bank_causal_sites_captures.pt',SNAPSHOT/'pytorch_model.bin']+files;hashes={str(p):sha(p) for p in paths}
 bank=json.loads(paths[2].read_text());data=torch.load(paths[3],weights_only=False,map_location='cpu');writes=torch.load(paths[4],weights_only=False,map_location='cpu')['writes'];ex=data['examples'];native=data['native'];states={p.stem:torch.load(p,weights_only=False,map_location='cpu') for p in files}
 meta=next(iter(states.values()))['metadata'];assert all(d['metadata']==meta for d in states.values());pos={r['index']:i for i,r in enumerate(meta)}
 rows=[r for r in bank['records'] if r['split']!='discovery' and all(i in pos for i in r['indices'])];assert len(rows)==96
 means={}
 for f in sorted({r['family'] for r in meta if not r['holdout']}):
  rr=[r for r in bank['records'] if r['family']==f and r['split']=='discovery'];means[f]=torch.stack([writes[r['indices'][1]][12]-writes[r['indices'][0]][12] for r in rr]).mean(0)
 cfg=dict(experiment='H985',hashes=hashes,layer=12,models=list(states),validation_pairs=len(rows),design='Add predicted donor-minus-recipient write difference to native recipient MLP12 output. Compare full donor write, identity/no-change and family mean discovery delta. All20 frozen candidates, no selection/refitting. Task mean uses family and endpoint orientation; labeled diagnostic, not unsupervised encoder. No task mean for held-out families.',metrics='KL(full donor edit || candidate edit), normalized by KL(full donor edit || native); aggregate numerator/denominator within cohort. Signed donor-directed margin change. Report absolute KL and native competence. No best-seed selection.',limits='Common MLP12 site is not strongest for every task; task holdouts have no training exposure but have been evaluated previously. These are development transfer checks, not new prospective tasks.')
 (ROOT/'configs'/f'{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 refs={};actual=0;identity=0.;records=[];gate={(r['family'],r['split']):r['pass_gate'] for r in bank['summary']}
 def run(jobs,reference=False):
  nonlocal actual,identity
  buckets=defaultdict(list)
  for job in jobs:buckets[len(ex[job[2]]['ids'])].append(job)
  for length,jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    batch=jj[st:st+24];ids=torch.tensor([ex[t[2]]['ids'] for t in batch],device='cuda');replacement=torch.stack([t[4] for t in batch]);y,_=forward(model,ids,patch=(12,replacement));actual+=len(batch)
    for k,(r,endpoint,i,arm,repl) in enumerate(batch):
     out=y[k].double().cpu();ln=native[i].log_softmax(0);lp=out.log_softmax(0)
     if reference:refs[i]=out;continue
     lr=refs[i].log_softmax(0);kl=float((lr.exp()*(lr-lp)).sum());den=float((lr.exp()*(lr-ln)).sum());bm=measure(native[i],ex[i]['groups'])['margin'];margin=measure(out,ex[i]['groups'])['margin'];refmargin=measure(refs[i],ex[i]['groups'])['margin'];sign=1 if endpoint else -1
     if arm=='identity':identity=max(identity,float((out-native[i]).abs().max()))
     records.append(dict(pair=r['id'],family=r['family'],split=r['split'],whole_task_holdout=r['whole_task_holdout'],native_pass=gate[r['family'],r['split']],recipient=endpoint,index=i,arm=arm,KL=kl,reference_KL=den,donor_directed_margin=sign*(margin-bm),reference_margin=sign*(refmargin-bm),edit_norm=float((repl-writes[i][12]).norm())))
   print('BUCKET',reference,length,actual,flush=True)
 jobs=[]
 for r in rows:
  for endpoint,i in enumerate(r['indices']):jobs.append((r,endpoint,i,'full_donor',writes[r['indices'][1-endpoint]][12]))
 run(jobs,True);jobs=[]
 for r in rows:
  for endpoint,i in enumerate(r['indices']):
   donor=r['indices'][1-endpoint];base=writes[i][12];jobs.append((r,endpoint,i,'identity',base))
   if r['family'] in means:jobs.append((r,endpoint,i,'task_mean',base+(1 if endpoint==0 else -1)*means[r['family']]))
   for name,d in states.items():delta=(d['reconstruction'][pos[donor]]-d['reconstruction'][pos[i]])*d['scale'];jobs.append((r,endpoint,i,name,base+delta))
 run(jobs);grouped=defaultdict(list)
 for r in records:grouped[r['family'],r['split'],r['arm']].append(r)
 summary=[]
 for (f,split,arm),rr in grouped.items():
  num=sum(r['KL'] for r in rr);den=sum(r['reference_KL'] for r in rr);summary.append(dict(family=f,split=split,arm=arm,n=len(rr),native_pass=gate[f,split],whole_task_holdout=rr[0]['whole_task_holdout'],KL_ratio=num/den if den>0 else None,mean_KL=num/len(rr),mean_reference_KL=den/len(rr),mean_signed_margin=float(np.mean([r['donor_directed_margin'] for r in rr])),mean_reference_margin=float(np.mean([r['reference_margin'] for r in rr]))))
 expected=192+len(jobs);audits=dict(hashes_unchanged=all(sha(p)==hashes[str(p)] for p in paths),identity_logits_max_abs=identity,whole_model_forwards=actual,expected_forwards=expected,finite=all(np.isfinite(r['KL']) for r in records),no_training=True,reserved_test_access=False)
 assert audits['hashes_unchanged'] and identity<1e-4 and actual==expected and audits['finite'];torch.save(refs,ART/f'{S}_reference_logits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print('COMPLETE',audits,flush=True)
if __name__=='__main__':main()
