"""H991 train-only PCA versus expanded sparse models under finite write edits."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_pca_behavior_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 files=[ART/f'expanded_joint_featurizer/{k}_paired_changes_expanded6_{s}.pt' for k in ['block','shared_fixed'] for s in [9831,9832]]
 paths=[Path(__file__),ROOT/'scripts/run_bilin18_diverse_causal_tasks.py',ART/'bilin18_context_training_bank_captures.pt',ART/'bilin18_expanded_task_bank_captures.pt',ART/'bilin18_task_bank_causal_sites_captures.pt',ART/'joint_change_featurizer/block_paired_changes_9831.pt',SNAPSHOT/'pytorch_model.bin']+files;hashes={str(p):sha(p) for p in paths};new=torch.load(paths[2],weights_only=False,map_location='cpu');old=torch.load(paths[3],weights_only=False,map_location='cpu');oldwrites=torch.load(paths[4],weights_only=False,map_location='cpu')['writes'];small=torch.load(paths[5],weights_only=False,map_location='cpu');states={p.stem:torch.load(p,weights_only=False,map_location='cpu') for p in files};d=next(iter(states.values()));meta=d['metadata'];pairs=[p for p in d['pairs'] if p['family']!='antonym' and (p['source']=='holdout' or p['split']!='discovery')]
 X=torch.stack([(new['writes'] if r['source']=='context' else oldwrites)[r['index']][12] for r in meta]).double();ex=[(new if r['source']=='context' else old)['examples'][r['index']] for r in meta];native=[(new if r['source']=='context' else old)['native'][r['index']] for r in meta];preds={n:v['predicted_writes'].double() for n,v in states.items()};bases={};rankinfo={};linearity=0.
 for regime,train in [('small6',torch.stack([oldwrites[r['index']][12] for r in small['metadata'] if r['split']=='discovery' and not r['holdout']]).double()),('expanded6',X[d['training_indices']])]:
  mu=train.mean(0);_,sv,vh=torch.linalg.svd(train-mu,full_matrices=False);rank=int((sv>sv[0]*1e-6).sum())
  for target in [32,128]:
   n=min(target,rank);B=vh[:n].T;name=f'PCA{target}_{regime}';preds[name]=(X-mu)@B@B.T+mu;bases[name]=dict(basis=B,center=mu,rank=n);rankinfo[name]=n
   for pair in pairs[:3]:
    a,b=pair['indices'];direct=(X[b]-X[a])@B@B.T;linearity=max(linearity,float((preds[name][b]-preds[name][a]-direct).abs().max()))
 cfg=dict(experiment='H991',hashes=hashes,layer=12,arms=list(preds),pca_ranks=rankinfo,design='Same H989 original-six-family non-discovery context pairs plus all old copying/spatial task-holdout pairs. Retain native-failing diagnostics. PCA fit only48small or144expanded training slots; no validation basis fitting. Allfour expanded6 pair-trained sparse models retained. Candidate adds frozen independently reconstructed donor-recipient difference to native recipient MLP12 write.',metric='KL(full donor edit||candidate) divided by KL(full donor edit||native), pooled within family/context/split then macro-averaged. Also signed margin responses and absolute KL. No refitting sparse models or seed selection.',limits='PCA is dense, not an interpretable sparse group model. Small PCA128 actually rank40. MLP12 is not every task strongest site. Existing development panels, not new prospective vocabulary.')
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.save(bases,ART/f'{S}_bases.pt');model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 refs={};records=[];identity=0.;actual=0
 def run(jobs,reference=False):
  nonlocal actual,identity
  buckets=defaultdict(list)
  for job in jobs:buckets[len(ex[job[2]]['ids'])].append(job)
  for length,jj in sorted(buckets.items()):
   for st in range(0,len(jj),24):
    batch=jj[st:st+24];ids=torch.tensor([ex[t[2]]['ids'] for t in batch],device='cuda');repl=torch.stack([t[4] for t in batch]);y,_=forward(model,ids,patch=(12,repl));actual+=len(batch)
    for k,(p,end,i,arm,replacement) in enumerate(batch):
     out=y[k].double().cpu()
     if reference:refs[i]=out;continue
     ln=native[i].log_softmax(0);lp=out.log_softmax(0);lr=refs[i].log_softmax(0);bm=measure(native[i],ex[i]['groups'])['margin'];sign=1 if end else -1
     if arm=='identity':identity=max(identity,float((out-native[i]).abs().max()))
     records.append(dict(source=p['source'],family=p['family'],context=p['context'],split=p['split'],native_pass=p['native_pass'],index=i,source_index=meta[i]['index'],recipient=end,arm=arm,KL=float((lr.exp()*(lr-lp)).sum()),reference_KL=float((lr.exp()*(lr-ln)).sum()),margin_effect=sign*(measure(out,ex[i]['groups'])['margin']-bm),reference_margin=sign*(measure(refs[i],ex[i]['groups'])['margin']-bm)))
   print('BUCKET',reference,length,actual,flush=True)
 jobs=[(p,end,i,'reference',X[p['indices'][1-end]]) for p in pairs for end,i in enumerate(p['indices'])];run(jobs,True);jobs=[]
 for p in pairs:
  for end,i in enumerate(p['indices']):
   donor=p['indices'][1-end];jobs.append((p,end,i,'identity',X[i]))
   for name,pred in preds.items():jobs.append((p,end,i,name,X[i]+pred[donor]-pred[i]))
 run(jobs);grouped=defaultdict(list)
 for r in records:grouped[r['source'],r['family'],r['context'],r['split'],r['arm']].append(r)
 summary=[]
 for key,rr in grouped.items():
  num=sum(r['KL'] for r in rr);den=sum(r['reference_KL'] for r in rr);summary.append(dict(zip(['source','family','context','split','arm'],key))|dict(native_pass=rr[0]['native_pass'],n=len(rr),KL_ratio=num/den,mean_KL=num/len(rr),mean_reference_KL=den/len(rr),mean_margin_effect=float(np.mean([r['margin_effect'] for r in rr])),mean_reference_margin=float(np.mean([r['reference_margin'] for r in rr]))))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[str(p)] for p in paths),identity_logits_max_abs=identity,projection_linearity_max_abs=linearity,whole_model_forwards=actual,expected_forwards=2*len(pairs)*(len(preds)+2),finite=all(np.isfinite(r['KL']) for r in records),test_evaluations=0)
 assert audits['hashes_unchanged'] and identity<1e-4 and linearity<1e-8 and actual==audits['expected_forwards'] and audits['finite'];torch.save(refs,ART/f'{S}_reference_logits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print('COMPLETE',audits,flush=True)
if __name__=='__main__':main()
