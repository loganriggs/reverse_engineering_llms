"""H992 train-only PCA versus expanded sparse models under finite write edits."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_pca_sparse_support'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 files=[ART/f'expanded_joint_featurizer/{k}_paired_changes_expanded6_{s}.pt' for k in ['block','shared_fixed'] for s in [9831,9832]]
 paths=[Path(__file__),ROOT/'scripts/run_bilin18_diverse_causal_tasks.py',ART/'bilin18_context_training_bank_captures.pt',ART/'bilin18_expanded_task_bank_captures.pt',ART/'bilin18_task_bank_causal_sites_captures.pt',ART/'joint_change_featurizer/block_paired_changes_9831.pt',SNAPSHOT/'pytorch_model.bin']+files+[ART/'bilin18_pca_behavior_transfer_bases.pt'];hashes={str(p):sha(p) for p in paths};new=torch.load(paths[2],weights_only=False,map_location='cpu');old=torch.load(paths[3],weights_only=False,map_location='cpu');oldwrites=torch.load(paths[4],weights_only=False,map_location='cpu')['writes'];small=torch.load(paths[5],weights_only=False,map_location='cpu');states={p.stem:torch.load(p,weights_only=False,map_location='cpu') for p in files};d=next(iter(states.values()));meta=d['metadata'];pairs=[p for p in d['pairs'] if p['family']!='antonym' and (p['source']=='holdout' or p['split']!='discovery')]
 X=torch.stack([(new['writes'] if r['source']=='context' else oldwrites)[r['index']][12] for r in meta]).double();ex=[(new if r['source']=='context' else old)['examples'][r['index']] for r in meta];native=[(new if r['source']=='context' else old)['native'][r['index']] for r in meta];preds={};bases={};rankinfo={};linearity=0.
 basis_state=torch.load(ART/'bilin18_pca_behavior_transfer_bases.pt',weights_only=False,map_location='cpu')['PCA128_expanded6'];B=basis_state['basis'];mu=basis_state['center'];z=(X-mu)@B;energy=z.square();supports={};memberships={};chosen={}
 assert torch.allclose(B.T@B,torch.eye(128,dtype=torch.double),atol=1e-10,rtol=1e-10)
 def topmask(count):
  order=energy.argsort(dim=1,descending=True);ranks=order.argsort(dim=1);return ranks<count if isinstance(count,int) else ranks<count[:,None]
 supports['dense128']=torch.ones_like(z,dtype=torch.bool);supports['dense32']=torch.arange(128)[None,:].expand(len(z),-1)<32;supports['scalar32']=topmask(32)
 def block_support(M):
  scores=energy@M.T;ix=scores.topk(8,dim=1).indices;g=torch.zeros(len(z),32,dtype=torch.double).scatter_(1,ix,1);return g@M>0,g
 M=torch.eye(32,dtype=torch.double).repeat_interleave(4,dim=1);supports['block_contiguous4'],chosen['block_contiguous4']=block_support(M);memberships['block_contiguous4']=M
 for seed in [9921,9922]:
  gen=torch.Generator().manual_seed(seed);perm=torch.randperm(128,generator=gen);M=torch.zeros(32,128,dtype=torch.double);M[torch.arange(32)[:,None],perm.reshape(32,4)]=1;name=f'block_random4_{seed}';supports[name],chosen[name]=block_support(M);memberships[name]=M
  M=torch.zeros(32,128,dtype=torch.double)
  for half in range(2):M[torch.arange(16)[:,None]+half*16,torch.randperm(128,generator=gen).reshape(16,8)]=1
  assert bool((M.sum(0)==2).all());q=torch.zeros_like(z,dtype=torch.bool);g=torch.zeros(len(z),32,dtype=torch.bool)
  for step in range(4):
   scores=(energy*(~q))@M.T;scores[g]=-float('inf');ix=scores.argmax(1);g[torch.arange(len(z)),ix]=True;q|=M[ix].bool()
  name=f'shared_greedy8_{seed}';supports[name]=q;memberships[name]=M;chosen[name]=g;supports[f'scalar_matched_{seed}']=topmask(q.sum(1));assert torch.equal(q,g.double()@M>0);assert bool((q.sum(1)<=32).all())
 for name,q in supports.items():preds[name]=(z*q)@B.T+mu
 # Exact scalar top-n is the best per-state support for this orthonormal dictionary.
 for seed in [9921,9922]:
  shared=supports[f'shared_greedy8_{seed}'];scalar=supports[f'scalar_matched_{seed}'];assert torch.equal(shared.sum(1),scalar.sum(1));assert bool(((energy*(~scalar)).sum(1)<=(energy*(~shared)).sum(1)+1e-8).all())
 support_stats={name:dict(mean=float(q.sum(1).double().mean()),minimum=int(q.sum(1).min()),maximum=int(q.sum(1).max())) for name,q in supports.items()}
 torch.save(dict(basis=B,center=mu,metadata=meta,pairs=pairs,supports=supports,memberships=memberships,chosen_groups=chosen,predicted_writes=preds),ART/f'{S}_candidates.pt')
 cfg=dict(experiment='H992',hashes=hashes,layer=12,arms=list(preds),support_stats=support_stats,design='Frozen H991 expanded6 PCA128 basis and exact single-state projection coordinates. Dense32/128; scalar Top32; disjoint4-dimensional blocks Top8 by energy (PCA-ordered or two seeded random partitions); overlapping8-atom groups, two balanced partitions and four greedy marginal-energy group selections. Shared coordinates decoded once. Scalar controls match each shared model actual per-state union size. No training or validation-label support selection.',metric='Same H991 finite full-donor-reference KL and signed margins. Cohort ratios then macro-average native-passing contexts/holdouts. All failed cohorts retained.',limits='Group memberships fixed, not learned semantic units. Overlap selection greedy, not a globally optimal union. Scalar top-n optimal for single-state Euclidean error only; not necessarily pair differences or output KL. Actual group budgets/inference compute differ. Development evaluation, no bootstrap claim.')
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
 audits=dict(hashes_unchanged=all(sha(p)==hashes[str(p)] for p in paths),identity_logits_max_abs=identity,exact_shared_union=True,scalar_matched_budget=True,whole_model_forwards=actual,expected_forwards=2*len(pairs)*(len(preds)+2),finite=all(np.isfinite(r['KL']) for r in records),test_evaluations=0)
 assert audits['hashes_unchanged'] and identity<1e-4 and linearity<1e-8 and actual==audits['expected_forwards'] and audits['finite'];torch.save(refs,ART/f'{S}_reference_logits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print('COMPLETE',audits,flush=True)
if __name__=='__main__':main()
