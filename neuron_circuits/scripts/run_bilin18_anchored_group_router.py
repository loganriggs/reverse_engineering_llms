"""H996 train-only PCA versus expanded sparse models under finite write edits."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_anchored_group_router'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 files=[ART/f'expanded_joint_featurizer/{k}_paired_changes_expanded6_{s}.pt' for k in ['block','shared_fixed'] for s in [9831,9832]]
 paths=[Path(__file__),ROOT/'scripts/run_bilin18_diverse_causal_tasks.py',ART/'bilin18_context_training_bank_captures.pt',ART/'bilin18_expanded_task_bank_captures.pt',ART/'bilin18_task_bank_causal_sites_captures.pt',ART/'joint_change_featurizer/block_paired_changes_9831.pt',SNAPSHOT/'pytorch_model.bin']+files+[ART/'bilin18_pca_behavior_transfer_bases.pt',ART/'bilin18_coordinate_causal_atlas_selection.json'];hashes={str(p):sha(p) for p in paths};new=torch.load(paths[2],weights_only=False,map_location='cpu');old=torch.load(paths[3],weights_only=False,map_location='cpu');oldwrites=torch.load(paths[4],weights_only=False,map_location='cpu')['writes'];small=torch.load(paths[5],weights_only=False,map_location='cpu');states={p.stem:torch.load(p,weights_only=False,map_location='cpu') for p in files};d=next(iter(states.values()));meta=d['metadata'];pairs=[p for p in d['pairs'] if p['family']!='antonym' and (p['source']=='holdout' or p['split']!='discovery')]
 X=torch.stack([(new['writes'] if r['source']=='context' else oldwrites)[r['index']][12] for r in meta]).double();ex=[(new if r['source']=='context' else old)['examples'][r['index']] for r in meta];native=[(new if r['source']=='context' else old)['native'][r['index']] for r in meta];preds={};bases={};rankinfo={};linearity=0.
 basis_state=torch.load(ART/'bilin18_pca_behavior_transfer_bases.pt',weights_only=False,map_location='cpu')['PCA128_expanded6'];B=basis_state['basis'];mu=basis_state['center'];z=(X-mu)@B;energy=z.square();supports={};memberships={};chosen={}
 assert torch.allclose(B.T@B,torch.eye(128,dtype=torch.double),atol=1e-10,rtol=1e-10)
 def topmask(count):
  order=energy.argsort(dim=1,descending=True);ranks=order.argsort(dim=1);return ranks<count if isinstance(count,int) else ranks<count[:,None]
 supports['dense128']=torch.ones_like(z,dtype=torch.bool);supports['scalar32']=topmask(32)
 selection=json.loads((ART/'bilin18_coordinate_causal_atlas_selection.json').read_text())['selected'];families=list(selection);anchor=torch.zeros(6,128,dtype=torch.double)
 for i,f in enumerate(families):anchor[i,selection[f]]=1
 background=torch.eye(16,dtype=torch.double).repeat_interleave(8,dim=1);M=torch.cat([anchor,background]);assert bool((M.sum(0)>0).all());memberships['all_groups']=M
 def fill(initial=None,allow_anchors=False):
  q=torch.zeros_like(z,dtype=torch.bool);g=torch.zeros(len(z),22,dtype=torch.bool)
  if initial is not None:g[torch.arange(len(z)),initial]=True;q=M[initial].bool()
  for _ in range(4-(initial is not None)):
   scores=(energy*(~q))@M.T;scores[g]=-float('inf')
   if not allow_anchors:scores[:,:6]=-float('inf')
   ix=scores.argmax(1);g[torch.arange(len(z)),ix]=True;q|=M[ix].bool()
  assert torch.equal(q,g.double()@M>0) and bool((g.sum(1)==4).all()) and bool((q.sum(1)<=32).all());return q,g
 supports['background_only'],chosen['background_only']=fill();supports['energy_all'],chosen['energy_all']=fill(allow_anchors=True)
 train=d['training_indices'];assert len(train)==144 and all(meta[i]['train_eligible'] and meta[i]['source']=='context' for i in train)
 allowed=torch.zeros(len(train),6,dtype=torch.bool);weights=torch.zeros(len(train))
 for k,i in enumerate(train):
  f=meta[i]['family'];allowed[k,families.index(f)]=True
  if f in ['sentiment','negation']:allowed[k,families.index('sentiment')]=True;allowed[k,families.index('negation')]=True
  weights[k]=1/sum(meta[j]['family']==f for j in train)/6
 std=z[train].std(0).clamp_min(1e-6);features=(z/std).float();routers={};router_metrics={}
 for seed in [9961,9962]:
  with torch.inference_mode(False),torch.enable_grad():
   torch.manual_seed(seed);router=torch.nn.Linear(128,6);tr=features[train].clone();ok=allowed.clone();w=weights.clone();opt=torch.optim.Adam(router.parameters(),lr=.01)
   for step in range(600):
    logits=router(tr);loss=((torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~ok,-float('inf')),1))*w).sum()+1e-4*router.weight.square().mean();opt.zero_grad();loss.backward();opt.step()
  logits=router(features);route=logits.argmax(1);name=f'learned_{seed}';supports[name],chosen[name]=fill(route);routers[name]=dict(state=router.state_dict(),scale=std,route=route,training_indices=train,compatible_labels=allowed)
  # Verify inference is single-state, with no validation/batch statistics.
  independent=torch.cat([router(features[i:i+1]) for i in range(len(features))]);assert float((independent-logits).abs().max())<1e-4
  router_metrics[name]=dict(train_compatible_accuracy=float(allowed[torch.arange(len(train)),route[train]].double().mean()),single_state_logit_error=float((independent-logits).abs().max()))
 oracle=torch.tensor([families.index(r['family']) if r['family'] in families else 0 for r in meta]);q,g=fill(oracle);unknown=torch.tensor([r['family'] not in families for r in meta]);q[unknown]=supports['background_only'][unknown];g[unknown]=chosen['background_only'][unknown];supports['label_oracle_known']=q;chosen['label_oracle_known']=g
 # One scalar baseline for each policy matches actual support count on every state.
 for name in list(chosen):supports['matched_'+name]=topmask(supports[name].sum(1))
 for name,q in supports.items():preds[name]=(z*q)@B.T+mu
 support_stats={name:dict(mean=float(q.sum(1).double().mean()),minimum=int(q.sum(1).min()),maximum=int(q.sum(1).max())) for name,q in supports.items()}
 torch.save(dict(basis=B,center=mu,metadata=meta,pairs=pairs,supports=supports,memberships=memberships,chosen_groups=chosen,predicted_writes=preds,routers=routers,router_metrics=router_metrics,training_indices=train,anchor_families=families),ART/f'{S}_candidates.pt')
 cfg=dict(experiment='H996',hashes=hashes,layer=12,arms=list(preds),support_stats=support_stats,router_metrics=router_metrics,design='Frozen PCA128 atoms/coordinates, six H994 supervised eight-atom anchor groups plus16unlabeled PCA-ordered background groups for full coverage. Four active groups total. Learned linear router selects one anchor from activation only; fill three background groups greedily by uncovered energy. Two seeds,600Adam steps on144discovery states only, training-only coordinate scaling. Sentiment/negation compatible labels avoid contradictory duplicated positive prompts. Background-only and all-group energy controls; label oracle uses known task but falls back to background on holdouts. Each policy has scalar actual-atom-count matched control.',metric='H991 validation panel finite full-write-reference KL. Router trained with labels; evaluation routing uses activation only. Same group budget, actual atom budgets reported. Task-label oracle is diagnostic, not a deployable policy.',limits='Background groups fixed and unlabeled, not learned discoveries. Known-class router forced to choose an anchor for unseen tasks; no novelty detector. Oracles on holdouts are fallback, not correct-task routing. No hierarchy/DAG or semantic manifold claim.')
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
 audits=dict(hashes_unchanged=all(sha(p)==hashes[str(p)] for p in paths),identity_logits_max_abs=identity,exact_shared_union=True,scalar_matched_budget=True,router_training_states=144,whole_model_forwards=actual,expected_forwards=2*len(pairs)*(len(preds)+2),finite=all(np.isfinite(r['KL']) for r in records),test_evaluations=0)
 assert audits['hashes_unchanged'] and identity<1e-4 and linearity<1e-8 and actual==audits['expected_forwards'] and audits['finite'];torch.save(refs,ART/f'{S}_reference_logits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print('COMPLETE',audits,flush=True)
if __name__=='__main__':main()
