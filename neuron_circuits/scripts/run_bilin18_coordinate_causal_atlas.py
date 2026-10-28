"""H994 bounded six-task discovery atlas of finite PCA-coordinate donor edits."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,numpy as np,time
S='bilin18_coordinate_causal_atlas'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths=[Path(__file__),ROOT/'scripts/run_bilin18_diverse_causal_tasks.py',ART/'bilin18_context_training_bank.json',ART/'bilin18_context_training_bank_captures.pt',ART/'bilin18_pca_behavior_transfer_bases.pt',SNAPSHOT/'pytorch_model.bin'];hashes={str(p):sha(p) for p in paths};bank=json.loads(paths[2].read_text());data=torch.load(paths[3],weights_only=False,map_location='cpu');bs=torch.load(paths[4],weights_only=False)['PCA128_expanded6'];B=bs['basis'];mu=bs['center'];families=['agreement','capital_recall','entity_binding','sentiment','negation','pronoun_gender'];contexts={f:min(r['context'] for r in bank['summary'] if r['family']==f and r['split']=='discovery' and r['pass_gate']) for f in families};rows=[r for r in bank['records'] if r['family'] in families and r['split']=='discovery' and r['context']==contexts[r['family']]];assert len(rows)==24;ex=data['examples'];native=data['native'];writes=data['writes']
 cfg=dict(experiment='H994',hashes=hashes,contexts=contexts,pair_ids=[r['id'] for r in rows],design='Per family lowest-index passing discovery context, allfour source pairs, no individual correctness filter. MLP12: replace one coordinate by its donor value in fixed H991 PCA128 basis. Each atomic edit has a Gaussian direction control with identical signed coefficient and norm; fixed seed by source index and atom. Also identity/fullwrite/PCA128 reference. All128 atoms scanned, discovery only. Save all profiles; choose top8 by descending mean signed donor-directed margin for future validation, low-index ties.',limits='Task-related intervention effects, not causal necessity or independent additive units. Atomic coefficient depends on donor difference. Magnitude/activity confound remains; random norms matched but no semantic matched-direction control. One context/fourpairs per task. No evaluation-task labels used.')
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');(ART/f'{S}_manifest.json').write_text(json.dumps(cfg,indent=2)+'\n');manifesthash=sha(ART/f'{S}_manifest.json')
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 jobs=defaultdict(list);normerror=0
 for r in rows:
  for end,i in enumerate(r['indices']):
   donor=r['indices'][1-end];base=writes[i][12].double();delta=writes[donor][12].double()-base;z=delta@B
   for arm,v in [('identity',base),('full_write',base+delta),('PCA128',base+B@z)]:jobs[len(ex[i]['ids'])].append((r,end,i,arm,-1,0.,v))
   for atom in range(128):
    coefficient=z[atom];edit=coefficient*B[:,atom];noise=torch.randn(1152,dtype=torch.double,generator=torch.Generator().manual_seed(9940000+i*128+atom));noise=noise/noise.norm()*coefficient;normerror=max(normerror,float(abs(noise.norm()-edit.norm())/edit.norm().clamp_min(1e-20)))
    for arm,v in [('coordinate',base+edit),('random',base+noise)]:jobs[len(ex[i]['ids'])].append((r,end,i,arm,atom,float(coefficient),v))
 records=[];identity=0.;actual=0
 for length,jj in sorted(jobs.items()):
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[t[2]]['ids'] for t in batch],device='cuda');repl=torch.stack([t[6] for t in batch]);y,_=forward(model,ids,patch=(12,repl));actual+=len(batch)
   for k,(r,end,i,arm,atom,coeff,v) in enumerate(batch):
    out=y[k].double().cpu();base=native[i];lp=out.log_softmax(0);ln=base.log_softmax(0);effect=(1 if end else -1)*(measure(out,ex[i]['groups'])['margin']-measure(base,ex[i]['groups'])['margin'])
    if arm=='identity':identity=max(identity,float((out-base).abs().max()))
    records.append(dict(family=r['family'],pair=r['id'],source=r['source'],context=r['context'],recipient=end,index=i,arm=arm,atom=atom,coefficient=coeff,effect=effect,KL=float((lp.exp()*(lp-ln)).sum())))
  print('BUCKET',length,actual,flush=True)
 grouped=defaultdict(list)
 for r in records:grouped[r['family'],r['arm'],r['atom']].append(r)
 summary=[]
 for (f,arm,atom),rr in grouped.items():summary.append(dict(family=f,arm=arm,atom=atom,n=len(rr),mean_effect=float(np.mean([r['effect'] for r in rr])),rms_effect=float(np.mean([r['effect']**2 for r in rr])**.5),positive_fraction=float(np.mean([r['effect']>0 for r in rr])),mean_KL=float(np.mean([r['KL'] for r in rr]))))
 selected={f:[r['atom'] for r in sorted([r for r in summary if r['family']==f and r['arm']=='coordinate'],key=lambda r:(-r['mean_effect'],r['atom']))[:8]] for f in families};(ART/f'{S}_selection.json').write_text(json.dumps(dict(selected=selected,rule=cfg['design'],manifest_sha256=manifesthash),indent=2)+'\n')
 audits=dict(hashes_unchanged=all(sha(p)==hashes[str(p)] for p in paths),manifest_unchanged=sha(ART/f'{S}_manifest.json')==manifesthash,identity_logits_max_abs=identity,norm_match_relative=normerror,whole_model_forwards=actual,expected_forwards=48*259,finite=all(np.isfinite(r['effect']) and np.isfinite(r['KL']) for r in records),reserved_test_access=False)
 assert audits['hashes_unchanged'] and audits['manifest_unchanged'] and identity<1e-4 and normerror<1e-10 and actual==48*259 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,selected=selected,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print('COMPLETE',audits,flush=True)
if __name__=='__main__':main()
