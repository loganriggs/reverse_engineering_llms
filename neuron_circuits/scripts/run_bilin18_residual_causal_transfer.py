"""H1000: fixed MLP12 residual-subspace interventions; no task fitting."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_residual_causal_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','screen':ART/'bilin18_residual_behavior_probes.json','captures':ART/'bilin18_residual_behavior_probes_captures.pt','eligibility':ART/'bilin18_residual_behavior_probes_eligibility.json','natural':ART/'bilin18_natural_residual_discovery_captures.pt','pca':ART/'bilin18_pca_behavior_transfer_bases.pt','model':SNAPSHOT/'pytorch_model.bin'}
 hashes={k:sha(p) for k,p in paths.items()}
 data=torch.load(paths['captures'],weights_only=False);natural=torch.load(paths['natural'],weights_only=False);pca=torch.load(paths['pca'],weights_only=False)['PCA128_expanded6']['basis'].double();screen=json.loads(paths['screen'].read_text());elig=json.loads(paths['eligibility'].read_text())
 rows=[r for r in screen['records'] if r['family'] in elig['candidate_families']];assert len(rows)==32 and all(r['same_final_token'] for r in rows)
 bases={};ranks={}
 for seed in [9981,9982]:
  c=natural['clusters'][seed]['centers'].double().T;c=c-pca@(pca.T@c);u,s,v=torch.linalg.svd(c,full_matrices=False);rank=int((s>s.max()*1e-8).sum());assert rank==8;bases[f'centers{seed}']=u[:,:rank];ranks[str(seed)]=rank
 for seed in [10001,10002]:
  c=torch.randn(1152,8,generator=torch.Generator().manual_seed(seed),dtype=torch.double);c-=pca@(pca.T@c);bases[f'random_span{seed}']=torch.linalg.qr(c).Q
 cfg=dict(experiment='H1000',hashes=hashes,layer=12,design='Fixed task PCA128 plus frozen natural residual cluster-center spans (two seeds, rank8), random residual spans rank8, full residual, full donor and identity. All32eligible pairs bidirectional; no outcome selection or fitting. Centers from natural discovery only, not new task examples.',controls='Two independent random residual spans; paired +/- isotropic residual noise norm-matched to centers9981 projected edit. Random controls are not matched to unknown causal directions.',metrics='Signed donor-directed task-margin change; KL(full-donor-patched output || arm output), KL(arm || native), edit energy. Full donor reference stays in recipient context.',limits='Fixed MLP12; negative results do not rule out other sites. Projection is a donor-informed intervention, not a deployable task predictor or evidence for discrete semantic clusters. Cue confounds from H999 remain; factorial controls pending.',holdouts='Only noun_expectation and word_fragment. List_marker explicitly excluded. No reserved tests or training.')
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 ex=data['examples'];buckets=defaultdict(list);decomp=0.;orth=0.;normerr=0.
 for name,b in bases.items():orth=max(orth,float((b.T@b-torch.eye(8)).abs().max()),float((pca.T@b).abs().max()))
 for r in rows:
  for rec,i in enumerate(r['indices']):
   donor=r['indices'][1-rec];w=data['writes'][i][12].double();delta=data['writes'][donor][12].double()-w;task=pca@(pca.T@delta);res=delta-task
   edits={'identity':torch.zeros_like(delta),'donor':delta,'task128':task,'residual':res}|{name:b@(b.T@res) for name,b in bases.items()}
   g=torch.Generator().manual_seed(100000+i);noise=torch.randn(1152,generator=g,dtype=torch.double);noise-=pca@(pca.T@noise);noise=noise/noise.norm()*edits['centers9981'].norm();edits['random_plus']=noise;edits['random_minus']=-noise
   decomp=max(decomp,float((task+res-delta).abs().max()));normerr=max(normerr,float(abs(noise.norm()-edits['centers9981'].norm())))
   for name,e in edits.items():buckets[len(ex[i]['ids'])].append((r,rec,i,donor,name,(w+e).float(),float(e.norm()),float(delta.norm())))
 model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 records=[];logits=[];identity=0.;actual=0
 for length,jobs in sorted(buckets.items()):
  for st in range(0,len(jobs),24):
   batch=jobs[st:st+24];y,_=forward(model,torch.tensor([ex[t[2]]['ids'] for t in batch],device='cuda'),patch=(12,torch.stack([t[5] for t in batch])))
   for k,(r,rec,i,donor,name,repl,en,dn) in enumerate(batch):
    out=y[k].double().cpu();base=data['native'][i];m=measure(out,ex[i]['groups'])['margin'];bm=measure(base,ex[i]['groups'])['margin'];lp=out.log_softmax(0);ln=base.log_softmax(0)
    if name=='identity':identity=max(identity,float((out-base).abs().max()))
    records.append(dict(id=r['id'],family=r['family'],split=r['split'],recipient=rec,index=i,donor=donor,kind=name,margin=m,native_margin=bm,effect=(1 if rec==1 else -1)*(m-bm),edit_norm=en,full_delta_norm=dn,KL_to_native=float((lp.exp()*(lp-ln)).sum())));logits.append(out.float())
   actual+=len(batch)
  print('DONE',length,actual,flush=True)
 refs={(r['id'],r['recipient']):logits[k].double().log_softmax(0) for k,r in enumerate(records) if r['kind']=='donor'}
 for k,r in enumerate(records):
  ref=refs[r['id'],r['recipient']];lp=logits[k].double().log_softmax(0);r['full_patch_KL']=float((ref.exp()*(ref-lp)).sum())
 grouped=defaultdict(list)
 for r in records:grouped[r['family'],r['split'],r['kind']].append(r)
 summary=[dict(zip(['family','split','kind'],key))|dict(n=len(rr),mean_effect=float(np.mean([r['effect'] for r in rr])),rms_effect=float(np.mean([r['effect']**2 for r in rr])**.5),mean_full_patch_KL=float(np.mean([r['full_patch_KL'] for r in rr])),mean_native_KL=float(np.mean([r['KL_to_native'] for r in rr])),mean_edit_norm=float(np.mean([r['edit_norm'] for r in rr]))) for key,rr in grouped.items()]
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),identity_logits_max_abs=identity,decomposition_error=decomp,orthogonality_error=orth,noise_norm_error=normerr,whole_model_forwards=actual,expected_forwards=640,finite=all(np.isfinite(r['effect']) and np.isfinite(r['full_patch_KL']) for r in records),test_evaluations=0,no_training=True)
 assert audits['hashes_unchanged'] and identity<1e-4 and decomp<1e-9 and orth<1e-8 and normerr<1e-8 and actual==640 and audits['finite']
 torch.save(dict(bases=bases,pca=pca,logits=torch.stack(logits)),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
