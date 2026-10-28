"""H1001: discovery-only residual contrast spans with frozen transfer evaluation."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import forward,measure
from pathlib import Path
from collections import defaultdict
import torch,json,time,numpy as np
S='bilin18_residual_discovery_spans'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
 paths={'runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','screen':ART/'bilin18_residual_behavior_probes.json','captures':ART/'bilin18_residual_behavior_probes_captures.pt','eligibility':ART/'bilin18_residual_behavior_probes_eligibility.json','natural':ART/'bilin18_natural_residual_discovery_captures.pt','pca':ART/'bilin18_pca_behavior_transfer_bases.pt','model':SNAPSHOT/'pytorch_model.bin'}
 hashes={k:sha(p) for k,p in paths.items()}
 data=torch.load(paths['captures'],weights_only=False);natural=torch.load(paths['natural'],weights_only=False);pca=torch.load(paths['pca'],weights_only=False)['PCA128_expanded6']['basis'].double();screen=json.loads(paths['screen'].read_text());elig=json.loads(paths['eligibility'].read_text())
 rows=[r for r in screen['records'] if r['family'] in elig['candidate_families']];assert len(rows)==32 and all(r['same_final_token'] for r in rows)
 bases={};fits={};families=elig['candidate_families'];training_indices=[]
 for family in families:
  rr=[r for r in rows if r['family']==family and r['split']=='discovery'];assert len(rr)==4
  training_indices.extend(i for r in rr for i in r['indices'])
  contrasts=torch.stack([data['writes'][r['indices'][1]][12].double()-data['writes'][r['indices'][0]][12].double() for r in rr]).T
  residual=contrasts-pca@(pca.T@contrasts);u,s,v=torch.linalg.svd(residual,full_matrices=False);assert int((s>s.max()*1e-8).sum())==4
  fits[family]=dict(indices=[r['indices'] for r in rr],residual=residual,singular_values=s)
  for rank in [1,4]:bases[f'{family}_rank{rank}']=u[:,:rank]
 pooled=torch.cat([fits[f]['residual']/fits[f]['residual'].norm() for f in families],dim=1);u,s,v=torch.linalg.svd(pooled,full_matrices=False);assert int((s>s.max()*1e-8).sum())==8
 for rank in [1,4,8]:bases[f'pooled_rank{rank}']=u[:,:rank]
 for seed in [10011,10012]:
  c=torch.randn(1152,8,generator=torch.Generator().manual_seed(seed),dtype=torch.double);c-=pca@(pca.T@c);q=torch.linalg.qr(c).Q
  for rank in [1,4,8]:bases[f'random{seed}_rank{rank}']=q[:,:rank]
 assert set(training_indices)==set(elig['candidate_discovery_indices'])
 train_texts={data['examples'][i]['text'] for i in training_indices};eval_texts={data['examples'][i]['text'] for r in rows if r['split']!='discovery' for i in r['indices']};assert not train_texts&eval_texts
 geometry={'cross_family_principal_cosines':torch.linalg.svdvals(bases[f'{families[0]}_rank4'].T@bases[f'{families[1]}_rank4']).tolist(),'pooled_singular_values':s.tolist(),'family_singular_values':{f:fits[f]['singular_values'].tolist() for f in families}}
 torch.save(dict(bases=bases,pca=pca,fits=fits,training_indices=training_indices,geometry=geometry),ART/f'{S}_bases.pt');basis_hash=sha(ART/f'{S}_bases.pt')
 cfg=dict(experiment='H1001',hashes=hashes,layer=12,basis_hash=basis_hash,training_indices=training_indices,design='Four discovery pair differences per family, projected outside old task PCA128. Uncentered SVD retains ranks1 and4; pooled contrasts equalize family Frobenius norm and retain ranks1,4,8. Freeze all bases before any intervention. Evaluate both families and all cohorts, no outcome rank selection.',controls='Two random residual bases, nested ranks1,4,8. Identity, full donor, old task128, full residual, and old task128 plus pooled rank8. Rank curves are diagnostic, not selected on validation.',metrics='Signed donor-directed margin effect and KL(full-donor-patched output || arm output), within recipient context.',limits='Subspaces fitted from paired activation differences, not causal optimization. Donor-informed projection, not deployable predictor. Discovery fit is in-sample. Noun action and instrument covary; fragment source vocabulary differs across source slots. Cross-task transfer alone does not establish shared computation.',holdouts='Only eligible noun_expectation and word_fragment; all original whole-task holdouts and list_marker excluded.')
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 ex=data['examples'];buckets=defaultdict(list);decomp=0.;orth=0.;normerr=0.
 for name,b in bases.items():orth=max(orth,float((b.T@b-torch.eye(b.shape[1])).abs().max()),float((pca.T@b).abs().max()))
 for r in rows:
  for rec,i in enumerate(r['indices']):
   donor=r['indices'][1-rec];w=data['writes'][i][12].double();delta=data['writes'][donor][12].double()-w;task=pca@(pca.T@delta);res=delta-task
   edits={'identity':torch.zeros_like(delta),'donor':delta,'task128':task,'residual':res}|{name:b@(b.T@res) for name,b in bases.items()}
   edits['augmented136']=task+edits['pooled_rank8']
   decomp=max(decomp,float((task+res-delta).abs().max()))
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
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),identity_logits_max_abs=identity,decomposition_error=decomp,orthogonality_error=orth,noise_norm_error=normerr,whole_model_forwards=actual,expected_forwards=1152,finite=all(np.isfinite(r['effect']) and np.isfinite(r['full_patch_KL']) for r in records),test_evaluations=0,discovery_only_subspace_fit=True,bases_unchanged=sha(ART/f'{S}_bases.pt')==basis_hash,train_validation_prompt_overlap=0)
 assert audits['hashes_unchanged'] and audits['bases_unchanged'] and identity<1e-4 and decomp<1e-9 and orth<1e-8 and normerr<1e-8 and actual==1152 and audits['finite']
 torch.save(dict(bases=bases,pca=pca,logits=torch.stack(logits)),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,geometry=geometry,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
