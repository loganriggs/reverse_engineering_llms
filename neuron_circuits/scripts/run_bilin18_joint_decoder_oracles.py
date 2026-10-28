"""H990 frozen decoder/support ceilings and train-only PCA change predictors."""
from run_bilin18_joint_change_featurizer import ROOT,ART,sha
from feature_structures.models import SharedGroups
from pathlib import Path
import torch,json,time
S='bilin18_joint_decoder_oracles'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();files=sorted((ART/'expanded_joint_featurizer').glob('*.pt'));assert len(files)==24
 paths=[Path(__file__),ROOT/'scripts/feature_structures/models.py',ART/'bilin18_context_training_bank_captures.pt',ART/'bilin18_task_bank_causal_sites_captures.pt']+files+[ART/f'joint_change_featurizer/{k}_{o}_{seed}.pt' for k in ['block','shared_fixed'] for o in ['reconstruction','paired_changes'] for seed in [9831,9832]];hashes={str(p):sha(p) for p in paths};new=torch.load(paths[2],weights_only=False,map_location='cpu')['writes'];old=torch.load(paths[3],weights_only=False,map_location='cpu')['writes']
 cfg=dict(experiment='H990',hashes=hashes,design='All24 H989 frozen models. Encoded donor-delta error, least-squares true-delta projection into full128-atom decoder span, and into union of actual recipient/donor encoder-selected supports. The union is a permissive ceiling, not independently estimated endpoint coordinates. Train-only centered PCA32 and PCA128, capped at observed numerical rank (singular values>1e-6 times largest). No fitting to validation other than explicitly teacher-informed oracle coefficients.',limits='Oracles see true validation changes and do not define deployable encoders. Activation-space Euclidean error is not behavioral KL. PCA uses dense coordinates, a different sparsity/parameter budget. Frozen decoder rank checked before QR support reduction; no learned atom or group discovery claim.')
 (ROOT/'configs'/f'{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');records=[];audits=[]
 for p in files:
  d=torch.load(p,weights_only=False,map_location='cpu');name=p.stem;kind='shared_fixed' if name.startswith('shared_fixed') else 'block';objective='paired_changes' if '_paired_changes_' in name else 'reconstruction';regime=next(x for x in ['frozen_small6','expanded6','expanded7'] if x in name);seed=int(name.rsplit('_',1)[1]);meta=d['metadata'];pairs=d['pairs'];X=torch.stack([(new if r['source']=='context' else old)[r['index']][12] for r in meta]).double().cuda();a=[q['indices'][0] for q in pairs];b=[q['indices'][1] for q in pairs];delta=X[b]-X[a];energy=delta.square().sum(1)
  model=SharedGroups(1152,128,32,8 if kind=='block' else 4,4 if kind=='block' else 8).cuda();model.load_state_dict(d['model']);Z=((X.float()-d['center'].cuda())/d['scale'].cuda());outs=[model(z,membership=d['membership'].cuda()) for z in Z.split(128)];supports=torch.cat([o['support'] for o in outs]).bool();pred=torch.cat([o['reconstruction'] for o in outs])*d['scale'].cuda()+d['center'].cuda();replay=float((pred.cpu()-d['predicted_writes']).abs().max());assert replay<1e-4
  D=model.dictionary_matrix().double();sv=torch.linalg.svdvals(D);assert float(sv[-1]/sv[0])>1e-10;Q,R=torch.linalg.qr(D,mode='reduced');y=delta@Q;full=delta-y@Q.T;full_error=full.square().sum(1);unions=supports[a]|supports[b];support_error=torch.empty_like(full_error);projection_audit=0.;orthogonality=0.
  for i in range(len(pairs)):
   mask=unions[i];q,_=torch.linalg.qr(R[:,mask],mode='reduced');res=y[i]-(y[i]@q)@q.T;support_error[i]=full_error[i]+res.square().sum();orthogonality=max(orthogonality,float((q.T@res).abs().max())/float(delta[i].norm()))
   if i<2:
    direct,_=torch.linalg.qr(D[:,mask],mode='reduced');de=(delta[i]-(delta[i]@direct)@direct.T).square().sum();projection_audit=max(projection_audit,float(abs(de-support_error[i])/energy[i]))
  encoded=(d['predicted_writes'].double().cuda()[b]-d['predicted_writes'].double().cuda()[a]-delta).square().sum(1)
  assert bool((full_error<=support_error+1e-7*energy).all()) and bool((support_error<=encoded+1e-5*energy).all())
  if regime=='frozen_small6':
   op=ART/f'joint_change_featurizer/{kind}_{objective}_{seed}.pt';od=torch.load(op,weights_only=False,map_location='cpu');train=torch.stack([old[r['index']][12] for r in od['metadata'] if r['split']=='discovery' and not r['holdout']]).double().cuda()
  else:train=X[d['training_indices']]
  centered=train-train.mean(0);_,s,vh=torch.linalg.svd(centered,full_matrices=False);rank=int((s>s[0]*1e-6).sum());methods={'encoded':encoded,'full_decoder_oracle':full_error,'selected_union_oracle':support_error};ranks={}
  for n in [32,128]:
   nactual=min(n,rank);basis=vh[:nactual].T;methods[f'PCA{n}']=(delta-(delta@basis)@basis.T).square().sum(1);ranks[f'PCA{n}']=nactual
  for source,f,context,split in sorted({(q['source'],q['family'],q['context'],q['split']) for q in pairs}):
   ii=[i for i,q in enumerate(pairs) if (q['source'],q['family'],q['context'],q['split'])==(source,f,context,split)];den=float(energy[ii].sum())
   for method,err in methods.items():records.append(dict(model=name,kind=kind,objective=objective,regime=regime,seed=seed,source=source,family=f,context=context,split=split,native_pass=pairs[ii[0]]['native_pass'],method=method,error_ratio=float(err[ii].sum())/den,union_atoms=float(unions[ii].sum(1).double().mean())))
  audits.append(dict(model=name,decoder_rank=128,decoder_smallest_relative_singular_value=float(sv[-1]/sv[0]),pca_numerical_rank=rank,pca_ranks=ranks,prediction_replay_max_abs=replay,direct_projection_max_relative_error=projection_audit,projection_orthogonality_relative=orthogonality,oracle_ordering_pass=True));print('DONE',name,'rank',rank,flush=True)
 assert all(sha(p)==hashes[str(p)] for p in paths);assert max(r['direct_projection_max_relative_error'] for r in audits)<1e-10
 (ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,audits=audits,hashes_unchanged=True,whole_model_forwards=0,elapsed_seconds=time.time()-start),indent=2)+'\n');print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
