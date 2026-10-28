"""H895 diagnostic, no new unit selection: disjoint discovery input subspaces."""
import json,itertools,time
from pathlib import Path
import torch,numpy as np
from run_bilin18_number_unit_recovery import ROOT,ART,SETS,sha
S='bilin18_number_subspace_stability'
def main():
 torch.set_num_threads(6);start=time.time();paths={'runner':Path(__file__),'states':ART/'bilin18_partner_subspace_fidelity_native_mlp_states.pt','map':ART/'bilin18_partner_subspace_fidelity.json','old':ART/'bilin18_multidomain_matched_competence_captures.pt','new':ART/'bilin18_number_transfer_screen_captures.pt','frozen_units':ART/'bilin18_number_unit_recovery_units.pt'};hashes={k:sha(p) for k,p in paths.items()};cfg=dict(experiment='H895',hashes=hashes,ranks=[1,2,4,7],scope='Existing disjoint discovery sets only. No new model forwards, unit selection, natural or reserved test evaluation. Labels used for diagnostic centroid classifiers, not PCA fitting.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');old=torch.load(paths['old'],weights_only=False);mapping=json.loads(paths['map'].read_text());oe=[old['examples'][i] for i in mapping['selection_original_indices']];os=torch.load(paths['states'],weights_only=False);new=torch.load(paths['new'],weights_only=False);frozen=torch.load(paths['frozen_units'],weights_only=False);fits={};pcrows=[];orth=0.;replay=0.
 for name,(source,pairs) in SETS.items():
  ex=oe if source=='old' else new['examples']
  for fmt in ['A','B']:
   ix=[i for i,e in enumerate(ex) if e['pair'] in pairs and e['family']=='task' and e['format']==fmt];assert len(ix)==8
   X=torch.stack([os[i,(12,'raw')] if source=='old' else new['states'][i][12] for i in ix]).double().cpu();X=X/(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();mean=X.mean(0);_,s,V=torch.linalg.svd(X-mean,full_matrices=False);V=V[:7];labels=torch.tensor([ex[i]['order'] for i in ix]);delta=X[labels==1].mean(0)-X[labels==0].mean(0);key=name+'_'+fmt;fits[key]=dict(X=X,mean=mean,V=V,s=s,labels=labels,delta=delta);orth=max(orth,float((V@V.T-torch.eye(7)).abs().max()));replay=max(replay,abs(1-abs(float(V[0]@frozen[key]['b']))))
   for j in range(7):
    z=(X-mean)@V[j];cc=torch.stack([z[labels==i].mean() for i in [0,1]]);pred=(z[:,None]-cc).square().argmin(1);pcrows.append(dict(unit=key,pc=j+1,variance_fraction=float(s[j]**2/s.square().sum()),training_number_accuracy=float((pred==labels).double().mean()),number_contrast_energy_fraction=float((delta@V[j]).square()/delta.square().sum())))
 geometry=[];transfer=[]
 for x,y in itertools.combinations(fits,2):
  for k in cfg['ranks']:
   sv=torch.linalg.svdvals(fits[x]['V'][:k]@fits[y]['V'][:k].T);geometry.append(dict(first=x,second=y,rank=k,mean_squared_cosine=float(sv.square().mean()),minimum_cosine=float(sv.min()),maximum_cosine=float(sv.max())))
 for x,F in fits.items():
  for y,G in fits.items():
   if x.split('_')[:-1]==y.split('_')[:-1]:continue # exclude same noun set, including alternate format
   for k in cfg['ranks']:
    V=F['V'][:k];train=(F['X']-F['mean'])@V.T;test=(G['X']-F['mean'])@V.T;centroids=torch.stack([train[F['labels']==i].mean(0) for i in [0,1]]);pred=(test[:,None,:]-centroids).square().sum(-1).argmin(1);centered=G['X']-G['mean'];transfer.append(dict(source=x,target=y,rank=k,accuracy=float((pred==G['labels']).double().mean()),target_number_contrast_capture=float((G['delta']@V.T).square().sum()/G['delta'].square().sum()),target_centered_variance_capture=float((centered@V.T).square().sum()/centered.square().sum())))
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),orthonormal_error=orth,first_PC_replay_error=replay,finite=all(np.isfinite(r['target_number_contrast_capture']) for r in transfer),whole_model_forwards=0,test_evaluations=0);assert audits['hashes_unchanged'] and orth<1e-10 and replay<1e-8 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,pc_diagnostics=pcrows,geometry=geometry,transfer=transfer,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits)
if __name__=='__main__':main()
