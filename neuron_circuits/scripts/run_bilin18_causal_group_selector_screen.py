"""H1029 bounded cross-fitted group utility selector using saved discovery interventions."""
from pathlib import Path
import json,hashlib,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_causal_group_selector_screen';out=A/'causal_group_selector_screen';out.mkdir(exist_ok=True)
torch.set_num_threads(6)
paths=[A/'bilin18_group_utility_targets_inputs.pt',A/'bilin18_group_utility_targets_targets.pt'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};inp=torch.load(paths[0],weights_only=False);tar=torch.load(paths[1],weights_only=False);C=inp['coordinates'].double().numpy();L=tar['KL'].double().numpy();meta=inp['metadata'];arms=tar['arms'];scalar=arms.index('scalar32');domains=np.array([m['domain'] for m in meta]);family=np.array([m['unit'].split('-context')[0] if m['domain']=='old' else m['domain'] for m in meta]);unit=np.array([m['domain']+'/'+m['unit'] for m in meta]);fold=np.array([int(hashlib.sha256(u.encode()).hexdigest()[:8],16)%5 for u in unit]);panels=['real','shuffle10081','shuffle10082'];splits=[]
for f in sorted(set(family[domains!='natural'])):splits.append((f,np.where(family!=f)[0],np.where(family==f)[0]))
for k in range(5):
 te=np.where((domains=='natural')&(fold==k))[0];tr=np.where(~((domains=='natural')&(fold==k)))[0];splits.append(('natural_fold'+str(k),tr,te))
config={'experiment':'H1029','source_hashes':hashes,'features':'128 coordinates and their absolute values; train-only centering and per-feature standard deviations; intercept','estimator':'Ridge lambda1 on equal-domain weighted mean squared error of raw KL gain relative to scalar. Select largest positive predicted gain, otherwise scalar. No hyperparameter or threshold search.','splits':'Leave one historical family out from utility fitting; natural fivefold by document hash. Dictionary was fitted on these discovery data: selector-only transfer, NOT whole-featurizer holdout.','panels':panels,'stop':'If real selector fails to improve scalar and beat shuffled selectors, do not expand semantic gate fitting on this dictionary.','scope':'Saved native-to-reconstruction KL for exactly32coordinate actions; no paired-edit fidelity or new task discovery measured. Original whole-task holdout data not loaded.','ridge':1.0}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');features=np.concatenate([C,np.abs(C)],axis=1);records=[];fits=[];seen=np.zeros(len(C),int)
for split,tr,te in splits:
 assert not set(unit[tr])&set(unit[te]);seen[te]+=1
 mu=features[tr].mean(0);sd=features[tr].std(0);sd=np.maximum(sd,1e-8);X=np.c_[np.ones(len(tr)),(features[tr]-mu)/sd];V=np.c_[np.ones(len(te)),(features[te]-mu)/sd];w=np.zeros(len(tr))
 for d in set(domains[tr]):mask=domains[tr]==d;w[mask]=1/(len(set(domains[tr]))*mask.sum())
 penalty=np.eye(X.shape[1]);penalty[0,0]=0;gram=X.T@(w[:,None]*X)+penalty
 for panel in panels:
  ix=[i for i,a in enumerate(arms) if a.startswith(panel+'/')];Y=L[tr,scalar,None]-L[np.ix_(tr,ix)];coef=np.linalg.solve(gram,X.T@(w[:,None]*Y));pred=V@coef;pick=pred.argmax(1);chosen=np.where(pred.max(1)>0,np.array(ix)[pick],scalar);oracle=np.minimum(L[te,scalar],L[np.ix_(te,ix)].min(1));fits.append(dict(split=split,panel=panel,train=tr,test=te,center=mu,scale=sd,coef=coef,columns=ix))
  for k,i in enumerate(te):records.append(dict(row=int(i),domain=domains[i],family=family[i],split=split,panel=panel,chosen=arms[chosen[k]],KL=float(L[i,chosen[k]]),scalar_KL=float(L[i,scalar]),oracle_KL=float(oracle[k]),predicted_gain=float(max(0,pred[k].max()))))
assert np.all(seen==1);summary=[]
for domain in ['old','noun','natural']:
 for panel in panels:
  rr=[r for r in records if r['domain']==domain and r['panel']==panel];a=np.mean([r['KL'] for r in rr]);b=np.mean([r['scalar_KL'] for r in rr]);summary.append(dict(domain=domain,panel=panel,n=len(rr),mean_KL=float(a),scalar_KL=float(b),relative_improvement=float((b-a)/b),oracle_relative_improvement=float((b-np.mean([r['oracle_KL'] for r in rr]))/b),selected_group_fraction=float(np.mean([r['chosen']!='scalar32' for r in rr]))))
audit={'every_state_tested_once_per_panel':bool(np.all(seen==1)),'train_test_unit_overlap':0,'all_finite':bool(np.isfinite([r['KL'] for r in records]).all()),'source_hashes_unchanged':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in paths),'model_forwards':0,'original_whole_task_holdout_reads':0,'selector_fits':len(fits)};assert audit['source_hashes_unchanged'] and audit['all_finite'];torch.save(fits,out/'fits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,audits=audit),indent=2))
