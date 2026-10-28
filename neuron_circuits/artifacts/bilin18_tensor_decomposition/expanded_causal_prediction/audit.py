import json,numpy as np,torch
from pathlib import Path
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');p=A/'expanded_causal_prediction';a=np.load(p/'models_and_predictions.npz');old=np.load(A/'causal_source_prediction/captures.npz');meta=json.loads((p/'metadata.json').read_text());train=a['train'];assert len(train)==224
assert train.tolist()==[i for i,m in enumerate(meta) if m['split'] in ('discovery','expanded_discovery')]
assert not {meta[i]['token_sha256'] for i in train}&{m['token_sha256'] for i,m in enumerate(meta) if i not in train}
maxfit=maxfeature=maxold=0
clean=a['clean'];delta=a['delta'];features={'norms15':np.concatenate([np.linalg.norm(clean,axis=2),np.linalg.norm(delta,axis=2),np.einsum('npd,npd->np',clean,delta)/(np.linalg.norm(clean,axis=2)*np.linalg.norm(delta,axis=2)).clip(1e-12)],1)}
for rank in [4,8]:
 z=(a['X']-a['pca_mean'])@a['pca_basis'][:rank].T;features[f'pca{rank}_linear']=z;z=z/z[train].std(0).clip(1e-12);products=np.stack([z[:,i]*z[:,j] for i in range(rank) for j in range(i,rank)],1);features[f'pca{rank}_quadratic']=np.concatenate([z,products],1)
for name,ff in features.items():
 maxfeature=max(maxfeature,float(np.max(np.abs(ff-a[name+'_features']))));f=torch.tensor(ff);fm=f[train].mean(0);fs=f[train].std(0,correction=0).clamp_min(1e-12);h=(f-fm)/fs;ht=h[train];Y=torch.tensor(a['Y']);ym=Y[train].mean(0);w=torch.linalg.solve(ht.T@ht/224+.1*torch.eye(h.shape[1],dtype=torch.float64),ht.T@(Y[train]-ym)/224);pred=ym+h@w;maxfit=max(maxfit,float((pred-torch.tensor(a['prediction_expanded_'+name])).abs().max()));maxold=max(maxold,float(np.max(np.abs(a['prediction_old_'+name][:160]-old['prediction_'+name]))))
assert maxfit<1e-7 and maxfeature<1e-7 and maxold<1e-7
r={'independent_torch_fit_max_abs':maxfit,'independent_feature_construction_max_abs':maxfeature,'frozen_old_predictor_replay_max_abs':maxold,'train_eval_token_hash_overlap':0}
(p/'independent_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
r=json.loads((A/'bilin18_expanded_causal_prediction.json').read_text());print(r['controls'])
for split in ['context_ood','lexical_ood','joint_ood','historical_fresh','new_lexical_test','new_joint_test']:
 print(split,[(x['model'],round(x['joint_relative_l2'],3)) for x in r['summary'] if x['split']==split])
