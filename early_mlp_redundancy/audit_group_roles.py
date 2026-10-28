"""M13 NumPy ablation metrics and independent role-prediction accounting."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
def lse(x):
 m=x.max();return float(m+np.log(np.exp(x-m).sum()))
ip=P/'group_roles.json';r=json.loads(ip.read_text());mp=P/'group_role_matching.json';m=json.loads(mp.read_text());assert sha(ip)==m['group_roles_sha256'];assert sha(P/'group_roles_protocol.json')==r['protocol_sha256'];sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);errors=[];deltaerrors=[]
for s in samples:
 row=r['records'][s['row']];ref=s['intact'].numpy();out=s['ablated'].numpy();lp=logsoft(ref);lq=logsoft(out);kl=(np.exp(lp)*(lp-lq)).sum(-1);ids=s['ids'].numpy();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],ids[:,1:,None],axis=-1).mean(),top1_agreement=(ref.argmax(-1)==out.argmax(-1)).mean());atoms=s['atoms'].numpy();delta=(s['code'].numpy()[...,atoms].astype(float)@s['D'].numpy()[:,atoms].astype(float).T)*float(s['scale']);saved=s['contribution'].numpy();norm=max(np.linalg.norm(delta),1e-20);deltaerrors.append(float(np.linalg.norm(delta-saved)/norm));assert deltaerrors[-1]<1e-6;assert np.array_equal((s['q']-s['contribution']).numpy(),s['changed'].numpy());energy=np.sum(saved.astype(float)**2)/np.sum(s['q'].numpy().astype(float)**2);vals['removed_relative_energy']=energy
 for k,v in vals.items():errors.append(abs(float(v)-row[k]));assert errors[-1]<1e-7
 margins=[]
 for field,lg in [('intact_task',ref),('ablated_task',out)]:
  x=lg[0,-1].astype(float);g=s['groups'];v=s['variant'];margin=lse(x[g[v]])-lse(x[g[1-v]]);assert abs(margin-row[field]['margin'])<1e-10;margins.append(margin)
 assert abs(margins[0]-margins[1]-row['margin_loss'])<1e-10
assert len(r['records'])==9216
keys=set()
for row in r['records']:
 key=tuple(row[k] for k in ['id','endpoint','kind','seed','grouping','group']);assert key not in keys;keys.add(key)
 if row['noop']:assert row['margin_loss']==0 and row['mean_KL']==0 and row['mean_removed_active_coordinates']==0
for key,sig in m['signatures'].items():
 kind,grouping,seed,part=key.split(':');families=m['families'][part];matrix=np.zeros((32,len(families)));counts=np.zeros_like(matrix)
 for row in r['records']:
  if row['kind']==kind and row['grouping']==grouping and row['seed']==int(seed) and row['split']==part:
   g=row['group'];f=families.index(row['family']);matrix[g,f]+=row['margin_loss'];counts[g,f]+=1
 assert (counts>0).all();matrix/=counts;assert np.max(np.abs(matrix-np.array(sig['margin'])))<1e-12
predictionerrors=[]
for row in m['matches']:
 prefix=f"{row['kind']}:{row['grouping']}";a=np.array(m['signatures'][prefix+':71:train']['margin']);b=np.array(m['signatures'][prefix+':72:train']['margin']);at=np.array(m['signatures'][prefix+':71:evaluation']['margin']);bt=np.array(m['signatures'][prefix+':72:evaluation']['margin']);assignment=row['assignment'];assert sorted(assignment)==list(range(32));pred=[];target=[]
 for g,h in enumerate(assignment):
  norm=np.dot(a[g],a[g]);alpha=max(0,float(np.dot(a[g],b[h])/norm)) if norm>0 else 0;assert abs(alpha-row['train_amplitudes'][g])<1e-10;pred.append(alpha*at[g]);target.append(bt[h])
 pred=np.array(pred);target=np.array(target);error=float(sum(np.dot(p-t,p-t) for p,t in zip(pred,target))/sum(np.dot(t,t) for t in target));predictionerrors.append(abs(error-row['evaluation_relative_squared_error']));assert predictionerrors[-1]<1e-10;cos=float(np.vdot(pred,target)/max(np.linalg.norm(pred)*np.linalg.norm(target),1e-20));assert abs(cos-row['evaluation_flattened_cosine'])<1e-10
out=dict(status='passed',ablation_samples=len(samples),unique_intervention_records=len(keys),signature_matrices=len(m['signatures']),matching_predictions=len(m['matches']),max_metric_absolute_error=max(errors),max_group_decode_relative_error=max(deltaerrors),max_prediction_absolute_error=max(predictionerrors),group_roles_sha256=sha(ip),matching_sha256=sha(mp),scope='Independent NumPy samplegroupdecode/outputscores; exhaustiveuniqueness/noopchecks; familymeansandtrain-onlyamplitude/evalpredictionaccounting. Doesnotindependentlyreimplementfullnative model.')
(P/'group_roles_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
