"""M38 independent affinity, proposal decisions, frozen states and local decode audit."""
from pathlib import Path
from collections import Counter
from itertools import combinations
import hashlib,json,math
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-7),(a,b)
def pairs(group):return set(combinations(sorted(group),2))
torch.set_num_threads(1);rp=P/'coactivation_groups.json';r=json.loads(rp.read_text());pp=P/'coactivation_group_protocol.json';p=json.loads(pp.read_text());assert r['protocol_sha256']==sha(pp) and r['source_sha256']==sha(P.parent/'build_coactivation_groups.py');ep=Path(r['evidence_path']);assert sha(ep)==r['evidence_sha256'];evidence=torch.load(ep,weights_only=True);m4=json.loads((P/'natural_token_context.json').read_text());assert sha(P/'natural_token_context.json')==p['natural_source_sha256'] and sha(Path(m4['cache']))==m4['cache_sha256'];data=torch.load(m4['cache'],weights_only=True);Y=data['Y'][1].numpy();tp=P/'response_training_targets.json';assert sha(tp)==p['target_source_sha256'];tm=json.loads(tp.read_text());assert sha(Path(tm['cache_path']))==tm['cache_sha256'];cache=torch.load(tm['cache_path'],weights_only=True);sp=P/'joint_shared_comparison_protocol.json';assert sha(sp)==p['training_schedule_sha256'];schedule=json.loads(sp.read_text());excluded=set(p['whole_task_exclusions']);checks=[];max_affinity=0.;max_metric=0.;max_support_ties=0
for source in p['source_models']:
 assert sha(Path(source['path']))==source['sha256'];state=torch.load(source['path'],weights_only=True);seed,arm=source['seed'],source['arm'];st=state['state'];encoder=st['encoder'].numpy().astype(float);bias=st['encoder_bias'].numpy().astype(float);mu=state['mean'].numpy().astype(float);scale=float(state['scale']);a=((Y[:4096].astype(float)-mu)/scale)@encoder.T+bias;en=a*a;cov=.5*(en.T@en)/4096;task=[]
 assignments=[x for x in schedule['target_assignments'] if x['seed']==seed];assert len(assignments)==120
 for assignment in assignments:
  assert all(tm['records'][i]['family'] not in excluded for i in assignment['target_indices']);v=cache['inputs'][assignment['input_key']]['write'].numpy().reshape(-1,1152);ac=((v.astype(float)-mu)/scale)@encoder.T+bias;en=ac*ac;cov+=.5*(en.T@en)/(120*len(v));task.append(v)
 diag=np.sqrt(np.diag(cov));den=diag[:,None]*diag[None,:];W=np.divide(cov,den,out=np.zeros_like(cov),where=den>0);np.fill_diagonal(W,0);e=next(x for x in evidence if x['seed']==seed and x['source_arm']==arm);err=float(np.max(np.abs(W-e['affinity'].numpy())));max_affinity=max(max_affinity,err);assert err<1e-10
 M=e['random_membership'].numpy();generator=torch.Generator().manual_seed(38000+seed);expected=np.zeros((32,128),int)
 for half in range(2):expected[np.arange(16)[:,None]+16*half,torch.randperm(128,generator=generator).numpy().reshape(16,8)]=1
 assert np.array_equal(M,expected);groups=[set(np.flatnonzero(row)) for row in M];counts=Counter(edge for g in groups for edge in pairs(g));accepted={x[0]:x for x in e['accepted_swaps']};rng=np.random.default_rng(39000+seed);naccept=0
 # Independent set-based edge accounting checks all accepted AND rejected proposals.
 for step in range(20000):
  half=int(rng.integers(2));g,h=rng.choice(16,2,replace=False)+half*16;ia=int(rng.choice(sorted(groups[g])));ib=int(rng.choice(sorted(groups[h])));ng=groups[g]-{ia}|{ib};nh=groups[h]-{ib}|{ia};change=Counter()
  for edge in pairs(groups[g]):change[edge]-=1
  for edge in pairs(groups[h]):change[edge]-=1
  for edge in pairs(ng):change[edge]+=1
  for edge in pairs(nh):change[edge]+=1
  gain=sum(W[edge]*((counts[edge]+d>0)-(counts[edge]>0)) for edge,d in change.items())
  if gain>1e-12:
   assert accepted.get(step)==[step,int(g),int(h),ia,ib],(seed,arm,step,gain);groups[g],groups[h]=ng,nh;counts.update(change);naccept+=1
  else:assert step not in accepted,(seed,arm,step,gain)
 learned=np.zeros((32,128),int)
 for i,g in enumerate(groups):learned[i,list(g)]=1
 assert naccept==len(accepted) and np.array_equal(learned,e['learned_membership'].numpy())
 for membership in [M,learned]:assert np.all(membership.sum(0)==2) and np.all(membership.sum(1)==8) and np.all(membership[:16].sum(0)==1) and np.all(membership[16:].sum(0)==1)
 sources=next(x for x in r['sources'] if x['seed']==seed and x['source_arm']==arm);close(sum(W[edge] for edge,c in counts.items() if c>0),sources['learned_objective']);assert sources['accepted_swaps']==naccept
 for entry in [x for x in r['models'] if x['seed']==seed and x['source_arm']==arm]:
  assert sha(Path(entry['path']))==entry['sha256'];candidate=torch.load(entry['path'],weights_only=True);cs=candidate['state'];grouping=entry['grouping']
  assert torch.equal(candidate['mean'],state['mean']) and torch.equal(candidate['scale'],state['scale'])
  for k,v in st.items():
   if k!='membership':assert torch.equal(v,cs[k])
  membership=cs['membership'].numpy()
  if grouping!='scalar':assert np.array_equal(membership,M if grouping=='random' else learned)
  D=cs['dictionary'].numpy().astype(float);D/=np.linalg.norm(D,axis=0,keepdims=True)
  for cohort,v in [('natural_train',Y[:4096]),('task_train',np.concatenate(task)),('natural_evaluation',Y[4096:])]:
   row=next(x for x in r['records'] if (x['seed'],x['source_arm'],x['grouping'],x['cohort'])==(seed,arm,grouping,cohort));x=((v-state['mean'].numpy())/scale).astype(np.float32);a=x.astype(float)@encoder.T+bias;scores=np.abs(a) if grouping=='scalar' else (a*a)@membership.T;k=32 if grouping=='scalar' else 4;selected=np.argsort(-scores,axis=-1)[:,:k];g=np.zeros_like(scores);np.put_along_axis(g,selected,1,axis=1);q=g if grouping=='scalar' else (g@membership>0).astype(float);pred=(a*q)@D.T+cs['bias'].numpy();recon=pred*scale+mu;relative=float(np.sum((recon-v)**2)/np.sum(v.astype(float)**2));mse=float(np.mean((pred-x)**2));close(relative,row['relative_write_error']);close(mse,row['normalized_mse']);max_metric=max(max_metric,abs(relative-row['relative_write_error']),abs(mse-row['normalized_mse']));assert len(v)==row['positions'];close(float(q.sum(1).mean()),row['active_mean']);assert int(q.sum(1).min())==row['active_min'] and int(q.sum(1).max())==row['active_max'];assert int(np.any(a*q!=0,axis=0).sum())==row['used_atoms']
 checks.append(dict(seed=seed,source_arm=arm,proposals_checked=20000,accepted=naccept,affinity_max_error=err))
out=dict(status='passed',result_sha256=sha(rp),protocol_sha256=sha(pp),checks=checks,model_checks=12,local_metric_checks=36,all_nonmembership_tensors_bitwise_frozen=True,max_affinity_error=max_affinity,max_local_metric_error=max_metric,scope='Independent NumPy affinity reconstruction from training data, set-based replay of all80000 accepted/rejected swap decisions, exact degree/partition constraints and all36 local decoded metrics. No native transfer or semantic validation.')
(P/'coactivation_groups_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
