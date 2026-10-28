"""M15 frozen M13 predictions on passing context cohorts; no fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(a,b,assignment,amplitudes,families):
 target=b[assignment];pred=a*np.array(amplitudes)[:,None];energy=(target*target).sum(0);error=((pred-target)**2).sum(0);keep=np.array([f!='antonym' for f in families]);return dict(relative_squared_error=float(error.sum()/energy.sum()),flattened_cosine=float((pred*target).sum()/max(np.linalg.norm(pred)*np.linalg.norm(target),1e-20)),equal_family_mean_relative_error=float(np.mean(error/energy)),relative_error_without_antonym=float(error[keep].sum()/energy[keep].sum()),family_energy_shares=dict(zip(families,(energy/energy.sum()).tolist())),per_family_errors=dict(zip(families,(error/energy).tolist())))
def main():
 ip=P/'context_group_roles.json';r=json.loads(ip.read_text());cp=P/'context_challenge_protocol.json';challenge=json.loads(cp.read_text());mp=P/'group_role_matching.json';assert sha(mp)==challenge['matching_sha256'];m=json.loads(mp.read_text());screen=json.loads((P/'context_competence.json').read_text());matrices={};results=[];fidelity=[]
 for context in challenge['prefixes']:
  families=sorted(x['family'] for x in screen['cohorts'] if x['context']==context and x['pass_gate']);assert screen['context_aggregate_eligible'][context]
  for kind,grouping in [('scalar','contiguous'),('block','contiguous'),('block','shuffled')]:
   for seed in [71,72]:
    rows=[x for x in r['records'] if x['context']==context and x['kind']==kind and x['grouping']==grouping and x['seed']==seed];array=np.empty((32,len(families)))
    for g in range(32):
     for j,family in enumerate(families):
      rr=[x for x in rows if x['group']==g and x['family']==family];assert rr;array[g,j]=np.mean([x['margin_loss'] for x in rr])
    matrices[f'{context}:{kind}:{grouping}:{seed}']=array.tolist()
   for match in [x for x in m['matches'] if x['kind']==kind and x['grouping']==grouping]:
    a=np.array(matrices[f'{context}:{kind}:{grouping}:71']);b=np.array(matrices[f'{context}:{kind}:{grouping}:72']);idx=[m['families']['evaluation'].index(f) for f in families];oa=np.array(m['signatures'][f'{kind}:{grouping}:71:evaluation']['margin'])[:,idx];ob=np.array(m['signatures'][f'{kind}:{grouping}:72:evaluation']['margin'])[:,idx];results.append(dict(context=context,kind=kind,grouping=grouping,method=match['method'],families=families,assignment=match['assignment'],amplitudes=match['train_amplitudes'],current=score(a,b,match['assignment'],match['train_amplitudes'],families),original_same_families=score(oa,ob,match['assignment'],match['train_amplitudes'],families)))
  for kind in ['scalar','block']:
   for seed in [71,72]:
    rr=[x for x in r['baselines'] if x['context']==context and x['kind']==kind and x['seed']==seed];fidelity.append(dict(context=context,kind=kind,seed=seed,endpoints=len(rr),native_to_reconstruction_KL=float(np.mean([x['mean_KL'] for x in rr])),binary_rate=float(np.mean([x['task']['binary'] for x in rr])),task_top1_rate=float(np.mean([x['task']['top1'] for x in rr])),native_binary_rate=float(np.mean([x['native_task']['binary'] for x in rr])),native_task_top1_rate=float(np.mean([x['native_task']['top1'] for x in rr]))))
 out=dict(status='completed_no_refits',group_roles_sha256=sha(ip),frozen_matching_sha256=sha(mp),matrices=matrices,results=results,intact_fidelity=fidelity);(P/'context_role_transfer.json').write_text(json.dumps(out,indent=2)+'\n');print([(x['context'],x['kind'],x['grouping'],x['method'],round(x['current']['relative_squared_error'],3),round(x['original_same_families']['relative_squared_error'],3)) for x in results],flush=True)
if __name__=='__main__':main()
