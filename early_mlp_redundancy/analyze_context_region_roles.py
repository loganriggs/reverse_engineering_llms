"""M16 fixed region roles and nonadditivity; no fitting, matching or selection."""
import json,hashlib
from pathlib import Path
import numpy as np
from analyze_context_role_transfer import score
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return tuple(r[k] for k in ['id','endpoint','kind','seed','grouping','group'])
def main():
 rp=P/'context_region_roles.json';r=json.loads(rp.read_text());pp=P/'context_region_protocol.json';p=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];op=P/'context_group_roles.json';assert sha(op)==p['all_position_reference_sha256'];old=json.loads(op.read_text());mp=P/'group_role_matching.json';assert sha(mp)==p['matching_sha256'];matches=json.loads(mp.read_text());screen=json.loads((P/'context_competence.json').read_text());rows={region:{key(x):x for x in r['records'] if x['region']==region} for region in ['prefix','task']};whole={key(x):x for x in old['records']};assert all(set(x)==set(whole) for x in rows.values());matrices={};results=[];summary=[]
 for context in ['exercise','example']:
  families=sorted(x['family'] for x in screen['cohorts'] if x['context']==context and x['pass_gate'])
  for kind,grouping in [('scalar','contiguous'),('block','contiguous'),('block','shuffled')]:
   for seed in [71,72]:
    kk=[k for k,x in whole.items() if x['context']==context and x['kind']==kind and x['grouping']==grouping and x['seed']==seed];w=np.array([whole[k]['margin_loss'] for k in kk]);pre=np.array([rows['prefix'][k]['margin_loss'] for k in kk]);task=np.array([rows['task'][k]['margin_loss'] for k in kk]);interaction=w-pre-task;den=float(w@w)
    summary.append(dict(context=context,kind=kind,grouping=grouping,seed=seed,endpoint_group_records=len(kk),whole_margin_energy=den,prefix_margin_energy_over_whole=float(pre@pre/den),task_margin_energy_over_whole=float(task@task/den),interaction_energy_over_whole=float(interaction@interaction/den),prefix_final_KL=float(np.mean([rows['prefix'][k]['last_KL'] for k in kk])),task_final_KL=float(np.mean([rows['task'][k]['last_KL'] for k in kk])),whole_final_KL=float(np.mean([whole[k]['last_KL'] for k in kk])),prefix_noops=sum(rows['prefix'][k]['noop'] for k in kk),task_noops=sum(rows['task'][k]['noop'] for k in kk)))
    for region in ['prefix','task','interaction']:
     arr=np.zeros((32,len(families)));counts=np.zeros_like(arr)
     for k in kk:
      x=whole[k];g=x['group'];j=families.index(x['family']);value=x['margin_loss']-rows['prefix'][k]['margin_loss']-rows['task'][k]['margin_loss'] if region=='interaction' else rows[region][k]['margin_loss'];arr[g,j]+=value;counts[g,j]+=1
     arr/=counts;matrices[f'{context}:{kind}:{grouping}:{seed}:{region}']=arr.tolist()
   for region in ['prefix','task']:
    a=np.array(matrices[f'{context}:{kind}:{grouping}:71:{region}']);b=np.array(matrices[f'{context}:{kind}:{grouping}:72:{region}'])
    for match in [x for x in matches['matches'] if x['kind']==kind and x['grouping']==grouping]:
     results.append(dict(context=context,kind=kind,grouping=grouping,region=region,method=match['method'],families=families,assignment=match['assignment'],amplitudes=match['train_amplitudes'],score=score(a,b,match['assignment'],match['train_amplitudes'],families)))
 out=dict(status='completed_no_refits',records_sha256=sha(rp),protocol_sha256=sha(pp),summary=summary,matrices=matrices,results=results,interpretation='Region energies are squared margin-effect norms over individual endpoints and groups; not additive shares. Nonadditivity is of finite margin interventions. KL is scored separately, never subtracted as an additive causal effect. Role scores average within family first; their weighting differs from endpoint energy summaries.')
 (P/'context_region_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2));print([(x['context'],x['kind'],x['grouping'],x['region'],x['method'],round(x['score']['relative_squared_error'],3)) for x in results])
if __name__=='__main__':main()
