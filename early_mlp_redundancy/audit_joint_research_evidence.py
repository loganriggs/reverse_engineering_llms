"""M54 reconcile common-target evidence without fitting or new evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 names=['joint_shared_native_fidelity','joint_shared_outputs','coactivation_native_fidelity','coactivation_outputs'];data={n:json.loads((V/(n+'.json')).read_text()) for n in names}
 audits={'joint_shared_native_fidelity':'joint_shared_native_audit','joint_shared_outputs':'joint_shared_outputs_audit','coactivation_native_fidelity':'coactivation_native_audit','coactivation_outputs':'coactivation_outputs_audit'}
 for n,a in audits.items():
  audit=json.loads((V/(a+'.json')).read_text());assert audit['status']=='passed';assert audit['result_sha256']==sha(V/(n+'.json'))
 native=data['joint_shared_native_fidelity'];out=data['joint_shared_outputs'];targets={}
 for r in native['records']:
  key=tuple(r[k] for k in ['seed','pair_id','endpoint','generator','region','group']);sig=tuple(r[k] for k in ['delta_sha256','native_logprob_response_energy','native_margin_energy','native_margin_effect']);assert targets.setdefault(key,sig)==sig
 assert len(targets)==704
 excluded={'sequence_copy','spatial_inverse','weekday_successor','role_binding','tense'};assert not excluded&{r['family'] for r in native['records']};assert not excluded&{r['family'] for r in out['rows'] if r['family']}
 rows=[]
 for m in out['models']:
  key={k:m[k] for k in ['kind','seed','arm']};nat=[r for r in out['rows'] if all(r[k]==v for k,v in key.items()) and r['domain']=='natural'];joint=[r for r in out['summaries'] if all(r[k]==v for k,v in key.items()) and r['domain']=='expanded' and r['split']=='evaluation' and r['template_split']=='evaluation' and r['gate']=='all' and r['family'] is None];assert len(nat)==32 and len(joint)==1;j=joint[0];rec=key|dict(natural_inputs=32,natural_local_relative_error=float(np.mean([r['local_relative_error'] for r in nat])),natural_mean_KL=float(np.mean([r['mean_KL'] for r in nat])),natural_active_mean=float(np.mean([r['active_mean'] for r in nat])),joint_development_endpoints=j['endpoints'],joint_development_KL=j['mean_KL'],joint_binary_rate=j['binary_rate'],joint_task_top1_rate=j['task_top1_rate'])
  for region in ['prefix','task']:
   rr=[r for r in native['records'] if all(r[k]==v for k,v in key.items()) and r['region']==region];en=sum(r['native_logprob_response_energy'] for r in rr);err=sum(r['logprob_response_error'] for r in rr);assert len(rr)==176;rec[region+'_response_error']=err/en
  rows.append(rec)
 count=0
 for row in rows:
  if row['arm']=='intact_KL_plus_native_response':
   control=next(r for r in rows if r['kind']==row['kind'] and r['seed']==row['seed'] and r['arm']=='intact_KL');count+=sum(row[k]<control[k] for k in ['prefix_response_error','task_response_error'])
 assert count==7
 # M39 sources share exactly the same external targets as M37.
 for r in data['coactivation_native_fidelity']['records']:
  key=tuple(r[k] for k in ['seed','pair_id','endpoint','generator','region','group']);assert targets[key]==tuple(r[k] for k in ['delta_sha256','native_logprob_response_energy','native_margin_energy','native_margin_effect'])
 result=dict(status='passed',source_sha256=sha(Path(__file__)),sources={n:sha(V/(n+'.json')) for n in names},models=rows,common_native_directions=704,causal_supervision_wins=7,causal_supervision_comparisons=12,whole_task_training_exclusions=sorted(excluded),fresh_whole_task_transfer_claim=False,new_task_discovery_claim=False,scope='Reaggregation of existing development evidence. M37/M39 common external targets verified. M53 targets depend on candidate and are deliberately not included as architecture-comparable scores.')
 op=V/'joint_research_evidence.json';assert not op.exists();op.write_text(json.dumps(result,indent=2)+'\n');print('PASSED',len(rows),'models;',len(targets),'common targets')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
