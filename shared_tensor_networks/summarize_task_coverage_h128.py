"""Audit-gated teacher-level summary for the frozen crossed coverage study."""
import argparse,json
from pathlib import Path
import numpy as np
from h128_analysis_io import sha,write,training,case_name,case_path

def summarize(root):
 protocol,expected=training(root);out=root/'evaluation';audit=json.loads((out/'audit.json').read_text());assert audit['status']=='passed' and audit['fits']==len(expected)
 provenance=json.loads((out/'audit_cases/protocol.json').read_text());assert provenance['results_sha256']==sha(out/'results.json') and provenance['evaluation_protocol_sha256']==sha(out/'protocol.json') and audit['provenance_sha256']==sha(out/'audit_cases/protocol.json')
 for name,h in provenance['sources'].items():assert sha(Path(name))==h
 rows=json.loads((out/'results.json').read_text());checked={r['name']:r for r in audit['records']};assert len(checked)==len(rows)==len(expected)
 assert {r['name'] for r in rows}=={case_name(r) for r in expected}
 for row in rows:
  assert checked[row['name']]['record_sha256']==sha(out/f'{row["name"]}.json') and json.loads((out/f'{row["name"]}.json').read_text())==row
 primary=str(protocol['adaptation']);teachers=protocol['teachers'];conditions=[];teacher_rows={};witnesses=[]
 for n in protocol['ns']:
  for tasks in protocol['tasks']:
   group=[r for r in rows if r['training_record']['n']==n and r['training_record']['tasks']==tasks]
   for initialization in ['random','teacher']:
    selected=[r for r in group if (r['training_record']['initialization']=='teacher')==(initialization=='teacher')];tm=[]
    for teacher in teachers:
     rr=[r for r in selected if r['training_record']['teacher']==teacher];assert len(rr)==(1 if initialization=='teacher' else len(protocol['starts']))
     t=dict(teacher=teacher,transfer=float(np.mean([np.mean(r['transfer'][primary]['errors']) for r in rr])),source=float(np.mean([np.mean(r['source_mse']) for r in rr])),recovered_features=float(np.mean([sum(c>.99 for c in r['component_cosines']) for r in rr])))
     tm.append(t)
     if initialization=='random':teacher_rows[n,tasks,teacher]=t
    conditions.append(dict(n=n,tasks=tasks,initialization=initialization,teacher_means=tm,median_transfer=float(np.median([t['transfer'] for t in tm])),median_source=float(np.median([t['source'] for t in tm])),median_teacher_mean_recovered=float(np.median([t['recovered_features'] for t in tm])),full_recovery=sum(all(c>.99 for c in r['component_cosines']) for r in selected),runs=len(selected)))
   for teacher in teachers:
    rr=[r for r in group if r['training_record']['teacher']==teacher];control=next(r for r in rr if r['training_record']['initialization']=='teacher')
    for r in rr:
     if r is not control:witnesses.append(dict(name=r['name'],loss_difference=r['train_mse']-control['train_mse'],lower_by_margin=r['train_mse']-control['train_mse']>1e-8))
 low,high=protocol['ns'];few,many=protocol['tasks'];equal_labels=low*many==high*few
 assert equal_labels or protocol['smoke'], 'Scientific comparison must have equal labels'
 differences=[teacher_rows[high,few,t]['transfer']-teacher_rows[low,many,t]['transfer'] for t in teachers]
 fixed={str(n):[teacher_rows[n,many,t]['transfer']-teacher_rows[n,few,t]['transfer'] for t in teachers] for n in protocol['ns']}
 resources=[json.loads((case_path(root,r)/'trained.json').read_text())['resources'] for r in expected]
 result=dict(smoke=protocol['smoke'],conditions=conditions,equal_budget=dict(eligible=equal_labels,labels=low*many if equal_labels else None,differences=differences if equal_labels else None,more_inputs_wins=sum(d<0 for d in differences) if equal_labels else None),fixed_input={n:dict(differences=d,more_tasks_wins=sum(v<0 for v in d)) for n,d in fixed.items()},lower_loss_witnesses=witnesses,training_seconds=sum(r['seconds'] for r in resources),audit_sha256=sha(out/'audit.json'))
 lines=[f'# d{protocol["d"]}/h{protocol["h"]} coverage results','',('SMOKE FIXTURE ONLY — not scientific scaling evidence.' if protocol['smoke'] else 'All scientific cases passed the independent audit. Five teacher blocks, two random starts and a privileged teacher-start control per condition.'),'','Transfer MSE uses whole-task holdouts and the fixed primary adaptation budget. Medians are across teacher means; random starts are averaged within teacher. Full recovery requires every matched feature cosine to exceed.99. Finite-budget outcomes are not convergence certificates.','','| Inputs | Tasks | Start | Transfer MSE | Mean features, teacher median | Full recovery |','|---:|---:|---|---:|---:|---:|']
 for c in conditions:lines.append(f'|{c["n"]}|{c["tasks"]}|{c["initialization"]}|{c["median_transfer"]:.6g}|{c["median_teacher_mean_recovered"]:.3g}|{c["full_recovery"]}/{c["runs"]}|')
 lines+=['',(f'Equal label budget: more inputs wins on{sum(d<0 for d in differences)}/{len(teachers)}teachers. Paired differences and fixed-input comparisons remain in summary.json.' if equal_labels else 'No equal-label comparison: the tiny smoke grid does not have matching label budgets.'),f'{sum(w["lower_by_margin"] for w in witnesses)}/{len(witnesses)}random-start endpoints have lower-loss same-condition teacher-start witnesses. This does not certify strict local minima.',f'Total training time across independent model runs: {result["training_seconds"]/3600:.3f}hours, excluding preparation and analysis.','', 'Audit scope: input/checkpoint integrity, data regeneration, literal empirical predictions, Gaussian feature matching, exact population metrics through the shared verified Wick kernel, and independent SVD transfer solves. It does not replay every optimizer step. This panel does not establish depth3–5 recovery or a general scaling law.']
 write(out/'summary.json',result);(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);summarize(parser.parse_args().root)
