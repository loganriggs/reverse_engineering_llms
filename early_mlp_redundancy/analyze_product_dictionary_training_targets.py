"""M96 paired training-target changes with exact objective contribution accounting."""
import json
from collections import defaultdict
import numpy as np
import torch
from fit_pca_encoder import V,sha

def main():
 torch.set_num_threads(1);ip=V/'product_dictionary_calibration.json';fp=V/'product_dictionary_training_targets.json';initial=json.loads(ip.read_text());final=json.loads(fp.read_text());ia=json.loads((V/'product_dictionary_calibration_audit.json').read_text());fa=json.loads((V/'product_dictionary_training_targets_audit.json').read_text());assert ia['status']==fa['status']=='passed' and ia['calibration_sha256']==sha(ip) and fa['result_sha256']==sha(fp)
 tp=json.loads((V/'balanced_response_training_protocol.json').read_text());den={(x['family'],x['region']):x['denominator'] for x in tp['scales']};weight=json.loads((V/'product_dictionary_training_protocol.json').read_text())['joint_weight'];initials={(x['seed'],x['assignment'],x['key']):x for x in initial['rows']};branches={}
 for data,label in [(initial,'initial'),(final,'final')]:
  for file in data['files']:
   assert sha(file['path'])==file['sha256'];samples=torch.load(file['path'],weights_only=True)
   for sample in samples:
    row=data['rows'][sample['row']];key=(row['seed'],row.get('arm','initial'),row['assignment'],row['key']);nonidentity=sum(b['region']!='identity' for b in sample['branches']);br={}
    for b in sample['branches']:
     normalized=b['R']/den[row['family'],b['region']] if b['region']!='identity' else 0.
     br[b['region'],b['corner']]=dict(raw=b['R'],normalized=normalized,contribution=normalized/nonidentity)
    assert np.isclose(sum(x['contribution'] for x in br.values()),row['B'],rtol=1e-12,atol=1e-12);branches[key]=br
 rows=[];contributions=[]
 for row in final['rows']:
  identity=row['seed'],row['assignment'],row['key'];before=initials[identity];assert before['family']==row['family'];br0=branches[row['seed'],'initial',row['assignment'],row['key']];br1=branches[row['seed'],row['arm'],row['assignment'],row['key']];assert set(br0)==set(br1)
  record={k:row[k] for k in ['seed','arm','assignment','key','family']}
  for metric in ['K','B']:
   record[metric+'_initial']=before[metric];record[metric+'_final']=row[metric];record[metric+'_change']=row[metric]-before[metric]
  record['joint_objective_initial']=before['K']+weight*before['B'];record['joint_objective_final']=row['K']+weight*row['B'];rows.append(record)
  for (region,corner),b in br1.items():
   contributions.append({k:row[k] for k in ['seed','arm','assignment','key','family']}|dict(region=region,corner=corner,initial=br0[region,corner],final=b))
 summaries=[];branch_summaries=[]
 for seed in [71,72]:
  for arm in ['intact_KL','joint_three_corner']:
   for family in [None]+sorted({r['family'] for r in rows}):
    rr=[r for r in rows if r['seed']==seed and r['arm']==arm and (family is None or r['family']==family)];assert rr
    metrics={field:float(np.mean([r[field] for r in rr])) for field in ['K_initial','K_final','K_change','B_initial','B_final','B_change','joint_objective_initial','joint_objective_final']}
    metrics['joint_objective_change']=metrics['joint_objective_final']-metrics['joint_objective_initial'];summaries.append(dict(seed=seed,arm=arm,family=family,n=len(rr),**metrics))
    bb=[r for r in contributions if r['seed']==seed and r['arm']==arm and (family is None or r['family']==family)]
    all_contribution0=all_contribution1=0.
    for region in ['changed_input_positions','final_position','identity']:
     for corner in ['native_reconstructed','reconstructed_native','reconstructed_reconstructed']:
      subset=[r for r in bb if r['region']==region and r['corner']==corner];a=sum(r['initial']['contribution'] for r in subset)/len(rr);b=sum(r['final']['contribution'] for r in subset)/len(rr);all_contribution0+=a;all_contribution1+=b
      branch_summaries.append(dict(seed=seed,arm=arm,family=family,region=region,corner=corner,n=len(subset),all_cases=len(rr),B_contribution_initial=a,B_contribution_final=b,B_contribution_change=b-a,mean_normalized_R_initial=float(np.mean([r['initial']['normalized'] for r in subset])) if subset else None,mean_normalized_R_final=float(np.mean([r['final']['normalized'] for r in subset])) if subset else None))
    assert np.isclose(all_contribution0,metrics['B_initial'],rtol=1e-12,atol=1e-12) and np.isclose(all_contribution1,metrics['B_final'],rtol=1e-12,atol=1e-12)
 out=dict(status='completed',source_sha256=sha(__file__),initial_sha256=sha(ip),final_sha256=sha(fp),initial_audit_sha256=sha(V/'product_dictionary_calibration_audit.json'),final_audit_sha256=sha(V/'product_dictionary_training_targets_audit.json'),joint_weight=weight,rows=rows,summaries=summaries,branch_summaries=branch_summaries,scope='All528finalcases paired with264initialcases, exact repeated schedules retained. Each branch contributes R/den/nonidentity_branch_count to caseB; group contribution divides by allcases including ineligible source cases. Contributions sum to balancedB; not independent causal attribution. Joint objective shown for both arms as a diagnostic; KLonly did not optimize B.')
 dest=V/'product_dictionary_training_targets_comparison.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print('COMPARED',len(rows),len(summaries),len(branch_summaries))
if __name__=='__main__':main()
