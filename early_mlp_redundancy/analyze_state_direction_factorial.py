"""M63 paired objective comparisons for each fixed factorial corner."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=V/'state_direction_factorial.json';r=json.loads(rp.read_text());audit=json.loads((V/'state_direction_factorial_audit.json').read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(rp);models=json.loads((V/'balanced_response_training_results.json').read_text())['models'];rows=[];comparisons=[]
 for e in models:
  for region in ['changed_input_positions','final_position']:
   entry={k:e[k] for k in ['kind','seed','arm']}|dict(region=region);gs=[x for x in r['geometry'] if x['region']==region and all(x[k]==e[k] for k in ['kind','seed','arm'])];entry['local_delta_error']=sum(x['delta_error'] for x in gs)/sum(x['target_delta_energy'] for x in gs);entry['predicted_native_delta_energy_ratio']=sum(x['predicted_delta_energy'] for x in gs)/sum(x['target_delta_energy'] for x in gs)
   for corner in ['reconstructed_native','native_reconstructed','reconstructed_reconstructed']:
    ss=[x for x in r['summaries'] if x['region']==region and x['corner']==corner and x['family'] is not None and all(x[k]==e[k] for k in ['kind','seed','arm'])];assert len(ss)==6;entry[corner+'_family_mean']=sum(x['relative_response_error'] for x in ss)/6;entry[corner+'_pooled']=sum(x['error_energy'] for x in ss)/sum(x['target_energy'] for x in ss)
   rows.append(entry)
 for region in ['changed_input_positions','final_position']:
  for metric in ['local_delta_error']+[corner+suffix for corner in ['reconstructed_native','native_reconstructed','reconstructed_reconstructed'] for suffix in ['_family_mean','_pooled']]:
   pairs=[]
   for kind in ['scalar','block','shared_fixed']:
    for seed in [71,72]:
     b=next(x for x in rows if (x['kind'],x['seed'],x['arm'],x['region'])==(kind,seed,'KL_balanced_response',region));raw=next(x for x in rows if (x['kind'],x['seed'],x['arm'],x['region'])==(kind,seed,'KL_raw_response',region));pairs.append(dict(kind=kind,seed=seed,balanced=b[metric],raw=raw[metric],balanced_better=b[metric]<raw[metric]))
   comparisons.append(dict(region=region,metric=metric,balanced_vs_raw_wins=sum(x['balanced_better'] for x in pairs),pairs=pairs))
 op=V/'state_direction_factorial_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),rows=rows,comparisons=comparisons,scope='All fixed candidates/corners, no selection. Family means weight six development families equally; pooled and local energy metrics separate.'),indent=2)+'\n');print([(c['region'],c['metric'],c['balanced_vs_raw_wins']) for c in comparisons if c['metric'].endswith('family_mean')])
if __name__=='__main__':main()
