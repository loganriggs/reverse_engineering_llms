"""All split/family/layer/site summaries of the native finite-change atlas."""
from pathlib import Path
import json,hashlib,statistics
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=V/'native_change_coordination.json';r=json.loads(rp.read_text());ap=V/'native_change_coordination_audit.json';a=json.loads(ap.read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);summaries=[]
 for split in ['train','source_eval']:
  for family in [None]+sorted({x['family'] for x in r['rows'] if x['split']==split}):
   for layer in [0,1,2]:
    for region in ['changed_input_positions','final_position']:
     allrows=[x for x in r['rows'] if x['split']==split and x['layer']==layer and x['region']==region and (family is None or x['family']==family)];rr=[x for x in allrows if x['status']=='evaluated' and x['S']>0];S=sum(x['S'] for x in rr);C=sum(x['C'] for x in rr);linear=sum(x['linear_only_error'] for x in rr);quad=sum(x['quadratic_energy'] for x in rr);N=4608
     def median(k):return statistics.median(x[k] for x in rr) if rr else None
     summaries.append(dict(split=split,family=family,layer=layer,region=region,assigned=len(allrows),nonzero=len(rr),ineligible=sum(x['status']=='ineligible' for x in allrows),S=S,C=C,C_over_S=C/S if S else None,median_C_over_S=median('C_over_S'),net_cancelling=sum(x['C']>x['S'] for x in rr),net_constructive=sum(x['C']<x['S'] for x in rr),median_participation=median('channel_participation'),linear_only_relative_error=linear/S if S else None,quadratic_relative_energy=quad/S if S else None,predicted_raw_half_relative=sum(x['predicted_raw_half_error'] for x in rr)/S if S else None,predicted_doubled_half_relative=sum(x['predicted_doubled_half_error'] for x in rr)/S if S else None,doubled_expected_better=sum(x['predicted_doubled_half_error']<x['predicted_raw_half_error'] for x in rr),mc_raw_relative=sum(x['mc_raw_mean'] for x in rr)/S if S else None,mc_doubled_relative=sum(x['mc_doubled_mean'] for x in rr)/S if S else None))
 op=V/'native_change_coordination_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),audit_sha256=sha(ap),summaries=summaries,scope='Directed endpoint counts include reciprocal pairs; no independence claim. Participation is in native channel basis and is not a number of semantic features. Aggregate ratios energy-weight conditions.'),indent=2)+'\n');print(json.dumps([x for x in summaries if x['family'] is None],indent=2))
if __name__=='__main__':main()
