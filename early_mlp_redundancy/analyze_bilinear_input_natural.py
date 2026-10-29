"""M88 all paired retention comparisons; no model selection."""
import json
from fit_pca_encoder import sha,V

def main():
 rp=V/'bilinear_input_natural.json';r=json.loads(rp.read_text());a=json.loads((V/'bilinear_input_natural_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);manifest=json.loads((V/'bilinear_input_evaluation_models.json').read_text())['models'];index={(x['kind'],x['seed'],x['arm']):x for x in r['summaries']};comparisons=[];oracles=[]
 for kind in ['dense','scalar','block','shared_fixed']:
  for seed in [71,72]:
   for variant in ['input','output_oracle']:
    left=index[kind,seed,'intact_KL__'+variant];right=index[kind,seed,'joint_three_corner__'+variant];comparisons.append(dict(kind=kind,seed=seed,variant=variant,control=left,joint=right,deltas={k:right[k]-left[k] for k in ['mean_KL','delta_CE','top1_agreement','local_relative_error','active_mean']}))
   for arm in ['intact_KL','joint_three_corner']:
    inp=index[kind,seed,arm+'__input'];oracle=index[kind,seed,arm+'__output_oracle'];oracles.append(dict(kind=kind,seed=seed,arm=arm,input_KL=inp['mean_KL'],oracle_KL=oracle['mean_KL'],input_local_error=inp['local_relative_error'],oracle_local_error=oracle['local_relative_error']))
 out=dict(status='completed',source_sha256=sha(__file__),natural_sha256=sha(rp),audit_sha256=sha(V/'bilinear_input_natural_audit.json'),comparisons=comparisons,oracle_comparisons=oracles,scope='Descriptive matched pairs across all32naturalheldoutdocs. Historical exposure retained. Oracle sees nativewrite and isnot a behavioralupperbound; no seed/modelselection orcausaltransferclaim.')
 p=V/'bilinear_input_natural_comparison.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
 for row in comparisons:
  if row['variant']=='input':print(row['kind'],row['seed'],row['control']['mean_KL'],row['joint']['mean_KL'],row['deltas']['local_relative_error'])
 print('oracleKLrange',min(x['oracle_KL'] for x in oracles),max(x['oracle_KL'] for x in oracles))
if __name__=='__main__':main()
