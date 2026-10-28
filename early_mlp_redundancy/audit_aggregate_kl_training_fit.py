"""M27 sampled objective replay and full training-fit aggregation."""
from pathlib import Path
import json,hashlib,math
import numpy as np,torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
rp=P/'aggregate_kl_training_fit.json';r=json.loads(rp.read_text());tp=P/'response_training_targets.json';t=json.loads(tp.read_text());assert sha(tp)==r['targets_sha256'];assert sha(P/'aggregate_kl_models.json')==r['training_results_sha256'];sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];samples=torch.load(sp,weights_only=True);weights={(x['kind'],x['seed']):x['response_weight'] for x in t['calibration']};errors=[]
for sample in samples:
 x=r['records'][sample['row']];lp=logsoft(sample['teacher'].numpy());lq=logsoft(sample['output'].numpy());K=float((np.exp(lp)*(lp-lq)).sum(-1).mean());nr=logsoft(sample['native_changed'].numpy())-lp[0,-1];fr=logsoft(sample['reconstructed_changed'].numpy())-lq[0,-1];R=float((np.exp(lp[0,-1])*(fr-nr)**2).sum());close(K,x['K']);close(R,x['R']);errors.extend([abs(K-x['K']),abs(R-x['R'])])
seen=set()
for x in r['records']:
 key=x['target_index'],x['arm'];assert key not in seen;seen.add(key);row=t['records'][x['target_index']];assert all(x[k]==row[k] for k in ['kind','seed','family','region']);assert x['initial_K']==row['initial_K'] and x['initial_R']==row['initial_R'];assert x['weight']==weights[x['kind'],x['seed']];close(x['combined'],x['K']+x['weight']*x['R'])
 if row['noop']:assert x['R']==0
assert seen=={(i,a) for i in range(480) for a in ['aggregate_KL:1.0:live','aggregate_KL:0.1:live']};assert r['forward_count']==1646
for s in r['summaries']:
 rr=[x for x in r['records'] if all(x[k]==s[k] for k in ['kind','seed','arm']) and (s['family'] is None or x['family']==s['family'])];assert len(rr)==s['scheduled_endpoints']
 for k in ['K','R','combined','initial_K','initial_R']:close(sum(x[k] for x in rr)/len(rr),s[k])
out=dict(status='passed',sample_objectives=len(samples),scheduled_arm_records=len(seen),summaries=len(r['summaries']),max_metric_error=max(errors),result_sha256=sha(rp),scope='Training-target diagnostic only: sampled independent NumPy objectives and all target/arm/calibration/aggregate checks; not evaluation transfer.')
(P/'aggregate_kl_training_fit_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
