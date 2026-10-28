"""M35 independent NumPy sampled metric audit plus full calibration accounting."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(z):
 x=z.numpy().astype(float);x-=x.max(-1,keepdims=True);return x-np.log(np.exp(x).sum(-1,keepdims=True))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
cp=P/'joint_shared_calibration.json';cal=json.loads(cp.read_text());pp=P/'joint_shared_comparison_protocol.json';p=json.loads(pp.read_text());assert cal['protocol_sha256']==sha(pp);assert cal['source_sha256']==sha(P.parent/'calibrate_joint_shared.py');assert cal['loss_source_sha256']==sha(P.parent/'joint_shared_loss.py')
tp=P/'response_training_targets.json';t=json.loads(tp.read_text());assert sha(tp)==p['target_source_sha256'];vp=Path(t['cache_path']);assert sha(vp)==t['cache_sha256'];cache=torch.load(vp,weights_only=True)
sp=Path(cal['samples_path']);assert sha(sp)==cal['samples_sha256'];samples=torch.load(sp,weights_only=True);assert len(samples)==12
assign={(x['seed'],x['task_update'],x['endpoint']):x for x in p['target_assignments']};seen=set();calls=0
for row in cal['rows']:
 key=(row['kind'],row['seed'],row['task_update'],row['endpoint']);assert key not in seen;seen.add(key);a=assign[key[1:]];assert row['target_indices']==a['target_indices'] and row['input_key']==a['input_key'];close(row['R'],sum(row['generator_R'])/2);assert all(math.isfinite(row[k]) and row[k]>=-1e-10 for k in ['K','R']);assert 0<=row['active_min']<=row['active_mean']<=row['active_max']<=32;calls+=1+sum(not t['records'][i]['noop'] for i in row['target_indices'])
assert len(seen)==720 and calls==cal['forward_count']==1989
for entry in p['initial_models']:
 rs=[x for x in cal['rows'] if (x['kind'],x['seed'])==(entry['kind'],entry['seed'])];assert len(rs)==120;mean=next(x for x in cal['model_means'] if (x['kind'],x['seed'])==(entry['kind'],entry['seed']))
 for k in ['K','R']:close(sum(x[k] for x in rs)/120,mean['mean_'+k])
for k in ['K','R']:close(sum(x['mean_'+k] for x in cal['model_means'])/6,cal['pooled_'+k])
close(cal['pooled_K']/cal['pooled_R'],cal['response_weight'])
errors=[]
for sample in samples:
 row=sample['row'];assert row in cal['rows'];c=cache['inputs'][row['input_key']];native=lp(c['native']);out=lp(sample['output']);K=float(np.sum(np.exp(native)*(native-out),axis=-1).mean());close(K,row['K']);errors.append(abs(K-row['K']))
 for j,(i,v) in enumerate(zip(row['target_indices'],sample['changed'])):
  native_response=lp(cache['targets'][i]['native_changed'])-native[0,-1];response=lp(v)-out[0,-1];R=float(np.sum(np.exp(native[0,-1])*(response-native_response)**2));close(R,row['generator_R'][j]);errors.append(abs(R-row['generator_R'][j]))
  if t['records'][i]['noop']:assert torch.equal(v,sample['output'][0,-1])
 support=sample['support'];assert torch.all((support==0)|(support==1));assert abs(float(support.double().sum(-1).mean())-row['active_mean'])<5e-6
gp=P/'joint_shared_gradient_checks.json';g=json.loads(gp.read_text());assert g['status']=='passed' and g['calibration_sha256']==sha(cp);assert g['protocol_sha256']==sha(pp);assert g['loss_source_sha256']==cal['loss_source_sha256'];assert g['source_sha256']==sha(P.parent/'check_joint_shared_gradients.py');assert len(g['checks'])==12
for x in g['checks']:
 assert x['identity_K']<1e-9 and x['identity_R']<1e-9
 for fd in x['finite_differences']:
  close((fd['plus']-fd['minus'])/(2*fd['epsilon']),fd['numeric']);close(abs(fd['numeric']-fd['analytic']),fd['absolute_error']);assert fd['relative_error']<.03 or fd['absolute_error']<1e-6
 for b in x['branch_checks']:
  if not b['noop']:assert b['intact_gradient_norm']>0 and b['changed_gradient_norm']>0
out=dict(status='passed',calibration_sha256=sha(cp),gradient_sha256=sha(gp),protocol_sha256=sha(pp),full_scheduled_row_checks=720,sampled_full_output_checks=12,sampled_generator_response_checks=24,finite_difference_checks=24,max_sample_metric_error=max(errors),scope='Full schedule, pooling, inventory and hash checks; NumPy metric replay on first prefix/task example per start. Not full independent replay of all720 native forwards; gradient audit checks saved finite-difference arithmetic and provenance.')
(P/'joint_shared_calibration_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
