"""M21 NumPy replay accounting for all120 sentinel steps."""
from pathlib import Path
import json,hashlib,math
import numpy as np,torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(float);x-=x.max();return x-np.log(np.exp(x).sum())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-10),(a,b)
rp=P/'response_trajectory_replay.json';r=json.loads(rp.read_text());pp=P/'response_replay_protocol.json';p=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];assert r['status']=='exact_replay' and all(r['final_tensor_equal'].values()) and r['max_final_tensor_difference']==0;sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];data=torch.load(sp,weights_only=True);lp=logsoft(data['native'].numpy());nr=logsoft(data['native_changed'].numpy())-lp;weight=np.exp(lp);old=json.loads((P/'response_training_results.json').read_text());assert sha(P/'response_training_results.json')==p['training_results_sha256'];history=next(x['updates'] for x in old['histories'] if (x['kind'],x['seed'],x['arm'])==(p['kind'],p['seed'],p['arm']));errors=[]
assert len(data['steps'])==len(r['records'])==120
for i,(s,x) in enumerate(zip(data['steps'],r['records'])):
 assert s['step']==x['step']==i;close(x['training_loss'],history[i]['loss']);close(x['reference_training_loss'],history[i]['loss'])
 values={}
 for prefix,key in [('before','sentinel_before'),('after','sentinel_after'),('fixed','sentinel_after_old_support')]:
  fr=logsoft(s[prefix+'_perturbed_logits'].numpy())-logsoft(s[prefix+'_logits'].numpy());value=float((weight*(fr-nr)**2).sum());errors.append(abs(value-x[key]));close(value,x[key]);values[prefix]=value
 grad=s['gradient'].numpy().astype(float);delta=s['parameter_delta'].numpy().astype(float);close(float(grad@delta),x['first_order_prediction']);assert math.isclose(float(np.linalg.norm(delta)),x['parameter_delta_norm'],rel_tol=1e-6,abs_tol=1e-9);assert math.isclose(float(np.linalg.norm(grad)),x['gradient_norm'],rel_tol=1e-6,abs_tol=1e-9)
 oldsupport=s['old_support'].numpy();new=s['new_support'].numpy();assert np.all(oldsupport.sum(-1)==32) and np.all(new.sum(-1)==32);changed=oldsupport!=new;assert int(changed.any(-1).sum())==x['changed_tokens'] and int(changed.sum())==x['flipped_coordinates'];close(values['after']-values['before'],x['total_change']);close(values['fixed']-values['before'],x['smooth_change']);close(values['after']-values['fixed'],x['gate_change']);close(x['smooth_change']-x['first_order_prediction'],x['smooth_nonlinear_residual'])
 if not changed.any():close(values['after'],values['fixed'])
 if i:close(r['records'][i-1]['sentinel_after'],x['sentinel_before'])
close(sum(x['total_change'] for x in r['records']),r['records'][-1]['sentinel_after']-r['records'][0]['sentinel_before']);fit=json.loads((P/'response_training_fit.json').read_text());target=next(x for x in fit['records'] if x['target_index']==p['sentinel_index'] and x['arm']==p['arm']);close(target['R'],r['records'][-1]['sentinel_after'])
out=dict(status='passed',steps=120,independent_response_scores=360,gradient_update_dot_products=120,max_metric_error=max(errors),final_state_exact_match=True,final_sentinel_matches_M20=True,result_sha256=sha(rp),scope='All saved logits independently scored; first-order/decomposition/support/continuity/training-history accounting. Final tensor equality checked by replay and final loss checked against M20. Native transformer and analytic gradients not independently reimplemented.')
(P/'response_replay_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
