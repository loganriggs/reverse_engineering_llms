"""M22 independent Adam decomposition and one-step response accounting."""
from pathlib import Path
import json,hashlib,math
import numpy as np,torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logsoft(x):
 x=x.astype(float);x-=x.max();return x-np.log(np.exp(x).sum())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-10),(a,b)
rp=P/'response_adam_replay.json';r=json.loads(rp.read_text());pp=P/'response_adam_protocol.json';p=json.loads(pp.read_text());assert r['protocol_sha256']==sha(pp);assert r['status']=='exact_replay' and all(r['final_tensor_equal'].values()) and r['max_final_tensor_difference']==0;old=json.loads((P/'response_trajectory_replay.json').read_text());assert len(r['records'])==len(old['records'])==120
for x,y in zip(r['records'],old['records']):
 assert x.keys()==y.keys()
 for k in x:
  if isinstance(x[k],(int,float)):close(x[k],y[k])
  else:assert x[k]==y[k]
sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];data=torch.load(sp,weights_only=True);steps={x['step']:x for x in data['steps']};assert sorted(steps)==p['counterfactual_steps'];lp=logsoft(data['native'].numpy());nr=logsoft(data['native_changed'].numpy())-lp;weight=np.exp(lp);maxerr=0.;formulaerrors=[];momenterrors=[];rows={(x['step'],x['variant']):x for x in r['counterfactuals']};assert len(rows)==15
for s in data['counterfactual_steps']:
 step=s['step'];before=steps[step];g=s['training_gradient'].numpy().astype(float);m=s['previous_first_moment'].numpy().astype(float);v=s['previous_second_moment'].numpy().astype(float);mm=s['updated_first_moment'].numpy().astype(float);vv=s['updated_second_moment'].numpy().astype(float);b1,b2=s['beta1'],s['beta2'];time=s['time'];assert time==step+1;assert b1==.9 and b2==.999 and s['learning_rate']==.0003 and s['epsilon']==1e-8
 for pred,actual in [(b1*m+(1-b1)*g,mm),(b2*v+(1-b2)*g*g,vv)]:
  error=float(np.linalg.norm(pred-actual)/max(np.linalg.norm(actual),1e-30));momenterrors.append(error);assert error<1e-6
 factor=-s['learning_rate']/(1-b1**time)/(np.sqrt(vv/(1-b2**time))+s['epsilon']);carried=factor*b1*m;current=factor*(1-b1)*g;assert np.allclose(carried,s['carried_delta'].numpy(),rtol=1e-12,atol=1e-15);assert np.allclose(current,s['current_delta'].numpy(),rtol=1e-12,atol=1e-15);raw=s['raw_actual_delta'].numpy();error=float(np.linalg.norm(carried+current-raw)/np.linalg.norm(raw));formulaerrors.append(error);close(error,s['formula_relative_error']);assert error<1e-3;match=np.linalg.norm(raw)/np.linalg.norm(current);close(match,s['current_match_factor'])
 for variant in s['variants']:
  row=rows[step,variant['variant']];fr=logsoft(variant['perturbed_logits'].numpy())-logsoft(variant['logits'].numpy());value=float((weight*(fr-nr)**2).sum());maxerr=max(maxerr,abs(value-row['sentinel_loss']));close(value,row['sentinel_loss']);close(value-r['records'][step]['sentinel_before'],row['sentinel_change']);delta=variant['normalized_parameter_delta'].numpy().astype(float);grad=before['gradient'].numpy().astype(float);close(float(grad@delta),row['first_order_prediction']);assert math.isclose(np.linalg.norm(delta),row['normalized_update_norm'],rel_tol=1e-6,abs_tol=1e-9);rawnorm=np.linalg.norm(carried if variant['variant']=='carried_only' else current*(match if variant['variant']=='current_matched_raw_norm' else 1));close(rawnorm,row['raw_update_norm']);support=variant['support'].numpy();assert np.all(support.sum(-1)==32);diff=support!=before['old_support'].numpy();assert int(diff.sum())==row['flipped_coordinates'] and int(diff.any(-1).sum())==row['changed_tokens']
out=dict(status='passed',reference_steps_reproduced=120,final_tensors_equal=True,adam_decompositions=5,counterfactual_response_checks=15,max_response_error=maxerr,max_moment_relative_error=max(momenterrors),max_raw_update_formula_relative_error=max(formulaerrors),result_sha256=sha(rp),scope='All reference trajectory fields, independent NumPy first/second moment algebra and direction formulas, all counterfactual logits/norms/gradientdots/support checks. Counterfactuals share actual secondmoment and are not independently trained optimizers.')
(P/'response_adam_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
