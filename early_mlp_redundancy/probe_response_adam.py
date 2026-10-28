"""M22 fixed-step Adam carry/current counterfactuals on exact replay."""
from pathlib import Path
import json,hashlib
import torch
from torch.nn import functional as F
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from collect_response_targets import response_error
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');pp=P/'response_supervision_protocol.json';p=json.loads(pp.read_text());rp=P/'response_training_results.json';r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(pp);tp=P/'response_training_targets.json';t=json.loads(tp.read_text());assert r['targets_sha256']==sha(tp);cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);kind,seed,arm='block',71,'intact_KL_plus_response';sentinel_index=198;sentinel=t['records'][sentinel_index];assert (sentinel['kind'],sentinel['seed'])==(kind,seed);initial=next(x for x in p['initial_models'] if (x['kind'],x['seed'])==(kind,seed));expected_entry=next(x for x in r['models'] if (x['kind'],x['seed'],x['arm'])==(kind,seed,arm));assert sha(Path(initial['path']))==initial['sha256'];assert sha(Path(expected_entry['path']))==expected_entry['sha256'];expected=torch.load(expected_entry['path'],weights_only=True);old_history=next(x['updates'] for x in r['histories'] if (x['kind'],x['seed'],x['arm'])==(kind,seed,arm));schedule=p['schedules'][str(seed)];weight=next(x['response_weight'] for x in t['calibration'] if (x['kind'],x['seed'])==(kind,seed));protocol=dict(source_sha256=sha(Path(__file__)),training_results_sha256=sha(rp),training_protocol_sha256=sha(pp),targets_sha256=sha(tp),initial=initial,expected_final=expected_entry,kind=kind,seed=seed,arm=arm,sentinel_index=sentinel_index,selection='Post-M20 diagnostic: block71 largest proportional final combined-objective regression; target198 largest final response error in this fit. No new confirmation or representativeness claim.',design='Exact original120updates, freshAdam.0003, fixedtargets/weights/dataorder, decoder normalization. Before eachupdate compute sentinelR and its parametergradient without accumulating optimizer gradients. After update compare actual sentinelR, newparams with oldsupport fixed, and gradient dot actual normalized parameter delta. Require allfinaltensors and training histories reproduce oldrun. No alternative optimizer/loss/data tuning.',counterfactual_steps=[79,80,81,82,83],counterfactuals='At same preupdate state, split Adam first moment into beta1*m_previous and (1-beta1)*current_gradient, retaining actual updated secondmoment/biascorrections. Evaluate carried-only,current-only,andcurrent-only scaled to actual raw update norm; apply same unit-column normalization. Record achieved normalized update norms and support flips. Restore actualpostupdate parameters and prove optimizer state unchanged. Matchednorm is before decoder normalization, not after. Post-selected localdiagnosis, not a reset-optimizer training experiment.',stopping='Exactly five fixed steps; no LR/beta/weight search. Preserve unchanged reference trajectory and require exactfinalstate and oldM21 sentinel records.')
 protocolpath=P/'response_adam_protocol.json'
 if protocolpath.exists():assert json.loads(protocolpath.read_text())==protocol
 else:protocolpath.write_text(json.dumps(protocol,indent=2)+'\n')
 op=P/'response_adam_replay.json';assert not op.exists();model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;state=torch.load(initial['path'],weights_only=True);f=Featurizer(kind,seed).to(device);f.load_state_dict(state['state']);mu=state['mean'].to(device);scale=state['scale'].to(device);params=list(f.parameters());names=[n for n,_ in f.named_parameters()];opt=torch.optim.Adam(params,lr=p['learning_rate']);calls=0;natural={};cohort=json.loads((P/'natural_token_context_cohort.json').read_text())
 for doc in sorted(set(schedule['natural_documents'][::2])):
  ids=torch.tensor([cohort['token_ids'][doc]],device=device);out,cap=trace(model,ids);calls+=1
  with torch.no_grad():v=mlp.Down(cap[1]['products'])
  natural[doc]=dict(ids=ids.cpu(),write=v.cpu(),native=out.cpu())
 lookup={(x['task_update'],x['endpoint']):i for i,x in enumerate(t['records']) if (x['kind'],x['seed'])==(kind,seed)};sd=cache['inputs'][sentinel['input_key']];st=cache['targets'][sentinel_index];sids=sd['ids'].to(device);sx=(sd['write'].to(device)-mu)/scale;sn=sd['native'].to(device)[0,-1];snc=st['native_changed'].to(device);sdelta=st['delta'].to(device).detach()
 def logits(q,ids,delta=None):
  nonlocal calls
  output,_=trace.__wrapped__(model,ids,replace=lambda i,z,w:(q if delta is None else q-delta)+mlp.Down_bias if i==1 else w);calls+=1;return output
 def sentinel_eval(old_support=None):
  if old_support is None:rec,support,_=f(sx)
  else:
   support=old_support;a=F.linear(sx,f.encoder,f.encoder_bias);rec=(a*support)@F.normalize(f.dictionary,dim=0).T+f.bias
  q=rec*scale+mu;a=logits(q,sids)[0,-1];b=logits(q,sids,sdelta)[0,-1];return response_error(sn,snc,a,b),support,a,b
 def train_loss(data,index=None):
  ids=data['ids'].to(device);v=data['write'].to(device);teacher=data['native'].to(device);rec,_,_=f((v-mu)/scale);q=rec*scale+mu;out=logits(q,ids);lp=teacher.double().log_softmax(-1);K=(lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean();R=K.new_zeros(())
  if index is not None:
   target=cache['targets'][index];metadata=t['records'][index];delta=target['delta'].to(device).detach();changed=out[0,-1] if metadata['noop'] else logits(q,ids,delta)[0,-1];R=response_error(teacher[0,-1],target['native_changed'].to(device).detach(),out[0,-1],changed)
  return K+(weight if index is not None else 0.)*R
 records=[];samples=[];counterfactuals=[];counter_samples=[]
 for step in range(p['steps']):
  opt.zero_grad();before_params=[x.detach().clone() for x in params];before,old_support,ba,bb=sentinel_eval();grad=torch.autograd.grad(before,params);assert all(x.grad is None for x in params);before_value=float(before.detach());old_support=old_support.detach();flat_grad=torch.cat([g.detach().reshape(-1) for g in grad]);sample=dict(step=step,before_logits=ba.detach().cpu(),before_perturbed_logits=bb.detach().cpu(),old_support=old_support.cpu(),gradient=flat_grad.cpu())
  if step%2==0:
   doc=schedule['natural_documents'][step];loss=train_loss(natural[doc]);loss.backward();keys=[f'natural:{doc}'];update_domain='natural';family=None
  else:
   u=schedule['task_updates'][step//2];keys=[f"expanded:{u['expanded_pair_id']}:{ep}" for ep in [0,1]];values=[]
   for ep in [0,1]:
    index=lookup[step//2,ep];tr=t['records'][index];loss=train_loss(cache['inputs'][tr['input_key']],index)/2;loss.backward();values.append(float(loss.detach()))
   loss=torch.tensor(sum(values));update_domain='task';family=u['family']
  assert keys==old_history[step]['keys'];lossvalue=float(loss.detach());assert abs(lossvalue-old_history[step]['loss'])<1e-7;assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in params)
  selected=step in protocol['counterfactual_steps']
  if selected:
   training_grad=[x.grad.detach().clone() for x in params];previous_m=[opt.state[x]['exp_avg'].detach().clone() for x in params];previous_v=[opt.state[x]['exp_avg_sq'].detach().clone() for x in params]
  opt.step()
  if selected:raw_after=[x.detach().clone() for x in params]
  with torch.no_grad():
   f.dictionary.copy_(F.normalize(f.dictionary,dim=0));delta=torch.cat([(x-old).reshape(-1) for x,old in zip(params,before_params)]);linear=float((flat_grad.double()*delta.double()).sum());after,new_support,aa,ab=sentinel_eval();fixed,_,fa,fb=sentinel_eval(old_support);after_value=float(after);fixed_value=float(fixed);changed=old_support!=new_support;tokens=int(changed.any(-1).sum());coordinates=int(changed.sum());row=dict(step=step,update_domain=update_domain,family=family,training_loss=lossvalue,reference_training_loss=old_history[step]['loss'],sentinel_before=before_value,sentinel_after=after_value,sentinel_after_old_support=fixed_value,total_change=after_value-before_value,smooth_change=fixed_value-before_value,gate_change=after_value-fixed_value,first_order_prediction=linear,smooth_nonlinear_residual=fixed_value-before_value-linear,changed_tokens=tokens,flipped_coordinates=coordinates,parameter_delta_norm=float(delta.norm()),gradient_norm=float(flat_grad.norm()));records.append(row);sample.update(after_logits=aa.cpu(),after_perturbed_logits=ab.cpu(),fixed_logits=fa.cpu(),fixed_perturbed_logits=fb.cpu(),new_support=new_support.cpu(),parameter_delta=delta.cpu());
   if selected:samples.append(sample)
  if selected:
   with torch.no_grad():
    actual_params=[x.detach().clone() for x in params];group=opt.param_groups[0];beta1,beta2=group['betas'];lr=group['lr'];eps=group['eps'];assert group['weight_decay']==0 and not group['amsgrad'];current=[];carried=[];updated_m=[opt.state[x]['exp_avg'].detach().clone() for x in params];updated_v=[opt.state[x]['exp_avg_sq'].detach().clone() for x in params];time=int(opt.state[params[0]]['step'].item());assert time==step+1
    for oldm,gg,vv in zip(previous_m,training_grad,updated_v):
     denom=(vv.double()/(1-beta2**time)).sqrt()+eps;factor=-lr/(1-beta1**time)/denom;carried.append(factor*(beta1*oldm.double()));current.append(factor*((1-beta1)*gg.double()))
    raw_delta=torch.cat([(x-old).reshape(-1) for x,old in zip(raw_after,before_params)]).double();cf=torch.cat([x.reshape(-1) for x in current]);mf=torch.cat([x.reshape(-1) for x in carried]);formula_error=float((cf+mf-raw_delta).norm()/raw_delta.norm());assert formula_error<1e-3;factor=float(raw_delta.norm()/cf.norm());variants={'carried_only':carried,'current_only':current,'current_matched_raw_norm':[x*factor for x in current]};local=[]
    for name,direction in variants.items():
     for param,before_param,dd in zip(params,before_params,direction):param.copy_((before_param.double()+dd).to(param.dtype))
     f.dictionary.copy_(F.normalize(f.dictionary,dim=0));value,ss,la,lb=sentinel_eval();post_delta=torch.cat([(x-old).reshape(-1) for x,old in zip(params,before_params)]);pre_delta=torch.cat([x.reshape(-1) for x in direction]);record=dict(step=step,variant=name,sentinel_loss=float(value),sentinel_change=float(value)-before_value,first_order_prediction=float((flat_grad.double()*post_delta.double()).sum()),raw_update_norm=float(pre_delta.norm()),normalized_update_norm=float(post_delta.norm()),changed_tokens=int((ss!=old_support).any(-1).sum()),flipped_coordinates=int((ss!=old_support).sum()));counterfactuals.append(record);local.append(dict(variant=name,logits=la.cpu(),perturbed_logits=lb.cpu(),support=ss.cpu(),normalized_parameter_delta=post_delta.cpu()))
    for param,actual in zip(params,actual_params):param.copy_(actual)
    assert all(torch.equal(opt.state[x]['exp_avg'],m) and torch.equal(opt.state[x]['exp_avg_sq'],v) for x,m,v in zip(params,updated_m,updated_v));assert all(torch.equal(x,a) for x,a in zip(params,actual_params))
    counter_samples.append(dict(step=step,beta1=beta1,beta2=beta2,learning_rate=lr,epsilon=eps,time=time,training_gradient=torch.cat([x.reshape(-1) for x in training_grad]).cpu(),previous_first_moment=torch.cat([x.reshape(-1) for x in previous_m]).cpu(),previous_second_moment=torch.cat([x.reshape(-1) for x in previous_v]).cpu(),updated_first_moment=torch.cat([x.reshape(-1) for x in updated_m]).cpu(),updated_second_moment=torch.cat([x.reshape(-1) for x in updated_v]).cpu(),raw_actual_delta=raw_delta.cpu(),current_delta=cf.cpu(),carried_delta=mf.cpu(),current_match_factor=factor,formula_relative_error=formula_error,variants=local))
    print('COUNTERFACTUAL',step,[(x['variant'],round(x['sentinel_loss'],6)) for x in counterfactuals if x['step']==step],flush=True)
  assert all(x.grad is None for x in model.parameters())
  if step%20==19:print('REPLAY',step+1,'SENTINEL_R',after_value,flush=True)
 equal={k:torch.equal(v.cpu(),expected['state'][k]) for k,v in f.state_dict().items()};maxdiff=max(float((v.cpu()-expected['state'][k]).abs().max()) for k,v in f.state_dict().items());ap=A/'emlp_m22_response_adam_samples.pt';torch.save(dict(native=sn.cpu(),native_changed=snc.cpu(),parameter_names=names,parameter_shapes=[list(x.shape) for x in params],steps=samples,counterfactual_steps=counter_samples),ap);out=dict(status='exact_replay' if all(equal.values()) else 'replay_mismatch',protocol_sha256=sha(protocolpath),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,final_tensor_equal=equal,max_final_tensor_difference=maxdiff,records=records,counterfactuals=counterfactuals,samples_path=str(ap),samples_sha256=sha(ap));op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',out['status'],'FORWARDS',calls,flush=True);assert all(equal.values())
if __name__=='__main__':main()
