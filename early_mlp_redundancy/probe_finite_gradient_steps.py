"""M25 frozen finite-direction test; no fitting or selection from results."""
from pathlib import Path
import json,hashlib,collections
import torch
from torch.nn import functional as F
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from collect_response_targets import response_error
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');pp=P/'finite_gradient_protocol.json';protocol=json.loads(pp.read_text());op=P/'finite_gradient_steps.json';assert not op.exists();p=json.loads((P/'response_supervision_protocol.json').read_text());t=json.loads((P/'response_training_targets.json').read_text());assert sha(P/'response_training_targets.json')==protocol['targets_sha256'];assert sha(P/'response_supervision_protocol.json')==protocol['training_protocol_sha256'];cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);gp=P/'family_gradient_compatibility.json';assert sha(gp)==protocol['gradient_results_sha256'];gr=json.loads(gp.read_text());assert json.loads((P/'family_gradient_audit.json').read_text())['result_sha256']==sha(gp);sp=Path(gr['samples_path']);assert sha(sp)==gr['samples_sha256'];gradients=torch.load(sp,weights_only=True);model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;calls=0;cohort=json.loads((P/'natural_token_context_cohort.json').read_text());natural={};records=[];summaries=[];details=[];samples=[]
 docs=sorted({s['natural_documents'][step] for s in p['schedules'].values() for step in range(0,120,2)});assert all(d<32 for d in docs)
 for d in docs:
  ids=torch.tensor([cohort['token_ids'][d]],device=device);out,cap=trace(model,ids);calls+=1
  with torch.no_grad():w=mlp.Down(cap[1]['products'])
  natural[d]=dict(ids=ids.cpu(),native=out.cpu(),write=w.cpu())
 for entry in p['initial_models']:
  kind,seed=entry['kind'],entry['seed'];assert sha(Path(entry['path']))==entry['sha256'];state=torch.load(entry['path'],weights_only=True);f=Featurizer(kind,seed).to(device);f.load_state_dict(state['state']);params=list(f.parameters());original=[v.detach().clone() for v in params];mu=state['mean'].to(device);scale=state['scale'].to(device);gm=next(x for x in gradients if (x['kind'],x['seed'])==(kind,seed));agg=gm['gradients'][gm['names'].index('schedule:combined')];weight=next(x['response_weight'] for x in t['calibration'] if (x['kind'],x['seed'])==(kind,seed));indices=[i for i,r in enumerate(t['records']) if (r['kind'],r['seed'])==(kind,seed)];schedule=p['schedules'][str(seed)];dc=collections.Counter(schedule['natural_documents'][step] for step in range(0,120,2));supports={}
  with torch.no_grad():
   for i in indices:
    key=t['records'][i]['input_key']
    if key not in supports:supports[key]=f((cache['inputs'][key]['write'].to(device)-mu)/scale)[1].cpu()
   for d in dc:supports['natural:'+str(d)]=f((natural[d]['write'].to(device)-mu)/scale)[1].cpu()
  # Reproduce the actual first scheduled Adam step (natural K, fresh moments).
  c=natural[schedule['natural_documents'][0]];ids=c['ids'].to(device);q=f((c['write'].to(device)-mu)/scale)[0]*scale+mu;out,_=trace.__wrapped__(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;lp=c['native'].to(device).double().log_softmax(-1);loss=(lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean();loss.backward();g=torch.cat([v.grad.detach().flatten().double().cpu() for v in params]);opt=torch.optim.Adam(params,lr=p['learning_rate']);opt.step();raw=torch.cat([(v.detach()-o).flatten().double().cpu() for v,o in zip(params,original)]);formula=-p['learning_rate']*g/(g.abs()+1e-8);assert torch.max(torch.abs(raw-formula))<1e-7
  with torch.no_grad():
   for v,o in zip(params,original):v.copy_(o)
  f.zero_grad(set_to_none=True);directions={'minibatch_adam':raw,'aggregate_gradient':-agg/agg.norm()*raw.norm()};detail=dict(kind=kind,seed=seed,first_document=schedule['natural_documents'][0],raw_norm=float(raw.norm()),adam_formula_max_error=float((raw-formula).abs().max()),conditions=[]);dsave=[]
  conditions=[('initial',0.,'live')]+[(name,m,mode) for name in directions for m in protocol['multipliers'] for mode in ['live','fixed']]
  for direction,mult,mode in conditions:
   proposed=torch.zeros_like(raw) if direction=='initial' else directions[direction]*mult;offset=0
   with torch.no_grad():
    for v,o in zip(params,original):v.copy_(o+proposed[offset:offset+v.numel()].reshape(v.shape).to(v));offset+=v.numel()
    if direction!='initial':f.dictionary.copy_(F.normalize(f.dictionary,dim=0))
   delta=torch.cat([(v.detach()-o).flatten().double().cpu() for v,o in zip(params,original)]);label=f'{direction}:{mult}:{mode}';detail['conditions'].append(dict(label=label,post_normalization_norm=float(delta.norm()),predicted_schedule_change=float(torch.dot(agg,delta))));dsave.append(dict(label=label,delta=delta));memo={};local=[]
   @torch.no_grad()
   def intact(key,c):
    nonlocal calls
    ids=c['ids'].to(device);x=(c['write'].to(device)-mu)/scale;pred,support,_=f(x);old=supports[key].to(device)
    if mode=='fixed':pred=(F.linear(x,f.encoder,f.encoder_bias)*old)@F.normalize(f.dictionary,dim=0).T+f.bias
    q=pred*scale+mu;out,_=trace(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;lp=c['native'].to(device).double().log_softmax(-1);K=float((lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean());changed=int((support!=old).any(-1).sum());return q,out,K,changed,int(support.shape[-2])
   with torch.no_grad():
    for i in indices:
     row=t['records'][i];key=row['input_key'];c=cache['inputs'][key]
     if key not in memo:memo[key]=intact(key,c)
     q,out,K,changed,tokens=memo[key];target=cache['targets'][i];delta_write=target['delta'].to(device)
     if row['noop']:alt=out[0,-1]
     else:
      ids=c['ids'].to(device);a,_=trace(model,ids,replace=lambda j,z,w:q-delta_write+mlp.Down_bias if j==1 else w);calls+=1;alt=a[0,-1]
     R=float(response_error(c['native'].to(device)[0,-1],target['native_changed'].to(device),out[0,-1],alt));rr=dict(kind=kind,seed=seed,condition=label,target_index=i,family=row['family'],K=K,R=R,combined=K+weight*R,live_support_changed_tokens=changed,tokens=tokens);records.append(rr);local.append(rr)
     if direction=='initial':assert abs(K-row['initial_K'])<1e-7 and abs(R-row['initial_R'])<1e-7
     if i==indices[0]:samples.append(dict(row=len(records)-1,native=c['native'],output=out.cpu(),native_changed=target['native_changed'],changed=alt.cpu()))
    nrows=[]
    for d,n in dc.items():
     _,out,K,changed,tokens=intact('natural:'+str(d),natural[d]);nrows.append(dict(document=d,count=n,K=K,live_support_changed_tokens=changed,tokens=tokens))
   NK=sum(x['K']*x['count']/60 for x in nrows);TK=sum(x['K'] for x in local)/120;TR=sum(x['R'] for x in local)/120;families={fam:{key:sum(x[key] for x in local if x['family']==fam)/sum(x['family']==fam for x in local) for key in ['K','R','combined']} for fam in sorted({x['family'] for x in local})};summaries.append(dict(kind=kind,seed=seed,condition=label,natural_K=NK,task_K=TK,task_R=TR,response_weight=weight,schedule_objective=(NK+TK+weight*TR)/2,families=families,natural_records=nrows));print('CONDITION',kind,seed,label,'OBJECTIVE',summaries[-1]['schedule_objective'],flush=True); (P/'finite_gradient_progress.json').write_text(json.dumps(dict(status='in_progress_not_audited',completed_conditions=len(summaries),summaries=summaries),indent=2)+'\n')
  with torch.no_grad():
   for v,o in zip(params,original):v.copy_(o)
  assert all(torch.equal(v,o) for v,o in zip(params,original));assert all(v.grad is None for v in model.parameters());details.append(detail);samples.append(dict(kind=kind,seed=seed,parameter_deltas=dsave,aggregate_gradient=agg,first_gradient=g,raw_adam_delta=raw))
 sp=A/'emlp_m25_finite_gradient_samples.pt';torch.save(samples,sp);out=dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),forward_count=calls,records=records,summaries=summaries,details=details,samples_path=str(sp),samples_sha256=sha(sp),checkpoint_sha256=receipt.weights_sha256);op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()
