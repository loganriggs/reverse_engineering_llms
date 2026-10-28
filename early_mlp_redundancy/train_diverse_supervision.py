"""M11 execute the frozen M10 mixed-supervision protocol."""
import json,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent;A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');pp=P/'validation/diverse_supervision_protocol.json';protocol=json.loads(pp.read_text());sp=P/'validation/diverse_supervision_split.json';split=json.loads(sp.read_text());assert sha(sp)==protocol['split_sha256'];assert json.loads((P/'validation/diverse_supervision_split_audit.json').read_text())['protocol_sha256']==sha(pp);panel=json.loads((P/'validation/training_source_context_audit.json').read_text());cohort=json.loads((P/'validation/natural_token_context_cohort.json').read_text());trainids=set(split['train_pair_ids']);evalids=set(split['evaluation_pair_ids']);pairs={x['id']:x for x in panel['rows']};dest=A/'emlp_m11_diverse_supervision';dest.mkdir(exist_ok=True)
 (P/'validation/diverse_supervision_run_receipt.json').write_text(json.dumps(dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),status='started_fixed_protocol'),indent=2)+'\n')
 model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;cache={};calls=0
 def collect(key,tokens,meta):
  nonlocal calls
  ids=torch.tensor([tokens],device=device);out,cap=trace(model,ids);calls+=1
  with torch.no_grad():v=mlp.Down(cap[1]['products']).cpu()
  cache[key]=dict(ids=ids.cpu(),teacher=out.cpu(),write=v,**meta)
 for d,tokens in enumerate(cohort['token_ids']):collect(f'natural:{d}',tokens,dict(domain='natural',document=d,split='train' if d<32 else 'evaluation',family=None))
 for pair in panel['rows']:
  for ep in [0,1]:collect(f"task:{pair['id']}:{ep}",pair['token_ids'][ep],dict(domain='task',id=pair['id'],endpoint=ep,split='train' if pair['id'] in trainids else 'evaluation',family=pair['family'],groups=pair['answer_groups'][ep],variant=int(pair['variants'][ep])))
 print('TEACHERS',len(cache),flush=True);models=[];histories=[];rows=[];samples=[]
 def kl_forward(f,mu,scale,key):
  nonlocal calls
  c=cache[key];ids=c['ids'].to(device);v=c['write'].to(device);pred,_,_=f((v-mu)/scale);q=pred*scale+mu;out,_=trace.__wrapped__(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;lp=c['teacher'].to(device).double().log_softmax(-1);return (lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean()
 for initial in protocol['initial_models']:
  assert sha(Path(initial['path']))==initial['sha256'];state=torch.load(initial['path'],weights_only=True);kind=initial['kind'];seed=initial['seed'];mu=state['mean'].to(device);scale=state['scale'].to(device);schedule=protocol['schedules'][str(seed)]
  for arm in protocol['arms']:
   f=Featurizer(kind,seed).to(device);f.load_state_dict(state['state']);opt=torch.optim.Adam(f.parameters(),lr=protocol['learning_rate']);history=[]
   for step in range(protocol['steps']):
    opt.zero_grad()
    if arm=='natural_only_KL' or step%2==0:
     key=f"natural:{schedule['natural_documents'][step]}";loss=kl_forward(f,mu,scale,key);loss.backward();used=[key]
    else:
     pairid=schedule['mixed_task_pairs'][step//2];assert pairid in trainids;used=[f'task:{pairid}:{ep}' for ep in [0,1]];values=[]
     for key in used:
      loss=kl_forward(f,mu,scale,key)/2;loss.backward();values.append(float(loss.detach()))
     loss=torch.tensor(sum(values))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in f.parameters());opt.step()
    with torch.no_grad():f.dictionary.copy_(F.normalize(f.dictionary,dim=0))
    history.append(dict(step=step,keys=used,loss=float(loss.detach())))
    if step%30==29:print('UPDATES',kind,seed,arm,step+1,flush=True)
   assert all(p.grad is None for p in model.parameters());path=dest/f'{kind}_{seed}_{arm}.pt';torch.save(dict(state={k:v.cpu() for k,v in f.state_dict().items()},mean=mu.cpu(),scale=scale.cpu(),kind=kind,seed=seed,arm=arm),path);models.append(dict(kind=kind,seed=seed,arm=arm,path=str(path),sha256=sha(path)));histories.append(dict(kind=kind,seed=seed,arm=arm,updates=history));sampledfamilies=set()
   with torch.no_grad():
    for key,c in cache.items():
     if c['domain']=='natural' and c['split']=='train':continue
     ids=c['ids'].to(device);native=c['teacher'].to(device);v=c['write'].to(device);pred,support,_=f((v-mu)/scale);q=pred*scale+mu;assert torch.all(support.sum(-1)==32);out,_=trace(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;row=dict(key=key,kind=kind,seed=seed,arm=arm,domain=c['domain'],split=c['split'],family=c['family'],**metrics(native,out,ids),local_relative_error=float((q.double()-v.double()).square().sum()/v.double().square().sum()))
     if c['domain']=='task':row.update(id=c['id'],endpoint=c['endpoint'],task=competence(out[0,-1],c['groups'],c['variant']),native_task=competence(native[0,-1],c['groups'],c['variant']))
     rows.append(row);take=(key=='natural:32') or (c['domain']=='task' and c['split']=='evaluation' and c['family'] not in sampledfamilies)
     if take:
      samples.append(dict(row=len(rows)-1,native=native.cpu(),output=out.cpu(),ids=ids.cpu(),write=v.cpu(),reconstruction=q.cpu(),support=support.cpu(),groups=c.get('groups'),variant=c.get('variant')))
      if c['domain']=='task':sampledfamilies.add(c['family'])
   print('EVALUATED',kind,seed,arm,flush=True);del f,opt
 summaries=[]
 for modelentry in models:
  for domain,part in [('natural','evaluation'),('task','train'),('task','evaluation')]:
   base=[x for x in rows if all(x[k]==modelentry[k] for k in ['kind','seed','arm']) and x['domain']==domain and x['split']==part]
   for family in [None]+(sorted({x['family'] for x in base}) if domain=='task' else []):
    rr=base if family is None else [x for x in base if x['family']==family];summary={k:modelentry[k] for k in ['kind','seed','arm']}|dict(domain=domain,split=part,family=family,endpoints=len(rr),**{k:sum(x[k] for x in rr)/len(rr) for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_error']})
    if domain=='task':summary.update(binary_rate=sum(x['task']['binary'] for x in rr)/len(rr),task_top1_rate=sum(x['task']['top1'] for x in rr)/len(rr),absolute_margin_error=sum(abs(x['task']['margin']-x['native_task']['margin']) for x in rr)/len(rr))
    summaries.append(summary)
 # Baseline summaries are immutable existing M8natural/M9task scores, reaggregated by the new split.
 m8=json.loads((P/'validation/causal_featurizers.json').read_text());m9=json.loads((P/'validation/featurizer_task_transfer.json').read_text());baseline=[]
 for initial in protocol['initial_models']:
  kind,seed=initial['kind'],initial['seed'];nat=next(x for x in m8['causal_summaries'] if x['kind']==kind and x['seed']==seed and x['objective']=='downstream_KL');baseline.append(dict(kind=kind,seed=seed,domain='natural',split='evaluation',mean_KL=nat['KL'],delta_CE=nat['delta_CE']))
  for part,pids in [('train',trainids),('evaluation',evalids)]:
   rr=[x for x in m9['rows'] if x['kind']==kind and x['seed']==seed and x['objective']=='downstream_KL' and x['id'] in pids];baseline.append(dict(kind=kind,seed=seed,domain='task',split=part,mean_KL=sum(x['mean_KL'] for x in rr)/len(rr),binary_rate=sum(x['task']['binary'] for x in rr)/len(rr),task_top1_rate=sum(x['task']['top1'] for x in rr)/len(rr)))
 ap=A/'emlp_m11_audit_samples.pt';torch.save(samples,ap);out=dict(status='completed',protocol_sha256=sha(pp),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,models=models,histories=histories,rows=rows,summaries=summaries,baseline=baseline,samples_path=str(ap),samples_sha256=sha(ap));(P/'validation/diverse_supervision_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([x for x in summaries if x['family'] is None],indent=2),flush=True)
if __name__=='__main__':main()
