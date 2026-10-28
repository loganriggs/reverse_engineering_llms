"""M18 execute the frozen paired context-exposure protocol."""
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
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');pp=P/'validation/context_training_protocol.json';protocol=json.loads(pp.read_text());assert json.loads((P/'validation/context_training_protocol_audit.json').read_text())['protocol_sha256']==sha(pp);sp=P/'validation/diverse_supervision_split.json';split=json.loads(sp.read_text());assert sha(sp)==protocol['split_sha256'];panel=json.loads((P/'validation/training_source_context_audit.json').read_text());expanded=json.loads((P/'validation/context_expansion_protocol.json').read_text());screen=json.loads((P/'validation/context_expansion_competence.json').read_text());assert sha(P/'validation/context_expansion_protocol.json')==protocol['expansion_sha256'];assert sha(P/'validation/context_expansion_competence.json')==protocol['competence_sha256'];passing={(c['template'],c['lexical_split'],c['family']):c['pass_gate'] for c in screen['cohorts']};cohort=json.loads((P/'validation/natural_token_context_cohort.json').read_text());trainids=set(split['train_pair_ids']);dest=A/'emlp_m18_context_training';dest.mkdir(exist_ok=True);assert not (P/'validation/context_training_results.json').exists();assert not list(dest.glob('*.pt')), 'Existing checkpoints: inspect run before resuming'
 (P/'validation/context_training_run_receipt.json').write_text(json.dumps(dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),status='started_fixed_protocol'),indent=2)+'\n')
 model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;cache={};calls=0
 def collect(key,tokens,meta):
  nonlocal calls
  ids=torch.tensor([tokens],device=device);out,cap=trace(model,ids);calls+=1
  with torch.no_grad():v=mlp.Down(cap[1]['products']).cpu()
  cache[key]=dict(ids=ids.cpu(),teacher=out.cpu(),write=v,**meta)
 for d,tokens in enumerate(cohort['token_ids']):collect(f'natural:{d}',tokens,dict(domain='natural',document=d,split='train' if d<32 else 'evaluation',family=None,template_split=None,pass_gate=None))
 for pair in panel['rows']:
  for ep in [0,1]:collect(f"task:{pair['id']}:{ep}",pair['token_ids'][ep],dict(domain='original',template_split=None,pass_gate=None,id=pair['id'],endpoint=ep,split='train' if pair['id'] in trainids else 'evaluation',family=pair['family'],groups=pair['answer_groups'][ep],variant=int(pair['variants'][ep])))
 for pair in expanded['rows']:
  for ep in [0,1]:collect(f"expanded:{pair['id']}:{ep}",pair['token_ids'][ep],dict(domain='expanded',id=pair['id'],endpoint=ep,split=pair['lexical_split'],template_split=pair['template_split'],pass_gate=passing[pair['template'],pair['lexical_split'],pair['family']],family=pair['family'],groups=pair['answer_groups'][ep],variant=int(pair['variants'][ep])))
 print('TEACHERS',len(cache),flush=True);models=[];histories=[];rows=[];samples=[]
 def kl_forward(f,mu,scale,key):
  nonlocal calls
  c=cache[key];ids=c['ids'].to(device);v=c['write'].to(device);pred,_,_=f((v-mu)/scale);q=pred*scale+mu;out,_=trace.__wrapped__(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;lp=c['teacher'].to(device).double().log_softmax(-1);return (lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean()
 for initial in protocol['initial_models']:
  assert sha(Path(initial['path']))==initial['sha256'];state=torch.load(initial['path'],weights_only=True);kind=initial['kind'];seed=initial['seed'];mu=state['mean'].to(device);scale=state['scale'].to(device);schedule=protocol['schedules'][str(seed)]
  for arm in ['initial']+protocol['arms']:
   f=Featurizer(kind,seed).to(device);f.load_state_dict(state['state']);opt=torch.optim.Adam(f.parameters(),lr=protocol['learning_rate']);history=[]
   for step in range(0 if arm=='initial' else protocol['steps']):
    opt.zero_grad()
    if step%2==0:
     key=f"natural:{schedule['natural_documents'][step]}";loss=kl_forward(f,mu,scale,key);loss.backward();used=[key]
    else:
     u=schedule['task_updates'][step//2];assert u['source_pair_id'] in trainids;prefix='task' if arm=='original_context' else 'expanded';pairid=u['source_pair_id'] if arm=='original_context' else u['expanded_pair_id'];used=[f'{prefix}:{pairid}:{ep}' for ep in [0,1]];values=[]
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
     ids=c['ids'].to(device);native=c['teacher'].to(device);v=c['write'].to(device);pred,support,_=f((v-mu)/scale);q=pred*scale+mu;assert torch.all(support.sum(-1)==32);out,_=trace(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;row=dict(key=key,kind=kind,seed=seed,arm=arm,domain=c['domain'],split=c['split'],template_split=c['template_split'],pass_gate=c['pass_gate'],family=c['family'],**metrics(native,out,ids),local_relative_error=float((q.double()-v.double()).square().sum()/v.double().square().sum()))
     if c['domain']!='natural':row.update(id=c['id'],endpoint=c['endpoint'],task=competence(out[0,-1],c['groups'],c['variant']),native_task=competence(native[0,-1],c['groups'],c['variant']))
     rows.append(row);take=(key=='natural:32') or (c['domain']!='natural' and c['split']=='evaluation' and (c['domain'],c['template_split'],c['family']) not in sampledfamilies)
     if take:
      samples.append(dict(row=len(rows)-1,native=native.cpu(),output=out.cpu(),ids=ids.cpu(),write=v.cpu(),reconstruction=q.cpu(),support=support.cpu(),groups=c.get('groups'),variant=c.get('variant')))
      if c['domain']!='natural':sampledfamilies.add((c['domain'],c['template_split'],c['family']))
   print('EVALUATED',kind,seed,arm,flush=True);del f,opt
 summaries=[]
 for modelentry in models:
  rr=[x for x in rows if all(x[k]==modelentry[k] for k in ['kind','seed','arm'])]
  strata=sorted({(x['domain'],x['split'],x['template_split'] or '') for x in rr})
  for domain,part,template in strata:
   base=[x for x in rr if x['domain']==domain and x['split']==part and (x['template_split'] or '')==template]
   for gate in ['all','native_passing'] if domain=='expanded' else ['all']:
    selected=[x for x in base if gate=='all' or x['pass_gate']]
    for family in [None]+(sorted({x['family'] for x in selected}) if domain!='natural' else []):
     xs=selected if family is None else [x for x in selected if x['family']==family]
     summary={k:modelentry[k] for k in ['kind','seed','arm']}|dict(domain=domain,split=part,template_split=template or None,gate=gate,family=family,endpoints=len(xs),**{k:sum(x[k] for x in xs)/len(xs) for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_error']})
     if domain!='natural':summary.update(binary_rate=sum(x['task']['binary'] for x in xs)/len(xs),task_top1_rate=sum(x['task']['top1'] for x in xs)/len(xs),absolute_margin_error=sum(abs(x['task']['margin']-x['native_task']['margin']) for x in xs)/len(xs))
     summaries.append(summary)
 ap=A/'emlp_m18_context_training_samples.pt';torch.save(samples,ap);out=dict(status='completed',protocol_sha256=sha(pp),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,models=models,histories=histories,rows=rows,summaries=summaries,samples_path=str(ap),samples_sha256=sha(ap));(P/'validation/context_training_results.json').write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,'MODELS',len(models),'ROWS',len(rows),flush=True)
if __name__=='__main__':main()
