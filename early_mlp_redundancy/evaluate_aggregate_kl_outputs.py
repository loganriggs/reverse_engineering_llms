"""M27 complete unchanged output inventory, every candidate retained."""
from pathlib import Path
import hashlib,json
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=P/'aggregate_kl_protocol.json';protocol=json.loads(pp.read_text());rp=P/'aggregate_kl_models.json';r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(pp);assert json.loads((P/'aggregate_kl_models_audit.json').read_text())['result_sha256']==sha(rp);op=P/'aggregate_kl_outputs.json';assert not op.exists();model,receipt=load_model();device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;models=r['models'];fs={};states={}
 for e in models:
  assert sha(Path(e['path']))==e['sha256'];s=torch.load(e['path'],weights_only=True);f=Featurizer(e['kind'],e['seed']).to(device);f.load_state_dict(s['state']);key=e['kind'],e['seed'],e['arm'];fs[key]=f;states[key]=(s['mean'].to(device),s['scale'].to(device))
 split=json.loads((P/'diverse_supervision_split.json').read_text());trainids=set(split['train_pair_ids']);cohort=json.loads((P/'natural_token_context_cohort.json').read_text());panel=json.loads((P/'training_source_context_audit.json').read_text());expanded=json.loads((P/'context_expansion_protocol.json').read_text());screen=json.loads((P/'context_expansion_competence.json').read_text());passing={(x['template'],x['lexical_split'],x['family']):x['pass_gate'] for x in screen['cohorts']};inputs=[]
 for d in range(32,64):inputs.append(dict(key=f'natural:{d}',token_ids=cohort['token_ids'][d],domain='natural',split='evaluation',family=None,template_split=None,pass_gate=None))
 for pair in panel['rows']:
  for ep in [0,1]:inputs.append(dict(key=f"task:{pair['id']}:{ep}",token_ids=pair['token_ids'][ep],domain='original',template_split=None,pass_gate=None,id=pair['id'],endpoint=ep,split='train' if pair['id'] in trainids else 'evaluation',family=pair['family'],groups=pair['answer_groups'][ep],variant=pair['variants'][ep]))
 for pair in expanded['rows']:
  for ep in [0,1]:inputs.append(dict(key=f"expanded:{pair['id']}:{ep}",token_ids=pair['token_ids'][ep],domain='expanded',id=pair['id'],endpoint=ep,split=pair['lexical_split'],template_split=pair['template_split'],pass_gate=passing[pair['template'],pair['lexical_split'],pair['family']],family=pair['family'],groups=pair['answer_groups'][ep],variant=pair['variants'][ep]))
 assert len(inputs)==1152;rows=[];samples=[];calls=0;sampled=set();completed=0;progress_path=A/'emlp_m27_output_progress.pt'
 if progress_path.exists():
  progress=torch.load(progress_path,weights_only=True);assert progress['protocol_sha256']==sha(pp) and progress['models_sha256']==sha(rp);rows=progress['rows'];samples=progress['samples'];calls=progress['calls'];sampled=set(progress['sampled']);completed=progress['completed'];assert calls==completed*13;print('RESUME',completed,flush=True)
 for ix,c in enumerate(inputs):
  if ix<completed:continue
  ids=torch.tensor([c['token_ids']],device=device);native,cap=trace(model,ids);calls+=1;v=mlp.Down(cap[1]['products']);stratum=c['domain'],c['split'],c['template_split'];take=stratum not in sampled
  for e in models:
   key=e['kind'],e['seed'],e['arm'];f=fs[key];mu,scale=states[key];pred,support,_=f((v-mu)/scale);q=pred*scale+mu;out,_=trace(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;row=dict(key=c['key'],kind=e['kind'],seed=e['seed'],arm=e['arm'],domain=c['domain'],split=c['split'],template_split=c['template_split'],pass_gate=c['pass_gate'],family=c['family'],**metrics(native,out,ids),local_relative_error=float((q.double()-v.double()).square().sum()/v.double().square().sum()))
   if c['domain']!='natural':row.update(id=c['id'],endpoint=c['endpoint'],task=competence(out[0,-1],c['groups'],c['variant']),native_task=competence(native[0,-1],c['groups'],c['variant']))
   rows.append(row)
   if take:samples.append(dict(row=len(rows)-1,native=native.cpu(),output=out.cpu(),ids=ids.cpu(),write=v.cpu(),reconstruction=q.cpu(),support=support.cpu(),groups=c.get('groups'),variant=c.get('variant')))
  sampled.add(stratum)
  if ix%64==63:print('INPUTS',ix+1,len(inputs),'FORWARDS',calls,flush=True)
  if ix%128==127:
   tmp=progress_path.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),models_sha256=sha(rp),rows=rows,samples=samples,calls=calls,sampled=list(sampled),completed=ix+1),tmp);tmp.replace(progress_path)
 summaries=[]
 for modelentry in models:
  rr=[x for x in rows if all(x[k]==modelentry[k] for k in ['kind','seed','arm'])]
  strata=sorted({(x['domain'],x['split'],x['template_split'] or '') for x in rr})
  for domain,part,template in strata:
   base=[x for x in rr if x['domain']==domain and x['split']==part and (x['template_split'] or '')==template]
   for gate in ['all','native_passing'] if domain=='expanded' else ['all']:
    selected=[x for x in base if gate=='all' or x['pass_gate']]
    for family in [None]+(sorted({x['family'] for x in selected}) if domain!='natural' else []):
     xs=selected if family is None else [x for x in selected if x['family']==family];summary={k:modelentry[k] for k in ['kind','seed','arm']}|dict(domain=domain,split=part,template_split=template or None,gate=gate,family=family,endpoints=len(xs),**{k:sum(x[k] for x in xs)/len(xs) for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_error']})
     if domain!='natural':summary.update(binary_rate=sum(x['task']['binary'] for x in xs)/len(xs),task_top1_rate=sum(x['task']['top1'] for x in xs)/len(xs),absolute_margin_error=sum(abs(x['task']['margin']-x['native_task']['margin']) for x in xs)/len(xs))
     summaries.append(summary)
 sp=A/'emlp_m27_aggregate_kl_output_samples.pt';torch.save(samples,sp);out=dict(protocol_sha256=sha(pp),models_sha256=sha(rp),models=models,checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp));op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,len(rows),flush=True)
if __name__=='__main__':main()
