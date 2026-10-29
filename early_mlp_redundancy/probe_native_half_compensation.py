"""M41 fixed native-half compensation: distributed averaging versus complementarity."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'native_half_compensation_protocol.json';protocol=json.loads(pp.read_text());assert protocol['source_sha256']==sha(Path(__file__))
 dp=V/'mlp1_group_deletion.json';assert sha(dp)==protocol['masks_sha256'];d=json.loads(dp.read_text());panel=json.loads((V/'training_source_context_audit.json').read_text());cohort=json.loads((V/'natural_token_context_cohort.json').read_text());split=json.loads((V/'diverse_supervision_split.json').read_text());train=set(split['train_pair_ids']);inputs=[]
 for pair in panel['rows']:
  for ep in [0,1]:inputs.append(dict(key=f"task:{pair['id']}:{ep}",ids=pair['token_ids'][ep],domain='task',split='train' if pair['id'] in train else 'evaluation',family=pair['family'],groups=pair['answer_groups'][ep],variant=pair['variants'][ep]))
 for doc in range(32,64):inputs.append(dict(key=f'natural:{doc}',ids=cohort['token_ids'][doc],domain='natural',split='evaluation',family=None))
 assert len(inputs)==192
 model,receipt=load_model();device=next(model.parameters()).device;rows=[];samples=[];calls=0;completed=0;progress=A/'emlp_m41_progress.pt';op=V/'native_half_compensation.json';assert not op.exists()
 if progress.exists():
  old=torch.load(progress,weights_only=True);assert old['protocol_sha256']==sha(pp);rows,samples,calls,completed=[old[k] for k in ['rows','samples','calls','completed']]
 for ix,c in enumerate(inputs):
  if ix<completed:continue
  ids=torch.tensor([c['ids']],device=device);base,cap=trace(model,ids);calls+=1
  for layer in [0,1,2]:
   mlp=model.transformer.h[layer].mlp;prod=cap[layer]['products'];net=mlp.Down(prod);ne=float(net.double().square().sum());ce=float((prod.double().square()*mlp.Down.weight.double().square().sum(0)).sum());expected=(4608*ce-ne)/4607
   for half in ['half_01','half_23']:
    keep=torch.ones(4608,device=device);keep[d['removed_sets'][half]]=0;retained=mlp.Down(prod*keep)
    for factor in [1.,2.]:
     write=factor*retained+mlp.Down_bias
     lg,_=trace(model,ids,replace=lambda j,z,w:write if j==layer else w);calls+=1
     row=dict(key=c['key'],domain=c['domain'],split=c['split'],family=c['family'],layer=layer,removed=half,factor=factor,**metrics(base,lg,ids),net_energy=ne,channel_energy=ce,expected_uniform_half_compensation_error=expected,local_error_energy=float((factor*retained.double()-net.double()).square().sum()),local_relative_error=float((factor*retained.double()-net.double()).square().sum())/ne)
     if c['domain']=='task':row.update(task=competence(lg[0,-1],c['groups'],c['variant']),native_task=competence(base[0,-1],c['groups'],c['variant']))
     rows.append(row)
     if ix in [0,160] and half=='half_01':samples.append(dict(row=len(rows)-1,native=base.cpu(),output=lg.cpu(),ids=ids.cpu(),products=prod.cpu(),D=mlp.Down.weight.cpu(),net=net.cpu(),retained=retained.cpu(),groups=c.get('groups'),variant=c.get('variant')))
  if (ix+1)%16==0 or ix==191:
   temp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),rows=rows,samples=samples,calls=calls,completed=ix+1),temp);temp.replace(progress);print('INPUTS',ix+1,192,'FORWARDS',calls,flush=True)
 summaries=[]
 for domain,part in [('task','train'),('task','evaluation'),('natural','evaluation')]:
  for layer in [0,1,2]:
   for factor in [1.,2.]:
    rr=[x for x in rows if (x['domain'],x['split'],x['layer'],x['factor'])==(domain,part,layer,factor)]
    summaries.append(dict(domain=domain,split=part,layer=layer,factor=factor,n=len(rr),**{k:sum(x[k] for x in rr)/len(rr) for k in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']},relative_local_error=sum(x['local_error_energy'] for x in rr)/sum(x['net_energy'] for x in rr),expected_uniform_half_relative_error=sum(x['expected_uniform_half_compensation_error'] for x in rr)/sum(x['net_energy'] for x in rr)))
 assert calls==2496 and len(rows)==2304
 sp=A/'emlp_m41_samples.pt';torch.save(samples,sp);op.write_text(json.dumps(dict(protocol_sha256=sha(pp),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp)),indent=2)+'\n');print('DONE',calls,len(rows),flush=True)
if __name__=='__main__':main()
