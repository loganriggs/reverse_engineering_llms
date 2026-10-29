"""M47 native outputs for paired versus fixed same-token context permutation."""
from pathlib import Path
import json,hashlib,collections
import torch
from inspect_model import load_model,trace
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'native_half_pairing_native_protocol.json';q=json.loads(pp.read_text());rp=V/'native_half_pairing.json';assert sha(rp)==q['pairing_result_sha256'];r=json.loads(rp.read_text());assert json.loads((V/'native_half_pairing_audit.json').read_text())['result_sha256']==sha(rp);cache=Path(r['cache_path']);assert sha(cache)==r['cache_sha256'];d=torch.load(cache,weights_only=True);meta=d['metadata'];mp=V/'native_half_token_controls.json';assert sha(mp)==q['models_sha256'];manifest=json.loads(mp.read_text());states={}
 for e in manifest['models']:
  path=Path(e['path']);assert sha(path)==e['sha256'];states[e['layer'],e['removed']]=torch.load(path,weights_only=True)
 known=set(int(t) for t in next(iter(states.values()))['token_ids']);groups=collections.defaultdict(list);index={}
 for i,x in enumerate(meta):groups[x['domain'],x['split'],x['token']].append(i);index[x['domain'],x['split'],tuple(x['prefix'])]=i
 donors=torch.arange(len(meta));eligible=torch.zeros(len(meta),dtype=torch.bool)
 for (_,_,tok),ii in groups.items():
  if tok not in known or len(ii)<2:continue
  ii=sorted(ii,key=lambda i:hashlib.sha256(json.dumps(meta[i]['prefix'],separators=(',',':')).encode()).hexdigest())
  for j,i in enumerate(ii):donors[i]=ii[(j+1)%len(ii)];eligible[i]=True
 assert torch.equal(torch.sort(donors).values,torch.arange(len(meta))) and bool((donors[eligible]!=torch.arange(len(meta))[eligible]).all())
 predictions={}
 for e in manifest['models']:
  st=states[e['layer'],e['removed']];x=d['halves'][f"{e['layer']}:{e['retained']}"].double();tok=torch.tensor([z['token'] for z in meta]);ix=torch.searchsorted(st['token_ids'],tok).clamp(max=len(st['token_ids'])-1);tx=st['token_input_mean'][ix];ty=st['token_target_mean'][ix];pred=(x-tx)@st['U']@st['B']+ty;pred[~eligible]=0;predictions[e['layer'],e['removed']]=pred.float()
 panel=json.loads((V/'training_source_context_audit.json').read_text());cohort=json.loads((V/'natural_token_context_cohort.json').read_text());train=set(json.loads((V/'diverse_supervision_split.json').read_text())['train_pair_ids']);masks=json.loads((V/'mlp1_group_deletion.json').read_text())['removed_sets'];inputs=[]
 for pair in panel['rows']:
  for ep in [0,1]:inputs.append(dict(key=f"task:{pair['id']}:{ep}",ids=pair['token_ids'][ep],domain='task',split='train' if pair['id'] in train else 'evaluation',family=pair['family'],groups=pair['answer_groups'][ep],variant=pair['variants'][ep]))
 for doc in range(32,64):inputs.append(dict(key=f'natural:{doc}',ids=cohort['token_ids'][doc],domain='natural',split='evaluation',family=None))
 model,receipt=load_model();assert receipt.weights_sha256==r['checkpoint_sha256'];device=next(model.parameters()).device;rows=[];samples=[];calls=0;completed=0;progress=A/'emlp_m47_progress.pt';op=V/'native_half_pairing_outputs.json';assert not op.exists()
 if progress.exists():
  saved=torch.load(progress,weights_only=True);assert saved['protocol_sha256']==sha(pp);rows,samples,calls,completed=[saved[k] for k in ['rows','samples','calls','completed']]
 for k,c in enumerate(inputs):
  if k<completed:continue
  ids=torch.tensor([c['ids']],device=device);base,cap=trace(model,ids);calls+=1;ii=torch.tensor([index[c['domain'],c['split'],tuple(c['ids'][:j+1])] for j in range(len(c['ids']))]);allowed=eligible[ii].to(device)[None];n=int(allowed.sum())
  for layer in [0,1,2]:
   mlp=model.transformer.h[layer].mlp;prod=cap[layer]['products'];nativewrite=cap[layer]['native_write'];net=mlp.Down(prod)
   for half in ['half_01','half_23']:
    keep=torch.ones(4608,device=device);keep[masks[half]]=0;ret=mlp.Down(prod*keep)
    for arm in ['paired','permuted']:
     selected=ii if arm=='paired' else donors[ii];pred=predictions[layer,half][selected].to(device)[None];write=torch.where(allowed[...,None],ret+pred+mlp.Down_bias,nativewrite)
     if n:out,_=trace(model,ids,replace=lambda j,z,w:write if j==layer else w);calls+=1
     else:out=base
     row=dict(key=c['key'],domain=c['domain'],split=c['split'],family=c['family'],layer=layer,removed=half,arm=arm,eligible_positions=n,positions=ids.numel(),**metrics(base,out,ids),local_error_energy=float((write.double()-nativewrite.double()).square().sum()),eligible_native_energy=float(net.double().square().sum(-1)[allowed].sum()))
     if c['domain']=='task':row.update(task=competence(out[0,-1],c['groups'],c['variant']),native_task=competence(base[0,-1],c['groups'],c['variant']))
     rows.append(row)
     if k in [0,160] and half=='half_01':samples.append(dict(row=len(rows)-1,native=base.cpu(),output=out.cpu(),ids=ids.cpu(),retained=ret.cpu(),native_write=nativewrite.cpu(),net=net.cpu(),write=write.cpu(),bias=mlp.Down_bias.cpu(),indices=ii,selected=selected,eligible=allowed.cpu(),groups=c.get('groups'),variant=c.get('variant')))
  if (k+1)%16==0 or k==191:
   temp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),rows=rows,samples=samples,calls=calls,completed=k+1),temp);temp.replace(progress);print('INPUTS',k+1,'FORWARDS',calls,flush=True)
 assert len(rows)==2304 and calls==192+sum(x['eligible_positions']>0 for x in rows)
 summaries=[]
 for dom,part in [('task','train'),('task','evaluation'),('natural','evaluation')]:
  for layer in [0,1,2]:
   for arm in ['paired','permuted']:
    rr=[x for x in rows if (x['domain'],x['split'],x['layer'],x['arm'])==(dom,part,layer,arm)];summaries.append(dict(domain=dom,split=part,layer=layer,arm=arm,n=len(rr),**{f:sum(x[f] for x in rr)/len(rr) for f in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']},local_error_energy=sum(x['local_error_energy'] for x in rr),eligible_native_energy=sum(x['eligible_native_energy'] for x in rr)))
 sp=A/'emlp_m47_samples.pt';torch.save(dict(samples=samples,donors=donors,eligible=eligible),sp);op.write_text(json.dumps(dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),pairing_result_sha256=sha(rp),models_sha256=sha(mp),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp)),indent=2)+'\n');print('DONE',calls,len(rows),flush=True)
if __name__=='__main__':main()
