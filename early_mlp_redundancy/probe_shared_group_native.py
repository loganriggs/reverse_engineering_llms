"""M49 frozen exclusive/shared/union external interventions on native MLP1 writes."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'shared_group_native_protocol.json';q=json.loads(pp.read_text());mp=V/'coactivation_groups.json';assert sha(mp)==q['models_sha256'];assert sha(V/'shared_group_decomposition.json')==q['decomposition_sha256'];entries=[x for x in json.loads(mp.read_text())['models'] if x['kind']=='shared_fixed'];assign={x['key']:x['group'] for x in q['assignments']};panel=json.loads((V/'training_source_context_audit.json').read_text());cohort=json.loads((V/'natural_token_context_cohort.json').read_text());train=set(json.loads((V/'diverse_supervision_split.json').read_text())['train_pair_ids']);inputs=[]
 for pair in panel['rows']:
  for ep in [0,1]:inputs.append(dict(key=f"task:{pair['id']}:{ep}",ids=pair['token_ids'][ep],domain='task',split='train' if pair['id'] in train else 'evaluation',family=pair['family'],groups=pair['answer_groups'][ep],variant=pair['variants'][ep]))
 for doc in range(32,64):inputs.append(dict(key=f'natural:{doc}',ids=cohort['token_ids'][doc],domain='natural',split='evaluation',family=None))
 assert {c['key'] for c in inputs}==set(assign);model,receipt=load_model();dev=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;models={}
 for e in entries:
  path=Path(e['path']);assert sha(path)==e['sha256'];s=torch.load(path,weights_only=True);f=Featurizer('shared_fixed',e['seed']).to(dev);f.load_state_dict(s['state']);models[e['seed'],e['arm']]=(f,s['mean'].to(dev),s['scale'].to(dev))
 rows=[];samples=[];calls=0;completed=0;progress=A/'emlp_m49_progress.pt';op=V/'shared_group_native.json';assert not op.exists()
 if progress.exists():
  old=torch.load(progress,weights_only=True);assert old['protocol_sha256']==sha(pp);rows,samples,calls,completed=[old[k] for k in ['rows','samples','calls','completed']]
 for k,c in enumerate(inputs):
  if k<completed:continue
  ids=torch.tensor([c['ids']],device=dev);base,cap=trace(model,ids);calls+=1;write=mlp.Down(cap[1]['products']);group=assign[c['key']]
  for e in entries:
   f,mu,scale=models[e['seed'],e['arm']];a=torch.nn.functional.linear((write-mu)/scale,f.encoder,f.encoder_bias);scores=a.square()@f.membership.T;gates=torch.zeros_like(scores).scatter_(-1,scores.topk(4,dim=-1).indices,1);counts=gates@f.membership;selected=gates[:,:,group,None].bool()&f.membership[group].bool();exclusive=selected&(counts==1);shared=selected&(counts>1);D=torch.nn.functional.normalize(f.dictionary,dim=0)
   for component,mask in [('exclusive',exclusive),('shared',shared),('union',selected)]:
    delta=(a*mask)@D.T*scale;noop=not bool(torch.count_nonzero(delta))
    if noop:out=base
    else:out,_=trace(model,ids,replace=lambda j,z,w:w-delta if j==1 else w);calls+=1
    row=dict(key=c['key'],domain=c['domain'],split=c['split'],family=c['family'],seed=e['seed'],source_arm=e['source_arm'],grouping=e['grouping'],arm=e['arm'],group=group,component=component,noop=noop,positions=ids.numel(),gate_active_positions=int(gates[:,:,group].sum()),atom_positions=int(mask.sum()),delta_energy=float(delta.double().square().sum()),native_write_energy=float(write.double().square().sum()),**metrics(base,out,ids))
    if c['domain']=='task':row.update(native_task=competence(base[0,-1],c['groups'],c['variant']),task=competence(out[0,-1],c['groups'],c['variant']))
    rows.append(row)
    if k in [0,160]:samples.append(dict(row=len(rows)-1,native=base.cpu(),output=out.cpu(),ids=ids.cpu(),a=a.cpu(),gates=gates.cpu(),membership=f.membership.cpu(),D=D.cpu(),scale=scale.cpu(),delta=delta.cpu(),mask=mask.cpu(),groups=c.get('groups'),variant=c.get('variant')))
  if (k+1)%16==0 or k==191:
   tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),rows=rows,samples=samples,calls=calls,completed=k+1),tmp);tmp.replace(progress);print('INPUTS',k+1,'FORWARDS',calls,flush=True)
 assert len(rows)==4608 and calls==192+sum(not x['noop'] for x in rows)
 summaries=[]
 for e in entries:
  for dom,part in [('task','train'),('task','evaluation'),('natural','evaluation')]:
   for component in ['exclusive','shared','union']:
    rr=[x for x in rows if (x['seed'],x['arm'],x['domain'],x['split'],x['component'])==(e['seed'],e['arm'],dom,part,component)];summaries.append({k:e[k] for k in ['seed','source_arm','grouping','arm']}|dict(domain=dom,split=part,component=component,n=len(rr),noops=sum(x['noop'] for x in rr),**{f:sum(x[f] for x in rr)/len(rr) for f in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']},delta_energy=sum(x['delta_energy'] for x in rr),native_write_energy=sum(x['native_write_energy'] for x in rr),atom_positions=sum(x['atom_positions'] for x in rr),gate_active_positions=sum(x['gate_active_positions'] for x in rr)))
 sp=A/'emlp_m49_samples.pt';torch.save(samples,sp);op.write_text(json.dumps(dict(protocol_sha256=sha(pp),models_sha256=sha(mp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp)),indent=2)+'\n');print('DONE',calls,len(rows),flush=True)
if __name__=='__main__':main()
