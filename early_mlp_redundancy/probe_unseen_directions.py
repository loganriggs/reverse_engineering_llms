"""M28 same-input unseen-group causal response; all candidates and noops retained."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from collect_response_targets import response_error
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=P/'unseen_direction_protocol.json';p=json.loads(pp.read_text());tp=P/'response_training_targets.json';t=json.loads(tp.read_text());assert p['targets_sha256']==sha(tp);cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);op=P/'unseen_direction_results.json';assert not op.exists();expanded={x['id']:x for x in json.loads((P/'context_expansion_protocol.json').read_text())['rows']};model,receipt=load_model();device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;records=[];samples=[];calls=0;completed=[];progress=A/'emlp_m28_unseen_progress.pt'
 if progress.exists():
  old=torch.load(progress,weights_only=True);assert old['protocol_sha256']==sha(pp);records=old['records'];samples=old['samples'];calls=old['calls'];completed=old['completed'];print('RESUME',completed,flush=True)
 for kind,seed in [('scalar',71),('block',71),('scalar',72),('block',72)]:
  if [kind,seed] in completed:continue
  es=[e for e in p['models'] if (e['kind'],e['seed'])==(kind,seed)];assert len(es)==5;fs={};states={};memo={};sampled=set()
  for e in es:
   assert sha(Path(e['path']))==e['sha256'];state=torch.load(e['path'],weights_only=True);f=Featurizer(kind,seed).to(device);f.load_state_dict(state['state']);fs[e['arm']]=f;states[e['arm']]=(state['mean'].to(device),state['scale'].to(device))
  selected=[x for x in p['rows'] if (x['kind'],x['seed'])==(kind,seed)]
  for k,spec in enumerate(selected):
   i=spec['target_index'];row=t['records'][i];c=cache['inputs'][row['input_key']];initial=cache['initials'][row['initial_key']];ids=c['ids'].to(device);teacher=c['native'][0,-1].to(device);lp=teacher.double().log_softmax(-1);pair=expanded[row['pair_id']];groups=pair['answer_groups'][row['endpoint']];variant=pair['variants'][row['endpoint']];mask=cache['targets'][i]['mask'].to(device);atoms=torch.arange(4*spec['group'],4*spec['group']+4,device=device);delta=(initial['code'].to(device)[:,:,atoms]@initial['D'].to(device)[:,atoms].T)*initial['scale'].to(device)*mask;noop=bool((delta==0).all());assert spec['group'] not in spec['supervised_groups']
   if noop:nchanged=teacher
   else:
    out,_=trace(model,ids,replace=lambda j,z,w:w-delta if j==1 else w);calls+=1;nchanged=out[0,-1].clone()
   nr=nchanged.double().log_softmax(-1)-lp;energy=float((lp.exp()*nr.square()).sum());nm=competence(teacher,groups,variant)['margin']-competence(nchanged,groups,variant)['margin']
   for arm,f in fs.items():
    key=row['input_key'],arm
    if key not in memo:
     mu,scale=states[arm];pred,_,_=f((c['write'].to(device)-mu)/scale);q=pred*scale+mu;out,_=trace(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;tlp=c['native'].to(device).double().log_softmax(-1);K=float((tlp.exp()*(tlp-out.double().log_softmax(-1))).sum(-1).mean());memo[key]=(q,out[0,-1].clone(),K)
     if arm=='initial':assert torch.equal(q.cpu(),initial['q']) and torch.equal(out.cpu(),initial['output'])
    q,intact,K=memo[key]
    if noop:changed=intact
    else:
     out,_=trace(model,ids,replace=lambda j,z,w:q-delta+mlp.Down_bias if j==1 else w);calls+=1;changed=out[0,-1].clone()
    R=float(response_error(teacher,nchanged,intact,changed));effect=competence(intact,groups,variant)['margin']-competence(changed,groups,variant)['margin'];rr=dict(kind=kind,seed=seed,arm=arm,target_index=i,family=row['family'],region=row['region'],input_key=row['input_key'],token_hash=spec['token_hash'],group=spec['group'],noop=noop,delta_energy=float(delta.double().square().sum()),K=K,R=R,native_response_energy=energy,native_margin_effect=nm,reconstructed_margin_effect=effect,margin_squared_error=(effect-nm)**2,native_margin_energy=nm**2);records.append(rr)
    sk=arm,row['region']
    if sk not in sampled:samples.append(dict(row=len(records)-1,native=teacher.cpu(),native_changed=nchanged.cpu(),intact=intact.cpu(),changed=changed.cpu(),delta=delta.cpu(),groups=groups,variant=variant));sampled.add(sk)
   if k%30==29:print('DIRECTIONS',kind,seed,k+1,'FORWARDS',calls,flush=True)
  completed.append([kind,seed]);tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),records=records,samples=samples,calls=calls,completed=completed),tmp);tmp.replace(progress)
 summaries=[]
 for e in p['models']:
  base=[r for r in records if all(r[k]==e[k] for k in ['kind','seed','arm'])]
  for region in [None,'prefix','task']:
   for family in [None]+sorted({x['family'] for x in base}):
    rs=[x for x in base if (region is None or x['region']==region) and (family is None or x['family']==family)];den=sum(x['native_response_energy'] for x in rs);md=sum(x['native_margin_energy'] for x in rs);err=sum(x['R'] for x in rs);me=sum(x['margin_squared_error'] for x in rs);summaries.append({k:e[k] for k in ['kind','seed','arm']}|dict(region=region,family=family,endpoints=len(rs),nonzero=sum(not x['noop'] for x in rs),R=err/len(rs),native_response_energy_sum=den,relative_response_error=err/den if den else None,margin_error_sum=me,native_margin_energy_sum=md,relative_margin_error=me/md if md else None))
 sp=A/'emlp_m28_unseen_samples.pt';torch.save(samples,sp);out=dict(protocol_sha256=sha(pp),targets_sha256=sha(tp),checkpoint_sha256=receipt.weights_sha256,source_sha256=sha(Path(__file__)),forward_count=calls,records=records,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp));op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,len(records),flush=True)
if __name__=='__main__':main()
