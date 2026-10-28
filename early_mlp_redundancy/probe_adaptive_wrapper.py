"""M31 aligned fixed task-span perturbations across paired wrappers."""
from pathlib import Path
import hashlib,json,math
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from collect_response_targets import response_error
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=P/'adaptive_wrapper_protocol.json';p=json.loads(pp.read_text());tp=P/'response_training_targets.json';t=json.loads(tp.read_text());assert sha(tp)==p['targets_sha256'];ep=P/'context_expansion_protocol.json';assert sha(ep)==p['expansion_sha256'];expanded={x['id']:x for x in json.loads(ep.read_text())['rows']};cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);up=P/'unseen_direction_results.json';old=json.loads(up.read_text());assert json.loads((P/'unseen_direction_audit.json').read_text())['result_sha256']==sha(up);reference={(x['target_index'],x['arm']):x for x in old['records']};fixedp=P/'wrapper_crossover_results.json';fixed=json.loads(fixedp.read_text());assert json.loads((P/'wrapper_crossover_audit.json').read_text())['result_sha256']==sha(fixedp);fixedrows={(x['target_index'],x['context'],x['arm']):x for x in fixed['records']};op=P/'adaptive_wrapper_results.json';assert not op.exists();model,receipt=load_model();device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;records=[];samples=[];calls=0;completed=[];nativecache={};progress=A/'emlp_m31_adaptive_wrapper_progress.pt'
 if progress.exists():
  prev=torch.load(progress,weights_only=True);assert prev['protocol_sha256']==sha(pp);records=prev['records'];samples=prev['samples'];calls=prev['calls'];completed=prev['completed'];nativecache=prev['nativecache'];print('RESUME',completed,flush=True)
 for kind,seed in [('scalar',71),('block',71),('scalar',72),('block',72)]:
  if [kind,seed] in completed:continue
  es=[e for e in p['models'] if (e['kind'],e['seed'])==(kind,seed)];fs={};states={};memo={};sampled=set();deltas={}
  for e in es:
   assert sha(Path(e['path']))==e['sha256'];s=torch.load(e['path'],weights_only=True);f=Featurizer(kind,seed).to(device);f.load_state_dict(s['state']);fs[e['arm']]=f;states[e['arm']]=(s['mean'].to(device),s['scale'].to(device))
  selected=[x for x in p['rows'] if (x['kind'],x['seed'])==(kind,seed)]
  for k,spec in enumerate(selected):
   i=spec['target_index'];target=t['records'][i];original=cache['inputs'][target['input_key']];initial=cache['initials'][target['initial_key']];pair=expanded[spec['pair_id']];endpoint=spec['endpoint'];key=spec['pair_id'],endpoint;ids=torch.tensor([pair['token_ids'][endpoint]],device=device);boundary=spec['prefix_length'];assert ids[0,boundary:].tolist()==spec['task_token_ids'];groups=pair['answer_groups'][endpoint];variant=pair['variants'][endpoint]
   if key not in nativecache:
    native,cap=trace(model,ids);calls+=1;write=mlp.Down(cap[1]['products']);nativecache[key]=dict(native=native[0,-1].cpu(),write=write.cpu(),ids=ids.cpu())
    if spec['context']=='reference':assert torch.equal(native.cpu(),original['native']) and torch.equal(write.cpu(),original['write'])
   c=nativecache[key];teacher=c['native'].to(device);write=c['write'].to(device);lp=teacher.double().log_softmax(-1);nt=competence(teacher,groups,variant)
   if i not in deltas:
    atoms=torch.arange(4*spec['group'],4*spec['group']+4,device=device);full=(initial['code'].to(device)[:,:,atoms]@initial['D'].to(device)[:,atoms].T)*initial['scale'].to(device)*cache['targets'][i]['mask'].to(device);assert not bool((full[:,:spec['source_prefix_length']]!=0).any());deltas[i]=full[:,spec['source_prefix_length']:].clone()
   transported=deltas[i];mu,scale=states['initial'];_,current_support,current_code=fs['initial']((write-mu)/scale);D=torch.nn.functional.normalize(fs['initial'].dictionary,dim=0);atoms=torch.arange(4*spec['group'],4*spec['group']+4,device=device);current_full=(current_code[:,:,atoms]@D[:,atoms].T)*scale;taskdelta=current_full[:,boundary:].clone()
   if spec['context']=='reference':assert torch.equal(taskdelta,transported)
   delta=torch.zeros_like(write);assert delta[:,boundary:].shape==taskdelta.shape;delta[:,boundary:]=taskdelta;noop=bool((delta==0).all());taskhash=hashlib.sha256(taskdelta.cpu().numpy().tobytes()).hexdigest();tn=transported.double().norm();an=taskdelta.double().norm();cosine=float((transported.double()*taskdelta.double()).sum()/(tn*an)) if tn>0 and an>0 else None;old_support=initial['support'].to(device)[:,spec['source_prefix_length']:,atoms];support_changes=int((old_support!=current_support[:,boundary:,atoms]).any(-1).sum());geometry=dict(transported_task_sha256=hashlib.sha256(transported.cpu().numpy().tobytes()).hexdigest(),transported_energy=float(tn.square()),adaptive_energy=float(an.square()),task_delta_cosine=cosine,task_delta_difference_energy=float((taskdelta.double()-transported.double()).square().sum()),group_support_changed_tokens=support_changes,task_tokens=taskdelta.shape[1])
   if noop:nchanged=teacher
   else:
    out,_=trace(model,ids,replace=lambda j,z,w:w-delta if j==1 else w);calls+=1;nchanged=out[0,-1].clone()
   nr=nchanged.double().log_softmax(-1)-lp;energy=float((lp.exp()*nr.square()).sum());nm=nt['margin']-competence(nchanged,groups,variant)['margin']
   for arm,f in fs.items():
    ik=key,arm
    if ik not in memo:
     mu,scale=states[arm];pred,_,_=f((write-mu)/scale);q=pred*scale+mu;out,_=trace(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;memo[ik]=(q,out[0,-1].clone())
    q,intact=memo[ik];it=competence(intact,groups,variant)
    if noop:changed=intact
    else:
     out,_=trace(model,ids,replace=lambda j,z,w:q-delta+mlp.Down_bias if j==1 else w);calls+=1;changed=out[0,-1].clone()
    R=float(response_error(teacher,nchanged,intact,changed));effect=it['margin']-competence(changed,groups,variant)['margin'];record=dict(**geometry,kind=kind,seed=seed,arm=arm,target_index=i,family=spec['family'],context=spec['context'],pair_id=spec['pair_id'],endpoint=endpoint,group=spec['group'],prefix_length=boundary,noop=noop,task_delta_sha256=taskhash,delta_energy=float(delta.double().square().sum()),R=R,native_response_energy=energy,native_margin_effect=nm,reconstructed_margin_effect=effect,margin_squared_error=(effect-nm)**2,native_margin_energy=nm**2,intact_KL=float((lp.exp()*(lp-intact.double().log_softmax(-1))).sum()),native_binary=nt['binary'],native_top1=nt['top1'],reconstructed_binary=it['binary'],reconstructed_top1=it['top1']);records.append(record)
    prior_fixed=fixedrows[i,spec['context'],arm]
    for field in ['intact_KL','native_binary','native_top1','reconstructed_binary','reconstructed_top1']:assert record[field]==prior_fixed[field]
    assert geometry['transported_task_sha256']==prior_fixed['task_delta_sha256']
    assert math.isclose(geometry['transported_energy'],prior_fixed['delta_energy'],rel_tol=1e-12,abs_tol=1e-9)
    if spec['context']=='reference':
     prior=reference[i,arm]
     for field in ['R','native_response_energy','native_margin_effect','reconstructed_margin_effect','delta_energy']:assert math.isclose(record[field],prior[field],rel_tol=1e-8,abs_tol=1e-10),(field,record[field],prior[field])
    sk=arm,spec['context']
    if sk not in sampled:samples.append(dict(row=len(records)-1,native=teacher.cpu(),native_changed=nchanged.cpu(),intact=intact.cpu(),changed=changed.cpu(),delta=delta.cpu(),task_delta=taskdelta.cpu(),ids=ids.cpu(),groups=groups,variant=variant,write=write.cpu(),current_support=current_support.cpu(),reference_support=old_support.cpu(),initial_key=target['initial_key']));sampled.add(sk)
   if k%30==29:print('CONTEXT_DIRECTIONS',kind,seed,k+1,'FORWARDS',calls,flush=True)
  completed.append([kind,seed]);tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),records=records,samples=samples,calls=calls,completed=completed,nativecache=nativecache),tmp);tmp.replace(progress)
 summaries=[]
 for e in p['models']:
  for context in p['contexts']:
   base=[x for x in records if all(x[k]==e[k] for k in ['kind','seed','arm']) and x['context']==context]
   for family in [None]+sorted({x['family'] for x in base}):
    rs=base if family is None else [x for x in base if x['family']==family];re=sum(x['R'] for x in rs);ne=sum(x['native_response_energy'] for x in rs);me=sum(x['margin_squared_error'] for x in rs);nm=sum(x['native_margin_energy'] for x in rs);summaries.append({k:e[k] for k in ['kind','seed','arm']}|dict(context=context,family=family,endpoints=len(rs),nonzero=sum(not x['noop'] for x in rs),R=re/len(rs),native_response_energy_sum=ne,relative_response_error=re/ne if ne else None,margin_error_sum=me,native_margin_energy_sum=nm,relative_margin_error=me/nm if nm else None,intact_KL=sum(x['intact_KL'] for x in rs)/len(rs),native_binary_rate=sum(x['native_binary'] for x in rs)/len(rs),native_top1_rate=sum(x['native_top1'] for x in rs)/len(rs)))
 sp=A/'emlp_m31_adaptive_wrapper_samples.pt';torch.save(samples,sp);out=dict(protocol_sha256=sha(pp),reference_results_sha256=sha(up),fixed_results_sha256=sha(fixedp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,unique_native_inputs=len(nativecache),records=records,summaries=summaries,samples_path=str(sp),samples_sha256=sha(sp));op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,len(records),flush=True)
if __name__=='__main__':main()
