"""M32 native fixed-delta replay and common-metric response geometry."""
from pathlib import Path
import json,hashlib,math,statistics
import torch
from inspect_model import load_model,trace
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=P/'native_geometry_protocol.json';p=json.loads(pp.read_text());wp=P/'wrapper_crossover_results.json';old=json.loads(wp.read_text());assert sha(wp)==p['fixed_results_sha256'];assert json.loads((P/'wrapper_crossover_audit.json').read_text())['result_sha256']==sha(wp);prior={(x['target_index'],x['context']):x for x in old['records'] if x['arm']=='initial'};tp=P/'response_training_targets.json';t=json.loads(tp.read_text());cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);expanded={x['id']:x for x in json.loads((P/'context_expansion_protocol.json').read_text())['rows']};model,receipt=load_model();device=next(model.parameters()).device;op=P/'native_response_geometry.json';assert not op.exists();nativecache={};vectors=[];replays=[];deltas={};calls=0;completed=[];progress=A/'emlp_m32_native_geometry_progress.pt'
 if progress.exists():
  saved=torch.load(progress,weights_only=True);assert saved['protocol_sha256']==sha(pp);nativecache=saved['nativecache'];vectors=saved['vectors'];replays=saved['replays'];deltas=saved['deltas'];calls=saved['calls'];completed=saved['completed'];print('RESUME',completed,flush=True)
 for kind,seed in [('scalar',71),('block',71),('scalar',72),('block',72)]:
  if [kind,seed] in completed:continue
  for num,spec in enumerate(x for x in p['rows'] if (x['kind'],x['seed'])==(kind,seed)):
   i=spec['target_index'];target=t['records'][i];orig=cache['inputs'][target['input_key']];initial=cache['initials'][target['initial_key']];pair=expanded[spec['pair_id']];ep=spec['endpoint'];key=spec['pair_id'],ep;ids=torch.tensor([pair['token_ids'][ep]],device=device);boundary=spec['prefix_length'];groups=pair['answer_groups'][ep];variant=pair['variants'][ep]
   if key not in nativecache:
    out,_=trace(model,ids);calls+=1
    if spec['context']=='reference':assert torch.equal(out.cpu(),orig['native'])
    nativecache[key]=out[0,-1].cpu().clone()
   native=nativecache[key].to(device)
   if i not in deltas:
    atoms=torch.arange(4*spec['group'],4*spec['group']+4,device=device);full=(initial['code'].to(device)[:,:,atoms]@initial['D'].to(device)[:,atoms].T)*initial['scale'].to(device)*cache['targets'][i]['mask'].to(device);deltas[i]=full[:,spec['source_prefix_length']:].cpu().clone()
   taskdelta=deltas[i].to(device);assert ids[0,boundary:].tolist()==spec['task_token_ids'];delta=torch.zeros((1,ids.shape[1],taskdelta.shape[-1]),device=device,dtype=taskdelta.dtype);delta[:,boundary:]=taskdelta;noop=bool((delta==0).all())
   if noop:changed=native
   else:
    out,_=trace(model,ids,replace=lambda j,z,w:w-delta if j==1 else w);calls+=1;changed=out[0,-1]
   lp=native.double().log_softmax(-1);response=changed.double().log_softmax(-1)-lp;nt=competence(native,groups,variant);margin=nt['margin']-competence(changed,groups,variant)['margin'];energy=float((lp.exp()*response.square()).sum());oldrow=prior[i,spec['context']];taskhash=hashlib.sha256(deltas[i].numpy().tobytes()).hexdigest();assert taskhash==oldrow['task_delta_sha256'];assert noop==oldrow['noop']
   for field,value in [('native_response_energy',energy),('native_margin_effect',margin),('delta_energy',float(delta.double().square().sum()))]:assert math.isclose(value,oldrow[field],rel_tol=1e-8,abs_tol=1e-10)
   assert nt['binary']==oldrow['native_binary'] and nt['top1']==oldrow['native_top1'];replays.append(dict(target_index=i,context=spec['context'],kind=kind,seed=seed,family=spec['family'],noop=noop,task_delta_sha256=taskhash,native_response_energy=energy,native_margin_effect=margin,native_binary=nt['binary'],native_top1=nt['top1']));vectors.append(dict(target_index=i,context=spec['context'],native_key=key,changed=changed.cpu().clone(),groups=groups,variant=variant))
   if num%60==59:print('NATIVE',kind,seed,num+1,'FORWARDS',calls,flush=True)
  completed.append([kind,seed]);tmp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),nativecache=nativecache,vectors=vectors,replays=replays,deltas=deltas,calls=calls,completed=completed),tmp);tmp.replace(progress)
 lookup={(x['target_index'],x['context']):x for x in vectors};metadata={s['target_index']:s for s in p['rows']};comparisons=[];maxidentity=0
 for i in sorted(metadata):
  ref=lookup[i,'reference'];a=nativecache[ref['native_key']].double();ac=ref['changed'].double();alp=a.log_softmax(-1);w=alp.exp();rref=ac.log_softmax(-1)-alp;meanref=(w*rref).sum();cref=rref-meanref;refenergy=(w*rref.square()).sum();refcenterenergy=(w*cref.square()).sum()
  for context in ['passage','sample']:
   v=lookup[i,context];b=nativecache[v['native_key']].double();bc=v['changed'].double();rb=bc.log_softmax(-1)-b.log_softmax(-1);cb=rb-(w*rb).sum();dl=bc-b;cl=dl-(w*dl).sum();identity=float((cb-cl).abs().max());maxidentity=max(maxidentity,identity);assert identity<1e-10
   difference=rb-rref;mean=(w*difference).sum();center=difference-mean;total=(w*difference.square()).sum();shape=(w*center.square()).sum();constant=mean.square();assert abs(float(total-shape-constant))<1e-10;eb=(w*cb.square()).sum();cos=float((w*cref*cb).sum()/(refcenterenergy*eb).sqrt()) if refcenterenergy>0 and eb>0 else None;meta=metadata[i];comparisons.append(dict(target_index=i,kind=meta['kind'],seed=meta['seed'],family=meta['family'],context=context,total_difference=float(total),centered_difference=float(shape),constant_difference=float(constant),reference_response_energy=float(refenergy),reference_centered_energy=float(refcenterenergy),context_centered_energy=float(eb),centered_cosine=cos,centered_norm_ratio=float((eb/refcenterenergy).sqrt()) if refcenterenergy>0 else None,identity_max_error=identity))
 summaries=[]
 for kind in ['scalar','block']:
  for seed in [71,72]:
   for context in ['passage','sample']:
    base=[x for x in comparisons if (x['kind'],x['seed'],x['context'])==(kind,seed,context)]
    for family in [None]+sorted({x['family'] for x in base}):
     rs=base if family is None else [x for x in base if x['family']==family];keys=['total_difference','centered_difference','constant_difference','reference_response_energy','reference_centered_energy','context_centered_energy'];sums={k:sum(x[k] for x in rs) for k in keys};valid=[x for x in rs if x['centered_cosine'] is not None];summaries.append(dict(kind=kind,seed=seed,context=context,family=family,directions=len(rs),nonzero_geometry=len(valid),sums=sums,centered_fraction=sums['centered_difference']/sums['total_difference'] if sums['total_difference'] else None,relative_total_difference=sums['total_difference']/sums['reference_response_energy'] if sums['reference_response_energy'] else None,relative_centered_difference=sums['centered_difference']/sums['reference_centered_energy'] if sums['reference_centered_energy'] else None,median_centered_cosine=statistics.median(x['centered_cosine'] for x in valid) if valid else None,median_centered_norm_ratio=statistics.median(x['centered_norm_ratio'] for x in valid) if valid else None))
 sp=A/'emlp_m32_native_response_vectors.pt';torch.save(dict(nativecache=nativecache,vectors=vectors,deltas=deltas),sp);out=dict(protocol_sha256=sha(pp),fixed_results_sha256=sha(wp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,unique_native_inputs=len(nativecache),replays=replays,comparisons=comparisons,summaries=summaries,max_centering_identity_error=maxidentity,vectors_path=str(sp),vectors_sha256=sha(sp));op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,len(vectors),len(comparisons),flush=True)
if __name__=='__main__':main()
