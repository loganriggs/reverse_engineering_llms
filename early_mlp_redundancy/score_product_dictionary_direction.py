"""M97 fixed-endpoint directional effects, training or source holdout; no fitting."""
import json,sys
import torch
from fit_pca_encoder import V,A,sha
from inspect_model import trace
from bilinear_input_causal_loss import InputCausalExperiment
from product_dictionary_direction import FrozenDictionaryDirection,scalar_derivative


def source_losses(e,qr,qd,key):
 c=e.cache[key];nlp=c['native'].cuda().double().log_softmax(-1)
 def run(write=None,delta=None):
  callback=(lambda j,z,w:w+delta if j==1 else w) if delta is not None else (lambda j,z,w:write+e.mlp.Down_bias if j==1 else w)
  logits,_=trace.__wrapped__(e.model,c['ids'].cuda(),replace=callback);e.calls+=1;return logits[0,-1]
 intact=run(qr);K=(nlp.exp()*(nlp-intact.double().log_softmax(-1))).sum();branches=[];balanced=[]
 for target in c['targets']:
  region=target['region'];pos,dp=e.addresses[key,region];nd=target['delta'].cuda();rd=torch.zeros_like(qr)
  if pos:rd=rd.index_copy(1,torch.tensor(pos,device='cuda'),qd[:,dp]-qr[:,pos])
  teacher=target['native_changed'].cuda().double().log_softmax(-1)-nlp
  for corner,background,delta,base in [('native_reconstructed',c['write'].cuda(),rd,c['native'].cuda()),('reconstructed_native',qr,nd,intact),('reconstructed_reconstructed',qr,rd,intact)]:
   if region=='identity':changed=base
   elif corner=='native_reconstructed':changed=run(delta=delta)
   else:changed=run(background+delta)
   response=changed.double().log_softmax(-1)-base.double().log_softmax(-1);R=(nlp.exp()*(response-teacher).square()).sum();branches.append(dict(region=region,corner=corner,base=base,changed=changed,R=R))
   if region=='identity':assert float(R.detach())==0
   else:balanced.append(R/e.den[c['family'],region])
 return K,torch.stack(balanced).mean(),intact,branches


def main():
 cohort=sys.argv[1];assert cohort in ['training','source'];torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');pp=V/'product_dictionary_direction_execution.json';protocol=json.loads(pp.read_text())
 for name,digest in protocol['references'].items():assert sha(V/name)==digest
 for name,digest in protocol['sources'].items():assert sha(V.parent/name)==digest
 e=InputCausalExperiment();results=[];files=[];replayed=0
 if cohort=='training':
  prior=json.loads((V/'product_dictionary_training_targets.json').read_text());case_sources=None
 else:
  prior=json.loads((V/'product_dictionary_source_factorial.json').read_text());np=json.loads((V/'source_holdout_native.json').read_text());assert sha(np['samples_path'])==np['samples_sha256'];native=torch.load(np['samples_path'],weights_only=True);pairs=json.loads((V/'source_holdout_native_protocol.json').read_text())['pairs'];case_sources={}
  for file in prior['files']:
   assert sha(file['path'])==file['sha256'];blob=torch.load(file['path'],weights_only=True,mmap=True);pair=next(p for p in pairs if p['id']==file['pair_id']);pid=pair['id'];case_sources[pid]=blob
   for ep in [0,1]:
    original=f'{pid}|{ep}';key='heldout:'+original;assert key not in e.cache;targets=[]
    for target in np['records']:
     if target['pair_id']!=pid or target['endpoint']!=ep or target['status']!='evaluated':continue
     delta=torch.zeros_like(blob['writes'][ep]);delta[:,target['recipient_positions']]=native['deltas'][target['sample_key']]['delta'];targets.append(dict(region=target['region'],delta=delta,native_changed=native['changed'][target['sample_key']]));e.addresses[key,target['region']]=(target['recipient_positions'],target['donor_positions'])
    e.cache[key]=dict(ids=torch.tensor([pair['token_ids'][ep]]),write=blob['writes'][ep],native=native['baselines'][original]['logits'],family=pair['family'],targets=targets);e.input_cache[key]=dict(input=blob['inputs'][ep])
 progress=A/f'emlp_m97_direction_{cohort}_progress.pt';output=V/f'product_dictionary_direction_{cohort}.json';assert not output.exists();completed=0
 if progress.exists():
  old=torch.load(progress,weights_only=True);assert old['protocol_sha256']==sha(pp) and old['source_sha256']==sha(__file__);results,completed,replayed,e.calls=[old[k] for k in ['rows','completed_seeds','replayed','calls']]
 for si,seed in enumerate([71,72]):
  if si<completed:continue
  direction=FrozenDictionaryDirection(seed)
  if cohort=='training':
   assignments=e.protocol['schedules'][str(seed)]['task_updates'];old_rows={};old_samples={}
   for file in prior['files']:
    if file['seed']!=seed:continue
    assert sha(file['path'])==file['sha256'];samples=torch.load(file['path'],weights_only=True)
    for sample in samples:
     row=prior['rows'][sample['row']];idx=(row['arm'],row['assignment'],row['key']);old_rows[idx]=row;old_samples[idx]=sample
  else:assignments=[dict(pair_id=p['id'],family=p['family']) for p in pairs]
  for assignment_index,assignment in enumerate(assignments):
   pid=assignment['pair_id'];prefix='' if cohort=='training' else 'heldout:'
   keys=[prefix+f'{pid}|{ep}' for ep in [0,1]];prepared=[direction.prepare(e.input_cache[k]['input'].cuda()) for k in keys]
   for ep,key in enumerate(keys):
    recipient,donor=prepared[ep],prepared[1-ep];native_energy=float(e.cache[key]['write'].double().square().sum());movement=float(recipient[1].double().square().sum());secant=float((recipient[0][1].double()-recipient[0][0].double()-recipient[1].double()).square().sum())
    for anchor,arm in enumerate(['intact_KL','joint_three_corner']):
     t=torch.tensor(float(anchor),device='cuda',requires_grad=True);qr=direction.at(*recipient,t,anchor);qd=direction.at(*donor,t,anchor)
     K,B,intact,branches=(e.joint_losses if cohort=='training' else lambda qr,qd,key:source_losses(e,qr,qd,key))(qr,qd,key)
     if cohort=='training':
      idx=(arm,assignment_index,key);old=old_samples[idx];assert float(K.detach())==old_rows[idx]['K'] and float(B.detach())==old_rows[idx]['B'];assert torch.equal(intact.detach().cpu(),old['intact'])
     else:
      blob=case_sources[pid];ms=f'product_dictionary:{seed}:{arm}';assert torch.equal(qr.detach().cpu(),blob['reconstructions'][ms+f'|{ep}']['write']);assert torch.equal(intact.detach().cpu(),blob['baselines'][ms+f'|{ep}'])
     slopes=[]
     for branch in branches:
      if cohort=='training':
       previous=next(b for b in old['branches'] if b['region']==branch['region'] and b['corner']==branch['corner']);assert torch.equal(branch['base'].detach().cpu(),previous['base']);assert torch.equal(branch['changed'].detach().cpu(),previous['changed'])
      else:assert torch.equal(branch['changed'].detach().cpu(),blob['changed'][ms+f'|{ep}|'+branch['region']+'|'+branch['corner']])
      replayed+=1;derivative=scalar_derivative(branch['R'],t)
      if branch['region']=='identity':assert derivative==float(branch['R'].detach())==0
      slopes.append(dict(region=branch['region'],corner=branch['corner'],R=float(branch['R'].detach()),slope=derivative))
     ks=scalar_derivative(K,t);bs=scalar_derivative(B,t);nonid=[b for b in slopes if b['region']!='identity'];independent_bs=sum(b['slope']/e.den[assignment['family'],b['region']] for b in nonid)/len(nonid);assert abs(bs-independent_bs)<=1e-7+1e-5*abs(bs)
     results.append(dict(seed=seed,arm=arm,anchor=anchor,assignment=assignment_index,key=f'{pid}|{ep}',family=assignment['family'],K=float(K.detach()),B=float(B.detach()),K_slope=ks,B_slope=bs,branches=slopes,native_write_energy=native_energy,write_direction_energy=movement,secant_rounding_error_energy=secant))
     assert all(p.grad is None for p in e.model.parameters());del qr,qd,K,B,intact,branches
   if (assignment_index+1)%11==0:print('DIRECTION',cohort,seed,assignment_index+1,flush=True)
  temp=progress.with_suffix('.tmp');torch.save(dict(protocol_sha256=sha(pp),source_sha256=sha(__file__),rows=results,completed_seeds=si+1,replayed=replayed,calls=e.calls),temp);temp.replace(progress)
 expected=528 if cohort=='training' else 112;assert len(results)==expected
 out=dict(status='scored_pending_audit',source_sha256=sha(__file__),protocol_sha256=sha(pp),cohort=cohort,rows=results,forward_count=e.calls,replayed_response_vectors=replayed,prior_result_sha256=sha(V/('product_dictionary_training_targets.json' if cohort=='training' else 'product_dictionary_source_factorial.json')),scope='Two existing endpoints only; all output vectors replayed bit-exact against prior saved evidence. Slopes through each own-background response, no optimizer updates. Anchored float32 endpoint values with independently checked affine write tangent; no heldout gradients used to train.')
 output.write_text(json.dumps(out,indent=2)+'\n');print('DONE',cohort,len(results),e.calls,replayed,flush=True)
if __name__=='__main__':main()
