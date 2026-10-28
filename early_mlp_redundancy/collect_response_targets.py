"""M19 training-only native perturbation targets and frozen loss calibration."""
from pathlib import Path
import json,hashlib,math
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def response_error(native,native_changed,reconstructed,reconstructed_changed):
 lp=native.double().log_softmax(-1);nr=native_changed.double().log_softmax(-1)-lp;fr=reconstructed_changed.double().log_softmax(-1)-reconstructed.double().log_softmax(-1);return (lp.exp()*(fr-nr).square()).sum(-1).mean()
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');pp=P/'response_supervision_protocol.json';p=json.loads(pp.read_text());op=P/'response_training_targets.json';assert not op.exists();expanded={x['id']:x for x in json.loads((P/'context_expansion_protocol.json').read_text())['rows']};screen=json.loads((P/'context_expansion_competence.json').read_text());passing={(x['template'],x['lexical_split'],x['family']) for x in screen['cohorts'] if x['pass_gate']};model,receipt=load_model();model.requires_grad_(False);device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;inputs={};initials={};records=[];tensors=[];calibration=[];calls=0
 for entry in p['initial_models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];state=torch.load(path,weights_only=True);kind,seed=entry['kind'],entry['seed'];f=Featurizer(kind,seed).to(device);f.load_state_dict(state['state']);mu=state['mean'].to(device);scale=state['scale'].to(device);D=torch.nn.functional.normalize(f.dictionary,dim=0);start=len(records)
  for u in p['schedules'][str(seed)]['task_updates']:
   row=expanded[u['expanded_pair_id']];assert row['training_eligible'] and (row['template'],row['lexical_split'],row['family']) in passing
   for ep in [0,1]:
    ik=f"{row['id']}:{ep}";fk=f'{kind}:{seed}:{ik}'
    if ik not in inputs:
     ids=torch.tensor([row['token_ids'][ep]],device=device);native,cap=trace(model,ids);calls+=1;v=mlp.Down(cap[1]['products']);inputs[ik]=dict(ids=ids.cpu(),write=v.cpu(),native=native.cpu(),prefix_length=row['prefix_length'])
    c=inputs[ik];ids=c['ids'].to(device);v=c['write'].to(device);native=c['native'].to(device)
    if fk not in initials:
     rec,support,code=f((v-mu)/scale);q=rec*scale+mu;out,_=trace(model,ids,replace=lambda i,z,w:q+mlp.Down_bias if i==1 else w);calls+=1;lp=native.double().log_softmax(-1);K=float((lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean());initials[fk]=dict(q=q.cpu(),output=out.cpu(),support=support.cpu(),code=code.cpu(),D=D.cpu(),mean=mu.cpu(),scale=scale.cpu(),K=K)
    initial=initials[fk];q=initial['q'].to(device);out=initial['output'].to(device);code=initial['code'].to(device);g=u['endpoint_groups'][ep];atoms=torch.arange(4*g,4*g+4,device=device);mask=torch.zeros_like(q[:,:,:1]);n=c['prefix_length'];mask[:,:n]=1 if u['region']=='prefix' else 0;mask[:,n:]=1 if u['region']=='task' else 0;delta=(code[:,:,atoms]@D[:,atoms].T)*scale*mask;noop=bool((delta==0).all())
    if noop:nchanged=native[0,-1];fchanged=out[0,-1]
    else:
     nn,_=trace(model,ids,replace=lambda i,z,w:w-delta if i==1 else w);ff,_=trace(model,ids,replace=lambda i,z,w:q-delta+mlp.Down_bias if i==1 else w);calls+=2;nchanged=nn[0,-1];fchanged=ff[0,-1]
    R=float(response_error(native[0,-1],nchanged,out[0,-1],fchanged));nlp=native[0,-1].double().log_softmax(-1);native_energy=float((nlp.exp()*(nchanged.double().log_softmax(-1)-nlp).square()).sum());records.append(dict(kind=kind,seed=seed,task_update=u['task_update'],pair_id=row['id'],endpoint=ep,family=row['family'],region=u['region'],group=g,input_key=ik,initial_key=fk,noop=noop,delta_energy=float(delta.double().square().sum()),initial_K=initial['K'],initial_R=R,native_response_energy=native_energy));tensors.append(dict(delta=delta.cpu(),mask=mask.cpu(),native_changed=nchanged.cpu(),initial_changed=fchanged.cpu()))
  rr=records[start:];Kbar=sum(x['initial_K'] for x in rr)/len(rr);Rbar=sum(x['initial_R'] for x in rr)/len(rr);assert Kbar>0 and Rbar>0 and math.isfinite(Kbar/Rbar);calibration.append(dict(kind=kind,seed=seed,scheduled_endpoints=len(rr),nonzero_directions=sum(not x['noop'] for x in rr),mean_initial_K=Kbar,mean_initial_R=Rbar,response_weight=Kbar/Rbar));print('TARGETS',kind,seed,len(rr),'WEIGHT',Kbar/Rbar,flush=True)
 ap=A/'emlp_m19_response_training_targets.pt';torch.save(dict(inputs=inputs,initials=initials,targets=tensors),ap);out=dict(status='completed_before_optimization',protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,unique_native_inputs=len(inputs),unique_initial_inputs=len(initials),records=records,calibration=calibration,cache_path=str(ap),cache_sha256=sha(ap));op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,'ENDPOINT_TARGETS',len(records),flush=True)
if __name__=='__main__':main()
