"""M27 final training-target fit, separated from causal transfer validation."""
from pathlib import Path
import json,hashlib
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from collect_response_targets import response_error
P=Path(__file__).resolve().parent/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');rp=P/'aggregate_kl_models.json';r=json.loads(rp.read_text());assert json.loads((P/'aggregate_kl_models_audit.json').read_text())['result_sha256']==sha(rp);tp=P/'response_training_targets.json';t=json.loads(tp.read_text());cp=Path(t['cache_path']);assert sha(cp)==t['cache_sha256'];cache=torch.load(cp,weights_only=True);op=P/'aggregate_kl_training_fit.json';assert not op.exists();model,receipt=load_model();device=next(model.parameters()).device;mlp=model.transformer.h[1].mlp;weights={(x['kind'],x['seed']):x['response_weight'] for x in t['calibration']};records=[];samples=[];calls=0
 for entry in [x for x in r['models'] if x['arm']!='initial']:
  assert sha(Path(entry['path']))==entry['sha256'];s=torch.load(entry['path'],weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(device);f.load_state_dict(s['state']);mu=s['mean'].to(device);scale=s['scale'].to(device);intacts={};sampled=set()
  for i,row in enumerate(t['records']):
   if (row['kind'],row['seed'])!=(entry['kind'],entry['seed']):continue
   key=row['input_key'];c=cache['inputs'][key];ids=c['ids'].to(device);teacher=c['native'].to(device)
   if key not in intacts:
    rec,_,_=f((c['write'].to(device)-mu)/scale);q=rec*scale+mu;out,_=trace(model,ids,replace=lambda j,z,w:q+mlp.Down_bias if j==1 else w);calls+=1;lp=teacher.double().log_softmax(-1);K=float((lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean());intacts[key]=(q,out,K)
   q,out,K=intacts[key];target=cache['targets'][i];delta=target['delta'].to(device)
   if row['noop']:changed=out[0,-1]
   else:
    lg,_=trace(model,ids,replace=lambda j,z,w:q-delta+mlp.Down_bias if j==1 else w);calls+=1;changed=lg[0,-1]
   R=float(response_error(teacher[0,-1],target['native_changed'].to(device),out[0,-1],changed));weight=weights[entry['kind'],entry['seed']];records.append(dict(target_index=i,kind=entry['kind'],seed=entry['seed'],arm=entry['arm'],family=row['family'],region=row['region'],K=K,R=R,weight=weight,combined=K+weight*R,initial_K=row['initial_K'],initial_R=row['initial_R']))
   if row['region'] not in sampled:samples.append(dict(row=len(records)-1,teacher=teacher.cpu(),output=out.cpu(),native_changed=target['native_changed'],reconstructed_changed=changed.cpu()));sampled.add(row['region'])
  print('TRAIN_TARGET_EVAL',entry['kind'],entry['seed'],entry['arm'],flush=True)
 summaries=[]
 for entry in [x for x in r['models'] if x['arm']!='initial']:
  base=[x for x in records if all(x[k]==entry[k] for k in ['kind','seed','arm'])]
  for family in [None]+sorted({x['family'] for x in base}):
   rr=base if family is None else [x for x in base if x['family']==family];summaries.append({k:entry[k] for k in ['kind','seed','arm']}|dict(family=family,scheduled_endpoints=len(rr),**{k:sum(x[k] for x in rr)/len(rr) for k in ['K','R','combined','initial_K','initial_R']}))
 ap=A/'emlp_m27_aggregate_kl_fit_samples.pt';torch.save(samples,ap);out=dict(status='completed_training_diagnostic_only',training_results_sha256=sha(rp),targets_sha256=sha(tp),forward_count=calls,records=records,summaries=summaries,samples_path=str(ap),samples_sha256=sha(ap));op.write_text(json.dumps(out,indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()
