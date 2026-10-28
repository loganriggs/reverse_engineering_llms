"""Common native intact and paired-generator response loss for M35 onward."""
from pathlib import Path
import hashlib,json
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
from collect_response_targets import response_error
P=Path(__file__).resolve().parent/'validation'
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Experiment:
 def __init__(self):
  self.pp=P/'joint_shared_comparison_protocol.json';self.p=json.loads(self.pp.read_text())
  tp=P/'response_training_targets.json';assert sha(tp)==self.p['target_source_sha256'];self.targets=json.loads(tp.read_text());cp=Path(self.targets['cache_path']);assert sha(cp)==self.targets['cache_sha256'];self.cache=torch.load(cp,weights_only=True)
  assert sha(P/'context_training_protocol.json')==self.p['schedule_source_sha256']
  assert sha(P/'diverse_supervision_split.json')==self.p['split_sha256']
  self.model,self.receipt=load_model();self.model.requires_grad_(False);self.device=next(self.model.parameters()).device;self.mlp=self.model.transformer.h[1].mlp;self.calls=0
  assert self.receipt.weights_sha256==self.targets['checkpoint_sha256']
 def featurizer(self,entry):
  path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);f=Featurizer(entry['kind'],entry['seed']).to(self.device);f.load_state_dict(s['state']);return f,s['mean'].to(self.device),s['scale'].to(self.device)
 def outputs(self,q,c,indices):
  ids=c['ids'].to(self.device)
  out,_=trace.__wrapped__(self.model,ids,replace=lambda j,z,w:q+self.mlp.Down_bias if j==1 else w);self.calls+=1;changed=[]
  for i in indices:
   meta=self.targets['records'][i];assert c is self.cache['inputs'][meta['input_key']]
   delta=self.cache['targets'][i]['delta'].to(self.device).detach()
   assert delta.shape==q.shape and not delta.requires_grad
   if meta['noop']:changed.append(out[0,-1])
   else:
    v,_=trace.__wrapped__(self.model,ids,replace=lambda j,z,w:q-delta+self.mlp.Down_bias if j==1 else w);self.calls+=1;changed.append(v[0,-1])
  return out,changed
 def losses(self,q,c,indices):
  out,changed=self.outputs(q,c,indices);native=c['native'].to(self.device);lp=native.double().log_softmax(-1)
  K=(lp.exp()*(lp-out.double().log_softmax(-1))).sum(-1).mean()
  Rs=[response_error(native[0,-1],self.cache['targets'][i]['native_changed'].to(self.device).detach(),out[0,-1],v) for i,v in zip(indices,changed)]
  R=torch.stack(Rs).mean() if Rs else K.new_zeros(())
  return K,R,out,changed,Rs
