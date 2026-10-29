"""M60 task loss interface: common frozen native deltas, raw vs cell scaling."""
from pathlib import Path
import hashlib,json
import torch
from inspect_model import load_model,trace
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Experiment:
 def __init__(self):
  self.pp=V/'balanced_response_training_protocol.json';self.protocol=json.loads(self.pp.read_text());self.cal=json.loads((V/'balanced_response_calibration.json').read_text());assert self.cal['protocol_sha256']==sha(self.pp);self.den={(s['family'],s['region']):s['denominator'] for s in self.protocol['scales']};np=V/'training_causal_coverage_protocol.json';nr=V/'training_causal_coverage.json';assert sha(np)==self.protocol['native_protocol_sha256'];assert sha(nr)==self.protocol['native_results_sha256'];inventory=json.loads(np.read_text());result=json.loads(nr.read_text());sp=Path(result['samples_path']);assert sha(sp)==result['samples_sha256'];saved=torch.load(sp,weights_only=True);self.model,self.receipt=load_model();self.model.requires_grad_(False);self.mlp=self.model.transformer.h[1].mlp;self.calls=0;self.cachepath=A/'emlp_m60_training_cache.pt'
  if self.cachepath.exists():
   data=torch.load(self.cachepath,weights_only=True);assert data['native_protocol_sha256']==sha(np) and data['checkpoint_sha256']==self.receipt.weights_sha256;self.cache=data['inputs'];return
  self.cache={}
  with torch.no_grad():
   for p in inventory['pairs']:
    for ep in [0,1]:
     key=p['id']+f'|{ep}';ids=torch.tensor([p['token_ids'][ep]],device='cuda');out,cap=trace(self.model,ids);self.calls+=1;assert torch.equal(out[0,-1].cpu(),saved['baselines'][key]['logits']);write=self.mlp.Down(cap[1]['products']);targets=[]
     for r in result['records']:
      if r['pair_id']==p['id'] and r['endpoint']==ep and r['layer']==1 and r['status']=='evaluated':
       delta=torch.zeros_like(write);delta[:,r['recipient_positions']]=saved['deltas'][r['sample_key']]['delta'].cuda();targets.append(dict(region=r['region'],delta=delta.cpu(),native_changed=saved['changed'][r['sample_key']],noop=r['noop']))
     self.cache[key]=dict(ids=ids.cpu(),write=write.cpu(),native=out[0,-1].cpu(),family=p['family'],targets=targets)
  torch.save(dict(native_protocol_sha256=sha(np),checkpoint_sha256=self.receipt.weights_sha256,inputs=self.cache),self.cachepath)
 def featurizer(self,e):
  p=Path(e['path']);assert sha(p)==e['sha256'];s=torch.load(p,weights_only=True);f=Featurizer(e['kind'],e['seed']).cuda();f.load_state_dict(s['state']);return f,s['mean'].cuda(),s['scale'].cuda()
 def losses(self,q,c):
  ids=c['ids'].cuda();out,_=trace.__wrapped__(self.model,ids,replace=lambda j,z,w:q+self.mlp.Down_bias if j==1 else w);self.calls+=1;z=out[0,-1];nlp=c['native'].cuda().double().log_softmax(-1);lp=z.double().log_softmax(-1);K=(nlp.exp()*(nlp-lp)).sum();raw=[];balanced=[];branches=[]
  for t in c['targets']:
   if t['noop']:changed=z
   else:
    delta=t['delta'].cuda().detach();v,_=trace.__wrapped__(self.model,ids,replace=lambda j,x,w:q+delta+self.mlp.Down_bias if j==1 else w);self.calls+=1;changed=v[0,-1]
   native_response=t['native_changed'].cuda().double().log_softmax(-1)-nlp;error=(changed.double().log_softmax(-1)-lp)-native_response;R=(nlp.exp()*error.square()).sum();branches.append((t,z,changed,R))
   if t['region']=='identity':assert float(R.detach())==0
   else:raw.append(R);balanced.append(R/self.den[c['family'],t['region']])
  return K,torch.stack(raw).mean(),torch.stack(balanced).mean(),branches
