"""M68 causal loss for differences encoded by the featurizer itself."""
import json,torch
from balanced_response_loss import Experiment,V,sha
from inspect_model import trace
class EncodedExperiment(Experiment):
 def __init__(self):
  super().__init__();self.addresses={}
  for r in json.loads((V/'training_causal_coverage.json').read_text())['records']:
   if r['layer']==1 and r['status']=='evaluated':self.addresses[r['pair_id']+f"|{r['endpoint']}",r['region']]=(r['recipient_positions'],r['donor_positions'])
 def donor_key(self,key):
  pair,ep=key.rsplit('|',1);return pair+'|'+str(1-int(ep))
 def encoded_losses(self,qr,qd,key):
  c=self.cache[key];ids=c['ids'].cuda();nlp=c['native'].cuda().double().log_softmax(-1)
  # Same task retention definition as M60/M61; only the causal branch changes.
  out,_=trace.__wrapped__(self.model,ids,replace=lambda j,z,w:qr+self.mlp.Down_bias if j==1 else w);self.calls+=1;intact=out[0,-1];K=(nlp.exp()*(nlp-intact.double().log_softmax(-1))).sum();raw=[];balanced=[];branches=[]
  for t in c['targets']:
   pos,dp=self.addresses[key,t['region']];delta=torch.zeros_like(c['write'],device='cuda')
   if pos:
    index=torch.tensor(pos,device='cuda');donorindex=torch.tensor(dp,device='cuda');difference=qd.index_select(1,donorindex)-qr.index_select(1,index);delta=delta.index_copy(1,index,difference)
    out,_=trace.__wrapped__(self.model,ids,replace=lambda j,z,w:w+delta if j==1 else w);self.calls+=1;changed=out[0,-1]
   else:changed=c['native'].cuda()
   nr=t['native_changed'].cuda().double().log_softmax(-1)-nlp;error=changed.double().log_softmax(-1)-nlp-nr;R=(nlp.exp()*error.square()).sum();branches.append(dict(region=t['region'],delta=delta,changed=changed,R=R))
   if t['region']=='identity':assert float(R.detach())==0 and not bool(delta.count_nonzero())
   else:raw.append(R);balanced.append(R/self.den[c['family'],t['region']])
  return K,torch.stack(raw).mean(),torch.stack(balanced).mean(),intact,branches
