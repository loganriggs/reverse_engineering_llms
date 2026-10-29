"""Three-corner causal objective for executable input-to-write predictions."""
import json
import torch
from encoded_change_loss import EncodedExperiment
from inspect_model import trace
from fit_pca_encoder import sha,V
class InputCausalExperiment(EncodedExperiment):
 def __init__(self):
  super().__init__();r=json.loads((V/'input_training_inventory.json').read_text());audit=json.loads((V/'input_training_inventory_audit.json').read_text());assert audit['status']=='passed' and audit['inventory_sha256']==sha(V/'input_training_inventory.json');assert sha(r['path'])==r['sha256'];self.input_cache=torch.load(r['path'],weights_only=True)['inputs'];assert set(self.input_cache)==set(self.cache)
 def joint_losses(self,qr,qd,key):
  c=self.cache[key];ids=c['ids'].cuda();nlp=c['native'].cuda().double().log_softmax(-1)
  def run(write):
   out,_=trace.__wrapped__(self.model,ids,replace=lambda j,z,w:write+self.mlp.Down_bias if j==1 else w);self.calls+=1;return out[0,-1]
  intact=run(qr);ilp=intact.double().log_softmax(-1);K=(nlp.exp()*(nlp-ilp)).sum();branches=[];balanced=[]
  for t in c['targets']:
   pos,dp=self.addresses[key,t['region']];nd=t['delta'].cuda();rd=torch.zeros_like(qr)
   if pos:
    index=torch.tensor(pos,device=qr.device);donorindex=torch.tensor(dp,device=qd.device);rd=rd.index_copy(1,index,qd.index_select(1,donorindex)-qr.index_select(1,index))
   teacher=t['native_changed'].cuda().double().log_softmax(-1)-nlp
   for corner,background,delta,base in [('native_reconstructed',c['write'].cuda(),rd,c['native'].cuda()),('reconstructed_native',qr,nd,intact),('reconstructed_reconstructed',qr,rd,intact)]:
    changed=base if t['region']=='identity' else run(background+delta)
    response=changed.double().log_softmax(-1)-base.double().log_softmax(-1);R=(nlp.exp()*(response-teacher).square()).sum();branches.append(dict(region=t['region'],corner=corner,base=base,changed=changed,R=R,delta=delta))
    if t['region']=='identity':assert float(R.detach())==0 and not bool(delta.count_nonzero())
    else:balanced.append(R/self.den[c['family'],t['region']])
  return K,torch.stack(balanced).mean(),intact,branches
