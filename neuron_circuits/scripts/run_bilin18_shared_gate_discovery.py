"""H692: label-free gate fitting conditional on a repaired dictionary."""
import torch,json,time,hashlib,itertools,copy
from pathlib import Path
from feature_structures.models import SharedGroups,binary_gate
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_shared_gate_discovery'
class EnergyGates(torch.nn.Module):
 def __init__(self):
  super().__init__();self.projection=torch.nn.Linear(16,32,bias=False);self.readout=torch.nn.Linear(32,4)
 def forward(self,x):return self.readout(self.projection(x).square())
def main():
 torch.set_num_threads(2);start=time.time();paths={'teacher':ART/'feature_structure_synthetic_captures.pt','fit':ART/'bilin18_shared_discrete_search_captures.pt','implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};teacher=torch.load(paths['teacher'],weights_only=False);B,M=teacher['true_dictionary'],teacher['true_membership'];fit=torch.load(paths['fit'],weights_only=False)['states'][11];model=SharedGroups(16,8,4,None,3)
 with torch.no_grad():
  for target,p in zip([model.dictionary,model.coordinate_encoder.weight,model.coordinate_encoder.bias,model.bias],fit['parameters']):target.copy_(p[0])
  model.membership_logits.copy_(fit['membership'][0]*4-2)
 for p in model.parameters():p.requires_grad_(False)
 trainsets=[s for n in [1,2] for s in itertools.combinations(range(4),n)];heldsets=[s for n in [3,4] for s in itertools.combinations(range(4),n)];gen=torch.Generator().manual_seed(692)
 def sample(sets,n):
  g=torch.zeros(n,4);which=torch.randint(len(sets),(n,),generator=gen)
  for i,k in enumerate(which):g[i,list(sets[int(k)])]=1
  a=torch.randn(n,8,generator=gen);return (a*(g@M>0))@B.T,g
 tx,tg=sample(trainsets,4096);vx,vg=sample(trainsets,2048);hx,hg=sample(heldsets,2048);scale=tx.square().mean();cost=.01
 cfg=dict(experiment='H692',hashes=hashes,steps=3000,lr=.003,batch=128,group_cost=cost,design='Freeze H691 seed11 dictionary, coordinate encoder and memberships. Fit linear or even quadratic-energy gate network from scratch via hard-forward straight-through reconstruction plus group count, without gate labels. Variable-count threshold gates. Select every100steps on independent singleton/pair validation objective, test higher-order combinations only afterward. Three gate seeds.',limitations='Conditional gate discovery, not joint unsupervised dictionary/group discovery. Different gate parameter counts. Same teacher reused. No hierarchy or LLM validation yet.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];saved={};fixed_before=copy.deepcopy(model.state_dict())
 def metrics(net,x,g):
  with torch.no_grad():
   logits=net(x);s=binary_gate(logits);out=model(x,gates=s);oracle=model(x,gates=g);allopen=model(x,gates=torch.ones_like(g));truth=(g@fit['membership'][0]>0).float()
   mse=float((out['reconstruction']-x).square().mean()/x.square().mean());return dict(relative_mse=mse,objective=float((out['reconstruction']-x).square().mean()/scale+cost*s.sum(-1).mean()),mean_groups=float(s.sum(-1).mean()),mean_atoms=float(out['support'].sum(-1).mean()),gate_entry_accuracy=float((s==g).float().mean()),gate_exact_accuracy=float((s==g).all(-1).float().mean()),support_exact_accuracy=float((out['support']==truth).all(-1).float().mean()),sign_flip_gate_agreement=float((s==binary_gate(net(-x))).float().mean()),teacher_gate_relative_mse=float((oracle['reconstruction']-x).square().sum()/x.square().sum()),all_open_relative_mse=float((allopen['reconstruction']-x).square().sum()/x.square().sum()))
 for seed in [21,22,23]:
  for kind in ['linear','energy']:
   torch.manual_seed(seed);net=torch.nn.Linear(16,4) if kind=='linear' else EnergyGates();opt=torch.optim.Adam(net.parameters(),lr=.003);batchgen=torch.Generator().manual_seed(seed+100);best=float('inf');beststate=None;history=[]
   for step in range(3001):
    if step%100==0:
     val=metrics(net,vx,vg);history.append(dict(step=step,**val))
     if val['objective']<best:best=val['objective'];beststate=copy.deepcopy(net.state_dict());beststep=step
    if step==3000:break
    ii=torch.randint(len(tx),(128,),generator=batchgen);s=binary_gate(net(tx[ii]));out=model(tx[ii],gates=s);loss=(out['reconstruction']-tx[ii]).square().mean()/scale+cost*s.sum(-1).mean();opt.zero_grad();loss.backward();opt.step()
   net.load_state_dict(beststate);rec=dict(seed=seed,kind=kind,parameters=sum(p.numel() for p in net.parameters()),selected_step=beststep,train=metrics(net,tx,tg),validation=metrics(net,vx,vg),heldout=metrics(net,hx,hg),history=history);records.append(rec);saved[kind,seed]=beststate;print({k:rec[k] for k in ['seed','kind','selected_step','heldout']},flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},frozen_dictionary_unchanged=all(torch.equal(p,fixed_before[k]) for k,p in model.state_dict().items()),finite=all(torch.isfinite(torch.tensor(r['heldout']['relative_mse'])).item() for r in records),whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and audits['frozen_dictionary_unchanged'];torch.save(dict(states=saved),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
