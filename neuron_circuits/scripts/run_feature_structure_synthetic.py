"""Initial label-assisted synthetic recovery; not unsupervised structure discovery."""
import json,time,itertools,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from feature_structures.models import SharedGroups,ComputationDAG
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='feature_structure_synthetic'
class SeparateBlocks(nn.Module):
 def __init__(self):
  super().__init__();self.enc=nn.Linear(16,12);self.D=nn.Parameter(torch.randn(4,3,16));self.bias=nn.Parameter(torch.zeros(16))
 def forward(self,x,gates):
  a=self.enc(x).view(-1,4,3)*gates[:,:,None];return torch.einsum('bgr,grd->bd',a,F.normalize(self.D,dim=-1))+self.bias

def main():
 torch.set_num_threads(6);torch.manual_seed(687);start=time.time();M=torch.tensor([[1,1,1,0,0,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,0,1,1,1,0],[1,0,0,0,0,0,1,1]],dtype=torch.float32);B=torch.linalg.qr(torch.randn(16,8)).Q;trainsets=[(0,),(1,),(2,),(3,),(0,1),(1,2),(2,3)];testsets=[s for n in range(2,5) for s in itertools.combinations(range(4),n) if s not in trainsets]
 def dataset(sets,n):
  g=torch.zeros(n,4);which=torch.randint(len(sets),(n,))
  for i,k in enumerate(which):g[i,list(sets[int(k)])]=1
  a=torch.randn(n,8);q=(g@M>0).float();return (a*q)@B.T,g
 trainx,traing=dataset(trainsets,4096);testx,testg=dataset(testsets,2048);records=[];states={};cfg=dict(experiment='H687',design='Known synthetic group gates supplied to all variants.16D data,8orthonormal true atoms,4overlapping groups of3. Train singleton and adjacentpair groups; hold otherpairs/triples/allgroups out. Compare fixed true membership, learned Top3 membership and separate3Dblocks. Three seeds,3000Adam steps. Initial plumbing/reuse benchmark, NOT unsupervised or parameter-matched superiority.',train_supports=trainsets,test_supports=testsets,steps=3000,seeds=[11,12,13]);(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2))
 for seed in cfg['seeds']:
  for kind in ['shared_known_membership','shared_learned_membership','separate_blocks']:
   torch.manual_seed(seed);m=SeparateBlocks() if kind=='separate_blocks' else SharedGroups(16,8,4,2,3,initial_membership=M if kind=='shared_known_membership' else None);opt=torch.optim.Adam(m.parameters(),lr=.003);gen=torch.Generator().manual_seed(seed+100);hist=[]
   for step in range(3000):
    ix=torch.randint(len(trainx),(128,),generator=gen);x,g=trainx[ix],traing[ix]
    out=m(x,g) if kind=='separate_blocks' else m(x,gates=g,membership=M if kind=='shared_known_membership' else None)['reconstruction'];loss=(out-x).square().mean();opt.zero_grad();loss.backward();opt.step()
    if step%500==0:hist.append(float(loss.detach()))
   with torch.no_grad():
    def eval(x,g):return m(x,g) if kind=='separate_blocks' else m(x,gates=g,membership=M if kind=='shared_known_membership' else None)['reconstruction']
    trainerr=float((eval(trainx,traing)-trainx).square().sum()/trainx.square().sum());testerr=float((eval(testx,testg)-testx).square().sum()/testx.square().sum());rec=dict(kind=kind,seed=seed,parameters=sum(p.numel() for p in m.parameters()),train_relative_mse=trainerr,heldout_combination_relative_mse=testerr,loss_history=hist)
    if kind!='separate_blocks':
     result=m(testx,gates=testg,membership=M if kind=='shared_known_membership' else None);rec['mean_active_atoms']=float(result['support'].sum(-1).mean());rec['membership']=result['membership'].tolist()
    else:rec['mean_active_atoms']=float(testg.sum(-1).mean()*3)
    records.append(rec);states[kind,seed]=m.state_dict();print(rec['kind'],seed,trainerr,testerr,flush=True)
 # Shared nonlinear computation benchmark: unrestricted learned weights on a fixed ordered graph.
 dagresults=[]
 for seed in [11,12,13]:
  torch.manual_seed(seed);m=ComputationDAG(2,3,[(),(),(0,1),(2,),(2,)],products=[(),(),((0,1),),(),()]);opt=torch.optim.Adam(m.parameters(),lr=.003);raw=torch.rand(8192,2)*4-2;x=raw[~((raw[:,0]>0)&(raw[:,1]>0))];test=torch.rand(2048,2)*2
  def target(x):u=x[:,0]*x[:,1];return torch.stack([2*u,3*u,x[:,0]],-1)
  y=target(x)
  for step in range(3000):
   ix=torch.randint(len(x),(128,));out=m(x[ix],gates=torch.ones(128,5))['reconstruction'];loss=F.mse_loss(out,y[ix]);opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():
   pred=m(test,gates=torch.ones(len(test),5))['reconstruction'];removed=m(test,gates=torch.ones(len(test),5),remove_nodes=(2,))['reconstruction'];truth=target(test);dagresults.append(dict(seed=seed,test_relative_mse=float((pred-truth).square().sum()/truth.square().sum()),node2_removal_mse_by_output=(removed-truth).square().mean(0).tolist(),baseline_mse_by_output=(pred-truth).square().mean(0).tolist()));states['dag',seed]=m.state_dict();print('DAG',dagresults[-1],flush=True)
 torch.save(dict(states=states,true_dictionary=B,true_membership=M),ART/f'{STEM}_captures.pt');audits=dict(finite=all(torch.isfinite(torch.tensor(r['heldout_combination_relative_mse'])) for r in records),whole_model_forwards=0,test_is_synthetic=True);audits['finite']=bool(audits['finite']);(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,dag_results=dagresults,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('COMPLETE',flush=True)
if __name__=='__main__':main()
