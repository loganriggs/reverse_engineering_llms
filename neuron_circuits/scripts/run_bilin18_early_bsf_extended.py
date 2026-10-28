"""H683: longer fixed-compute training with pilot replay and resumable checkpoints."""
import json,time,hashlib,copy
from pathlib import Path
import torch
from torch import nn
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_early_bsf_extended'
class BSF(nn.Module):
 def __init__(self,d,r):
  super().__init__();self.r=r;self.groups=2304//r;self.k=32//r;self.encoder=nn.Linear(d,2304);self.decoder=nn.Parameter(torch.randn(2304,d));self.bias=nn.Parameter(torch.zeros(d))
  with torch.no_grad():self.decoder.div_(self.decoder.norm(dim=1,keepdim=True));self.encoder.weight.copy_(self.decoder);self.encoder.bias.zero_()
 def encode(self,x):
  z=self.encoder(x).reshape(-1,self.groups,self.r);ids=z.square().sum(-1).topk(self.k,dim=1).indices;mask=torch.zeros(z.shape[:2],device=x.device).scatter_(1,ids,1);return z*mask[:,:,None],mask
 def forward(self,x):
  z,_=self.encode(x);return z.flatten(1)@self.decoder+self.bias
@torch.no_grad()
def evaluate(m,x):
 err=0.;den=0.
 for b in x.split(512):err+=float((m(b)-b).square().sum());den+=float(b.square().sum())
 return err/den
@torch.no_grad()
def geometry(m,x):
 counts=torch.zeros(m.groups,device=x.device);sums=torch.zeros(m.groups,m.r,device=x.device);second=torch.zeros(m.groups,m.r,m.r,device=x.device)
 for b in x.split(512):
  z,mask=m.encode(b);counts+=mask.sum(0);sums+=z.sum(0);second+=torch.einsum('bgr,bgs->grs',z,z)
 mu=sums/counts.clamp_min(1)[:,None];cov=second/counts.clamp_min(1)[:,None,None]-mu[:,:,None]*mu[:,None,:];eig=torch.linalg.eigvalsh(cov.double()).clamp_min(0);rank=eig.sum(-1).square()/eig.square().sum(-1).clamp_min(1e-20);eligible=counts>=32
 return dict(dead_blocks=int((counts==0).sum()),blocks=m.groups,blocks_with_32_examples=int(eligible.sum()),median_effective_rank=float(rank[eligible].median()) if eligible.any() else None,mean_active_blocks=float(counts.sum()/len(x)),occupancy=counts.cpu().tolist(),effective_rank=rank.cpu().tolist())
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('high');start=time.time();paths={'data':ART/'bilin18_early_bsf_data_captures.pt','manifest':ART/'bilin18_early_bsf_data_manifest.json','pilot_results':ART/'bilin18_early_bsf_pilot.json','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};cfg=dict(experiment='H683',hashes=hashes,sites=['value0','residual0','residual1','residual2'],block_dimensions=[1,4,16],seeds=[681,682,683],decoder_width=2304,active_scalar_budget=32,steps=5000,batch_size=512,optimizer='Adam lr0.0003',validation_interval=250,design='Signed linear encoder with block-L2 TopK, linear decoder with unit column norms. Train mean/global RMS scaling only. Same seeded minibatches across block widths. Validation checkpoint selection; test untouched. Fixed-compute pilot, not convergence claim.',limitations='Rank1 is signed absolute-TopK sparse dictionary baseline, not ReLU SAE. Fixed scalar budget changes active-block count; no single fairness criterion. No dead-block resampling or auxiliary loss. Finite training may favor easier optimizers.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));data=torch.load(paths['data'],weights_only=False);records=[];pilot_replays=[];dest=ART/'early_bsf_extended';dest.mkdir(exist_ok=True)
 for site in cfg['sites']:
  raw=data['activations'][site];train=raw[:2048].reshape(-1,1152).cuda();val=raw[2048:2304].reshape(-1,1152).cuda();mu=train.mean(0);scale=(train-mu).square().mean().sqrt();train=(train-mu)/scale;val=(val-mu)/scale
  for rank in cfg['block_dimensions']:
   for seed in cfg['seeds']:
    torch.manual_seed(seed);m=BSF(1152,rank).cuda();opt=torch.optim.Adam(m.parameters(),lr=3e-4);gen=torch.Generator(device='cuda').manual_seed(seed+10000);best=float('inf');state=None;history=[];begin=time.time()
    for step in range(1,5001):
     ids=torch.randint(len(train),(512,),generator=gen,device='cuda');x=train[ids];loss=(m(x)-x).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
     with torch.no_grad():m.decoder.div_(m.decoder.norm(dim=1,keepdim=True).clamp_min(1e-8))
     if step%250==0:
      err=evaluate(m,val);history.append(dict(step=step,train_batch_loss=float(loss.detach()),validation_relative_mse=err))
      if err<best:best=err;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};beststep=step
     if step==1000:
      reference=torch.load(ART/'early_bsf_pilot'/f'{site}_r{rank}_seed{seed}.pt',weights_only=False);diff=max(float((m.state_dict()[k].cpu()-v).abs().max()) for k,v in reference['state_dict'].items());assert reference['record']['best_step']==1000 and diff<1e-5;pilot_replays.append(diff)
    finalstate={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    def cpu_tree(x):
     if torch.is_tensor(x):return x.detach().cpu().clone()
     if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
     if isinstance(x,list):return [cpu_tree(v) for v in x]
     return x
    resume=dict(state_dict=finalstate,optimizer=cpu_tree(opt.state_dict()),generator_state=gen.get_state().cpu(),step=5000)
    m.load_state_dict(state);geo=geometry(m,val);rec=dict(site=site,block_dimension=rank,seed=seed,best_step=beststep,validation_relative_mse=best,training_relative_mse=evaluate(m,train),history=history,geometry=geo,elapsed_seconds=time.time()-begin);assert torch.isfinite(torch.tensor(best))
    torch.save(dict(resume=resume,state_dict=state,mean=mu.cpu(),scale=scale.cpu(),config=cfg,record=rec),dest/f'{site}_r{rank}_seed{seed}.pt');records.append(rec);(ART/f'{STEM}.partial.json').write_text(json.dumps(dict(status='running',config=cfg,records=records),indent=2));print('DONE',site,rank,seed,'val',round(best,5),'dead',geo['dead_blocks'],'rank',geo['median_effective_rank'],'seconds',round(time.time()-begin,1),flush=True)
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},pilot_weight_replay_max_abs=max(pilot_replays),models=len(records),test_evaluations=0,finite=all(torch.isfinite(torch.tensor(r['validation_relative_mse'])).item() for r in records),whole_model_forwards=0);assert len(records)==36;(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print('AUDITS',audits,flush=True)
if __name__=='__main__':main()
