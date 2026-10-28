"""H836 exact preMLP-site signed scalar/block and shared-union pilots."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from feature_structures.models import SharedGroups
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_early_local_geometry import sha
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
S='bilin18_balanced_shared_featurizers'
class Sparse(nn.Module):
 def __init__(self,r,budget=64):
  super().__init__();self.r=r;self.budget=budget;self.encoder=nn.Linear(1152,512);self.decoder=nn.Parameter(torch.randn(512,1152));self.bias=nn.Parameter(torch.zeros(1152))
  with torch.no_grad():self.decoder.div_(self.decoder.norm(dim=1,keepdim=True));self.encoder.weight.copy_(self.decoder);self.encoder.bias.zero_()
 def forward(self,x):
  a=self.encoder(x).reshape(-1,512//self.r,self.r);g=a.square().sum(-1).topk(self.budget//self.r,dim=1).indices;s=torch.zeros(a.shape[:2],device=x.device).scatter_(1,g,1);code=(a*s[:,:,None]).flatten(1);return {'reconstruction':code@self.decoder+self.bias,'code':code,'support':s.repeat_interleave(self.r,1)}
class BalancedShared(SharedGroups):
 def __init__(self,seed,fixed):
  super().__init__(1152,512,128,8,8)
  gen=torch.Generator().manual_seed(seed+836000);M=torch.zeros(128,512)
  for half in range(2):
   perm=torch.randperm(512,generator=gen).reshape(64,8)
   M[torch.arange(64)[:,None]+half*64,perm]=1
  assert bool((M.sum(0)==2).all()) and bool((M.sum(1)==8).all())
  self.register_buffer('initial_membership',M);self.fixed=fixed
  with torch.no_grad():self.membership_logits.copy_(.1*M-.05)
  if fixed:self.membership_logits.requires_grad_(False)
 def forward(self,x):return super().forward(x,membership=self.initial_membership if self.fixed else None)
@torch.no_grad()
def score(m,x):
 return sum(float((m(b)['reconstruction']-b).square().sum()) for b in x.split(128))/float(x.square().sum())
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_operator_state_prediction_captures.pt','implementation':ROOT/'scripts/feature_structures/models.py','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H836','hashes':hashes,'sites':[1,2],'seeds':[8351,8352],'steps':2400,'batch':128,'learning_rate':.001,'selection':'Development residual reconstruction MSE every200 steps incl0; retain both seeds.2400-step optimization comparison, not converged architecture test.','data':'H83232768 training states/site;512 development occurrences, no H834 confirmation use. Token residuals of normalized state; common train global RMS scaling.','models':'Signed scalarTop48/Top64; block8 Top8; shared512 atoms/128 groups/Top8 groups/8 memberships. Two random partitions give every atom degree2 initially; fixed balanced vs learned membership. Separate group encoder and exact union. Shared actual active atoms<=64 and reported. Fixed budgets, no learned variable L0 penalty.','fairness':'Same512 atoms,48/64 scalar controls; shared64 max union support,1024 membership slots, full initial coverage. Same initialized coordinate dictionary and batches; seeds8351/8352 retained to extend prior baseline trajectories. Shared has extra gate/membership parameters and may fewer active coordinates. Unit atoms; not Grassmannian within-block orthogonality. Same token/radius access, full original MLP weights.','evaluation':'Local derivative fidelity on H832 development probes. All models predicted shape=token mean+decoded residual, then normalize/restore actual radius. No unseen routing in this pilot; same fallback mean for all.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['data'],weights_only=False);dest=ART/'balanced_shared_featurizers';dest.mkdir(exist_ok=True);records=[];states={};predictions={};eps=torch.finfo(torch.float32).eps;decodeerr=[];unionerr=[]
 for l in [1,2]:
  f=data['fits'][l];x=data['captures']['train',l].double();u=x/(x.square().mean(1,keepdim=True)+eps).sqrt();lookup={int(t):i for i,t in enumerate(f['token_ids'])};idx=torch.tensor([lookup[int(t)] for t in data['train_token_ids']]);res=u-f['token_shape_mean'][idx];scale=res.square().mean().sqrt().float();tr=(res/scale).float().cuda();val=data['captures']['validation',l].double();rad=(val.square().mean(1,keepdim=True)+eps).sqrt();vu=val/rad;tm=torch.stack([f['token_shape_mean'][lookup[int(t)]] if int(t) in lookup else f['shape_mean'] for t in data['validation_token_ids']]);va=((vu-tm)/scale).float().cuda()
  for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
   for seed in cfg['seeds']:
    torch.manual_seed(seed);init=Sparse(1).cuda();initial={k:v.detach().clone() for k,v in init.state_dict().items()};del init
    if kind.startswith('shared'):
     m=BalancedShared(seed,kind=='shared_fixed').cuda()
     with torch.no_grad():m.dictionary.copy_(initial['decoder'].T);m.coordinate_encoder.weight.copy_(initial['encoder.weight']);m.coordinate_encoder.bias.zero_()
    else:m=Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(initial)
    gen=torch.Generator(device='cuda').manual_seed(seed);opt=torch.optim.Adam(m.parameters(),lr=.001);history=[];best=float('inf');begin=time.time()
    for step in range(2401):
     if step%200==0:
      sc=score(m,va);history.append({'step':step,'validation_residual_MSE':sc})
      if sc<best:best=sc;beststep=step;beststate={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
     if step==2400:break
     inds=torch.randint(len(tr),(128,),generator=gen,device='cuda');result=m(tr[inds]);loss=(result['reconstruction']-tr[inds]).square().mean();opt.zero_grad();loss.backward();assert torch.isfinite(loss) and all(p.grad is None or bool(p.grad.isfinite().all()) for p in m.parameters());opt.step()
     if not kind.startswith('shared'):
      with torch.no_grad():m.decoder.div_(m.decoder.norm(dim=1,keepdim=True).clamp_min(1e-10))
    m.load_state_dict(beststate)
    with torch.no_grad():
     outs=[m(b) for b in va.split(128)];recon=torch.cat([o['reconstruction'] for o in outs]);support=torch.cat([o['support'] for o in outs]);active=support.sum(1);assert active.max()<=64 and active.min()>=8
     if kind.startswith('shared'):
      D=m.dictionary_matrix();normerr=float((D.norm(dim=0)-1).abs().max())
      for o in outs:
       decodeerr.append(float((o['reconstruction']-o['code']@D.T-m.bias).abs().max()));explicit=(o['gates'].detach()@o['membership'].detach()>0).float();unionerr.append(float((o['support']-explicit).abs().max()))
      membership=int(outs[0]['membership'].sum());covered=int((outs[0]['membership'].sum(0)>0).sum())
      if kind=='shared_fixed':assert bool((outs[0]['membership'].sum(0)==2).all()) and covered==512
     else:normerr=float((m.decoder.norm(dim=1)-1).abs().max());membership=512;covered=512;decodeerr.extend(float((o['reconstruction']-o['code']@m.decoder-m.bias).abs().max()) for o in outs)
     assert normerr<1e-5;shape=tm+recon.cpu().double()*scale;raw=shape/shape.square().mean(1,keepdim=True).sqrt().clamp_min(1e-30)*(rad.square()-eps).sqrt();predictions[l,kind,seed]=raw
     rec={'layer':l,'kind':kind,'seed':seed,'selected_step':beststep,'history':history,'validation_residual_MSE':best,'train_residual_MSE':score(m,tr),'active_atoms_mean':float(active.mean()),'active_atoms_min':float(active.min()),'active_atoms_max':float(active.max()),'memberships':membership,'atoms_in_any_group':covered,'parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'fixed_membership':kind=='shared_fixed','decoder_norm_error':normerr,'elapsed':time.time()-begin};records.append(rec);state={'state_dict':beststate,'scale':float(scale),'kind':kind,'record':rec};states[l,kind,seed]=state;torch.save(state,dest/f'layer{l}_{kind}_{seed}.pt');print('FIT',l,kind,seed,best,rec['active_atoms_mean'],flush=True)
    del m,opt
 model=load_full_model('cuda');evals=[]
 with torch.no_grad():
  for l in [1,2]:
   mlp=model.blocks[l].mlp;assert not mlp.config.gated;L,R,D=[getattr(mlp,n).weight.double() for n in ['Left','Right','Down']];base=data['captures']['validation',l].double().cuda()
   def response(b,v):
    r=(b.square().mean(1,keepdim=True)+eps).sqrt();s=b/r;q=v/r-b*(b*v).mean(1,keepdim=True)/r.pow(3);return F.linear(F.linear(s,L)*F.linear(q,R)+F.linear(q,L)*F.linear(s,R),D)
   for di,v in enumerate(data['fits'][l]['directions'].cuda()):
    truth=response(base,v);den=truth.square().sum(1)
    for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
     for seed in cfg['seeds']:
      y=response(predictions[l,kind,seed].cuda(),v);err=(y-truth).square().sum(1)
      for i in range(512):evals.append({'layer':l,'kind':kind,'seed':seed,'document':data['validation_documents'][i],'probe':di,'squared_error':float(err[i]),'truth_squared_norm':float(den[i])})
 summary=[]
 for rec in records:
  rr=[r for r in evals if (r['layer'],r['kind'],r['seed'])==(rec['layer'],rec['kind'],rec['seed'])];summary.append({'layer':rec['layer'],'kind':rec['kind'],'seed':rec['seed'],'relative_RMSE':float(np.sqrt(sum(r['squared_error'] for r in rr)/sum(r['truth_squared_norm'] for r in rr)))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'decode_formula_max_abs':max(decodeerr),'union_exact_max_abs':max(unionerr),'whole_model_forwards':0,'test_evaluations':0,'all_fits_completed':len(records)==20};assert audits['hashes_unchanged'] and max(unionerr)==0 and max(decodeerr)<1e-5;torch.save({'states':states,'predicted_raw_states':predictions},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'evaluation_records':evals,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()
