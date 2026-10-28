"""H724: multiseed full-span sparse rotations and signed block-TopK baselines."""
import json,time,copy
from pathlib import Path
import torch,numpy as np
from torch import nn
from scipy.optimize import linear_sum_assignment
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_verified_head_sparse'
class SignedBSF(nn.Module):
 def __init__(self,r):
  super().__init__();self.r=r;self.encoder=nn.Linear(128,128);self.decoder=nn.Parameter(torch.randn(128,128));self.bias=nn.Parameter(torch.zeros(128))
  with torch.no_grad():self.decoder.div_(self.decoder.norm(dim=1,keepdim=True));self.encoder.weight.copy_(self.decoder);self.encoder.bias.zero_()
 def encode(self,x):
  z=self.encoder(x).reshape(-1,128//self.r,self.r);ids=z.square().sum(-1).topk(8//self.r,dim=1).indices;mask=torch.zeros(z.shape[:2],device=x.device).scatter_(1,ids,1);return (z*mask[:,:,None]).flatten(1),mask
 def forward(self,x):
  z,_=self.encode(x);return z@self.decoder+self.bias
@torch.no_grad()
def mse(m,x):return float((m(x)-x).square().sum()/x.square().sum())
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_verified_head_pca_data.pt','fits':ART/'bilin18_verified_head_write_pca_fits.pt','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};cfg=dict(experiment='H724',hashes=hashes,seeds=[7241,7242,7243],steps=3000,batch=512,active_coordinates=8,decoder_width=128,design='Task-blind H722naturaltrain/validation splits, H723write-PCAcoordinates. Full128Dwhitening thenorthogonalrotation trainedsmoothL1 withmatrix-exp skewparameter,lr.003. SelectcheckpointbyvalidationL1every250steps; decodeunwhitened andTop8byindividualwrite-contributionnorm. Controlsidentity(PCAadaptiveTop8),randominitialrotations. SignedBSF useslinearencoder128->128withbias, decoder128x128unitnormatoms, outputbias; blockwidth1or4 selects8or2blocks=8scalars,lr.001, MSEin globallyscaledwrite-PCAcoordinates. SelectvalidationMSEevery250steps. Samebatchseed/order and3000steps, allseedsreported. Decoderblockorthogonalitynotimposed. Parametercounts/objectives/optimizationdiffer; not equalcapacityorcompute claim.',limitations='Finite-trainingpilot, notconvergence. ThisBSFbaselineisunit-column signedblockTopK, notGrassmannianBSF. Samewidth/activebudget butrotation16384trainableparameters(8128orthogonaldegrees) vsBSF33024. Validationreused. No output/task/manifoldclaimsfromreconstructionalone; originaltestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));data=torch.load(paths['data'],weights_only=False)['activations'];fits=torch.load(paths['fits'],weights_only=False)['fits'];dest=ART/'verified_head_sparse';dest.mkdir(exist_ok=True);records=[];states={};representations={}
 for site,s in fits.items():
  mu=s['mean'].cuda().float();E=s['write_pca']['encode'].cuda().float();sd=s['eigenvalues'].cuda().float().sqrt();train=(data[site][:2048].reshape(-1,128).cuda()-mu)@E;val=(data[site][2048:].reshape(-1,128).cuda()-mu)@E;tw=train/sd;vw=val/sd;scale=train.square().mean().sqrt();den=val.square().sum()
  @torch.no_grad()
  def rot_eval(Q):
   z=vw@Q;D=sd[:,None]*Q;norm=D.norm(dim=0);ids=(z.abs()*norm).topk(8,dim=1).indices;mask=torch.zeros_like(z).scatter_(1,ids,1);recon=(z*mask)@D.T;return dict(validation_write_relative_mse=float((recon-val).square().sum()/den),validation_smooth_l1=float(torch.sqrt(z.square()+1e-6).mean()),orthogonality_error=float((Q.T@Q-torch.eye(128,device='cuda')).abs().max()),active_scalars=float(mask.sum(-1).mean())),D.T.cpu(),mask.cpu()
  rec,D,mask=rot_eval(torch.eye(128,device='cuda'));records.append(dict(site=site,kind='adaptive_pca',seed=None,**rec))
  for seed in cfg['seeds']:
   for kind in ['rotation','bsf1','bsf4']:
    file=dest/f'{site}_{kind}_{seed}.pt'
    if file.exists():
     cached=torch.load(file,weights_only=False);assert cached['input_hashes']==hashes;records.extend(cached['records']);states[site,kind,seed]=cached['state'];representations[site,kind,seed]=cached['representation'];print('REUSED',site,kind,seed,flush=True);continue
    torch.manual_seed(seed);gen=torch.Generator(device='cuda').manual_seed(seed+100);history=[];local=[];best=float('inf')
    if kind=='rotation':
     base=torch.linalg.qr(torch.randn(128,128,device='cuda')).Q;param=nn.Parameter(torch.zeros(128,128,device='cuda'));opt=torch.optim.Adam([param],lr=.003)
     def rotation():return base@torch.matrix_exp(param-param.T)
     init,_,_=rot_eval(base);local.append(dict(site=site,kind='random_rotation',seed=seed,**init))
     for step in range(cfg['steps']+1):
      if step%250==0:
       with torch.no_grad():Q=rotation();score=float(torch.sqrt((vw@Q).square()+1e-6).mean());history.append(dict(step=step,validation_objective=score))
       if score<best:best=score;beststate=Q.detach().cpu().clone();beststep=step
      if step==cfg['steps']:break
      ii=torch.randint(len(tw),(512,),generator=gen,device='cuda');loss=torch.sqrt((tw[ii]@rotation()).square()+1e-6).mean();opt.zero_grad();loss.backward();opt.step()
     metrics,D,mask=rot_eval(beststate.cuda());assert metrics['orthogonality_error']<1e-4;state=dict(rotation=beststate,sd=sd.cpu(),input_scale=float(scale));r=1
    else:
     r=int(kind[-1]);m=SignedBSF(r).cuda();opt=torch.optim.Adam(m.parameters(),lr=.001);x=train/scale;v=val/scale
     for step in range(cfg['steps']+1):
      if step%250==0:
       score=mse(m,v);history.append(dict(step=step,validation_objective=score))
       if score<best:best=score;beststate={k:t.detach().cpu().clone() for k,t in m.state_dict().items()};beststep=step
      if step==cfg['steps']:break
      ii=torch.randint(len(x),(512,),generator=gen,device='cuda');loss=(m(x[ii])-x[ii]).square().mean();opt.zero_grad();loss.backward();opt.step()
      with torch.no_grad():m.decoder.div_(m.decoder.norm(dim=1,keepdim=True).clamp_min(1e-10))
     m.load_state_dict(beststate)
     with torch.no_grad():z,mask=m.encode(v);metrics=dict(validation_write_relative_mse=mse(m,v),active_scalars=float(mask.sum(-1).mean())*r,dead_blocks=int((mask.sum(0)==0).sum()),decoder_norm_max_error=float((m.decoder.norm(dim=1)-1).abs().max()));D=m.decoder.detach().cpu();mask=mask.cpu()
     state=dict(model=beststate,block_dimension=r,input_scale=float(scale));assert metrics['decoder_norm_max_error']<1e-5
    assert metrics['active_scalars']==8;rec=dict(site=site,kind=kind,seed=seed,selected_step=beststep,history=history,**metrics);local.append(rec);representation=dict(decoder=D,mask=mask,block_dimension=r);cached=dict(input_hashes=hashes,records=local,state=state,representation=representation);torch.save(cached,file);records.extend(local);states[site,kind,seed]=state;representations[site,kind,seed]=representation;print('FIT',site,kind,seed,'step',beststep,'MSE',metrics['validation_write_relative_mse'],flush=True)
 stability=[]
 for site in fits:
  for kind in ['rotation','bsf1','bsf4']:
   seeds=cfg['seeds']
   for i in range(3):
    for k in range(i+1,3):
     a=representations[site,kind,seeds[i]];b=representations[site,kind,seeds[k]];r=a['block_dimension'];Da=a['decoder'].double();Db=b['decoder'].double()
     if r==1:Da=Fnormalize(Da);Db=Fnormalize(Db);sim=(Da@Db.T).abs()
     else:
      qa=torch.linalg.qr(Da.reshape(-1,r,128).transpose(1,2)).Q;qb=torch.linalg.qr(Db.reshape(-1,r,128).transpose(1,2)).Q;cross=torch.einsum('air,bis->abrs',qa,qb);sim=cross.square().sum((-1,-2))/r
     ia,ib=linear_sum_assignment(-sim.numpy());ma=a['mask'][:,ia].bool();mb=b['mask'][:,ib].bool();jac=(ma&mb).sum(0)/(ma|mb).sum(0).clamp_min(1);stability.append(dict(site=site,kind=kind,seeds=[seeds[i],seeds[k]],mean_matched_decoder_similarity=float(sim[ia,ib].mean()),mean_matched_support_jaccard=float(jac.mean()),similarity_metric='abs_cosine' if r==1 else 'mean_squared_principal_cosine'))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['validation_write_relative_mse']) for r in records),every_active_budget_eight=all(r['active_scalars']==8 for r in records),whole_model_forwards=0,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and audits['every_active_budget_eight'];torch.save(dict(states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,stability=stability,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
def Fnormalize(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-20)
if __name__=='__main__':main()
