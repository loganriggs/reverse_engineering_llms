"""H706: bounded model-activation sparse rotations with same-span controls."""
import torch,json,time,hashlib
from pathlib import Path
from run_bilin18_diverse_causal_tasks import ROOT,ART
STEM='bilin18_early_sparse_rotations'
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_early_bsf_data_captures.pt','spectra':ART/'bilin18_early_rotation_spectra_captures.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};cfg=dict(experiment='H706',hashes=hashes,rank=256,steps=2000,lr=.003,batch=512,seeds=[7061,7062,7063],sites=['value0','residual1'],design='Train-only H705 mean/eigensystem. Optimize orthogonal256Drotation of whitened training projections for smoothL1sqrt(z²+1e-6); select validationL1 every200steps. PCA/random same-span controls. HardTopK by individual decoded-contribution norm, budgets8/16/32/64/128/256. Report projection floor plus within-span loss. No groups fitted yet.',limitations='Undercomplete256Dspan, notfairbudgetmatchto2304atomBSF. Validation reused exploratory. TopK ranking heuristic is not optimal reconstruction in a nonorthogonal dictionary. Finite training, no semantic/manifold claims.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));data=torch.load(paths['data'],weights_only=False);spectra=torch.load(paths['spectra'],weights_only=False)['states'];records=[];states={};dest=ART/'early_sparse_rotations';dest.mkdir(exist_ok=True)
 for site in cfg['sites']:
  spec=spectra[site];mu=spec['mean'].cuda().float();V=spec['eigenvectors'][:,:256].cuda().float();sd=spec['eigenvalues'][:256].cuda().float().sqrt();raw=data['activations'][site];train=raw[:2048].reshape(-1,1152).cuda()-mu;val=raw[2048:2304].reshape(-1,1152).cuda()-mu;tp=train@V;vp=val@V;tw=tp/sd;vw=vp/sd;den=val.square().sum();floor=float((den-vp.square().sum())/den);del train
  @torch.no_grad()
  def evaluate(Q):
   codes=vw@Q;norms=(sd[:,None]*Q).norm(dim=0);order=(codes.abs()*norms).argsort(-1,descending=True);budget=[]
   for k in [8,16,32,64,128,256]:
    mask=torch.zeros_like(codes).scatter_(1,order[:,:k],1);recon=((codes*mask)@Q.T)*sd;extra=float((recon-vp).square().sum()/den);budget.append(dict(active_coordinates=k,relative_mse=floor+extra,projection_floor=floor,within_span_extra=extra))
   return dict(validation_smooth_l1=float(torch.sqrt(codes.square()+1e-6).mean()),budgets=budget,orthogonality_error=float((Q.T@Q-torch.eye(256,device='cuda')).abs().max()))
  identity=torch.eye(256,device='cuda');records.append(dict(site=site,kind='pca',seed=None,selected_step=0,**evaluate(identity)));states[site,'pca',None]=identity.cpu()
  for seed in cfg['seeds']:
   checkpoint=dest/f'{site}_{seed}.pt'
   if checkpoint.exists():
    cached=torch.load(checkpoint,weights_only=False);records.extend(cached['records']);states.update(cached['states']);print('REUSED',site,seed,flush=True);continue
   torch.manual_seed(seed);base=torch.linalg.qr(torch.randn(256,256,device='cuda')).Q;param=torch.nn.Parameter(torch.zeros(256,256,device='cuda'));opt=torch.optim.Adam([param],lr=.003);gen=torch.Generator(device='cuda').manual_seed(seed+100);best=float('inf');history=[]
   def rotation():return base@torch.matrix_exp(param-param.T)
   initial=evaluate(base);records.append(dict(site=site,kind='random',seed=seed,selected_step=0,**initial));states[site,'random',seed]=base.cpu()
   for step in range(2001):
    if step%200==0:
     with torch.no_grad():Q=rotation();score=float(torch.sqrt((vw@Q).square()+1e-6).mean());history.append(dict(step=step,validation_smooth_l1=score))
     if score<best:best=score;bestQ=Q.detach().clone();beststep=step
    if step==2000:break
    ii=torch.randint(len(tw),(512,),generator=gen,device='cuda');loss=torch.sqrt((tw[ii]@rotation()).square()+1e-6).mean();opt.zero_grad();loss.backward();opt.step()
   metrics=evaluate(bestQ);assert metrics['orthogonality_error']<1e-4;rec=dict(site=site,kind='learned',seed=seed,selected_step=beststep,history=history,**metrics);records.append(rec);states[site,'learned',seed]=bestQ.cpu();torch.save(dict(records=records[-2:],states={key:states[key] for key in [(site,'random',seed),(site,'learned',seed)]}),checkpoint);print(dict(site=site,seed=seed,step=beststep,l1=metrics['validation_smooth_l1'],budget32=metrics['budgets'][2]),flush=True)
  del val,tp,vp,tw,vw
 audits=dict(hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},finite=all(torch.isfinite(torch.tensor(r['validation_smooth_l1'])).item() for r in records),max_orthogonality_error=max(r['orthogonality_error'] for r in records),test_evaluations=0,whole_model_forwards=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'];torch.save(dict(rotations=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
