"""H755 matched task-blind scalar and block TopK baselines for L5H3/H5."""
import json,time
from pathlib import Path
import torch,numpy as np
from scipy.optimize import linear_sum_assignment
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_verified_head_sparse import SignedBSF,mse
STEM='bilin18_l5_sparse_featurizers'
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','fits':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','implementation':ROOT/'scripts/run_bilin18_verified_head_sparse.py','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H755','hashes':hashes,'seeds':[7551,7552,7553],'steps':3000,'batch_size':512,'learning_rate':.001,'width':128,'active_scalars':8,'block_dimensions':[1,4],'design':'Task-blind H754 write-PCA coordinates, global train RMS scale only. Signed linear encoder and unit-row decoder with learned biases. Scalar chooses8 signed coordinates; block4 chooses2 blocks by squared coordinate norm. Matched parameter count, initialization seed, training minibatch sequence, Adam schedule, and validation checkpoint rule (MSE every250steps including0). Every seed reported.','limitations':'Finite-training pilot, not converged. Unit-decoder signed block TopK baseline, not orthonormal-within-block Grassmannian BSF. Equal active dimension but different combinatorial supports. No task examples in fitting. No causal/manifold conclusion from reconstruction.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['data'],weights_only=False)['activations'];fits=torch.load(paths['fits'],weights_only=False)['fits'];records=[];states={};representations={};training_finite=True;forwarderrors=[];allbudgets=[]
 dest=ART/'l5_sparse_featurizers';dest.mkdir(exist_ok=True)
 for site,fit in fits.items():
  E=fit['write_pca']['encode'];mu=fit['mean'];train=((data[site][:2048].reshape(-1,128).double()-mu)@E).float().cuda();val=((data[site][2048:].reshape(-1,128).double()-mu)@E).float().cuda();scale=train.square().mean().sqrt();x=train/scale;v=val/scale
  for r in [1,4]:
   for seed in cfg['seeds']:
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);gen=torch.Generator(device='cuda').manual_seed(seed);m=SignedBSF(r).cuda();opt=torch.optim.Adam(m.parameters(),lr=.001);best=float('inf');history=[];beststep=0
    for step in range(3001):
     if step%250==0:
      score=mse(m,v);history.append({'step':step,'validation_write_relative_MSE':score})
      if score<best:best=score;beststep=step;beststate={k:t.detach().cpu().clone() for k,t in m.state_dict().items()}
     if step==3000:break
     ii=torch.randint(len(x),(512,),generator=gen,device='cuda');loss=(m(x[ii])-x[ii]).square().mean();opt.zero_grad();loss.backward();training_finite=training_finite and bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters());assert training_finite;opt.step()
     with torch.no_grad():m.decoder.div_(m.decoder.norm(dim=1,keepdim=True).clamp_min(1e-10))
    m.load_state_dict(beststate)
    with torch.no_grad():
     z,mask=m.encode(v);budget=mask.sum(-1)*r;allbudgets.append(bool((budget==8).all()));forwarderrors.append(float((m(v)-(z@m.decoder+m.bias)).abs().max()));D=m.decoder.detach().cpu();mask=mask.cpu();normerr=float((D.norm(dim=1)-1).abs().max());rec={'site':site,'block_dimension':r,'seed':seed,'selected_step':beststep,'history':history,'validation_write_relative_MSE':mse(m,v),'train_write_relative_MSE':mse(m,x),'dead_blocks':int((mask.sum(0)==0).sum()),'decoder_norm_max_error':normerr,'parameter_count':sum(p.numel() for p in m.parameters())};assert normerr<1e-5
    state={'model':beststate,'block_dimension':r,'input_scale':float(scale)};states[site,r,seed]=state;representations[site,r,seed]={'decoder':D,'mask':mask};records.append(rec);torch.save({'hashes':hashes,'record':rec,'state':state,'representation':representations[site,r,seed]},dest/f'{site}_r{r}_{seed}.pt');print('FIT',site,r,seed,beststep,rec['validation_write_relative_MSE'],flush=True)
 stability=[]
 for site in fits:
  for r in [1,4]:
   for a,b in [(7551,7552),(7551,7553),(7552,7553)]:
    ra,rb=representations[site,r,a],representations[site,r,b];da,db=ra['decoder'].double(),rb['decoder'].double()
    if r==1:sim=((da/da.norm(dim=1,keepdim=True))@(db/db.norm(dim=1,keepdim=True)).T).abs()
    else:
     qa=torch.linalg.qr(da.reshape(-1,r,128).transpose(1,2)).Q;qb=torch.linalg.qr(db.reshape(-1,r,128).transpose(1,2)).Q;sim=torch.einsum('air,bis->abrs',qa,qb).square().sum((-1,-2))/r
    ia,ib=linear_sum_assignment(-sim.numpy());ma,mb=ra['mask'][:,ia].bool(),rb['mask'][:,ib].bool();jac=(ma&mb).sum(0)/(ma|mb).sum(0).clamp_min(1);stability.append({'site':site,'block_dimension':r,'seeds':[a,b],'decoder_similarity':float(sim[ia,ib].mean()),'support_jaccard':float(jac.mean()),'similarity_metric':'absolute_cosine' if r==1 else 'mean_squared_principal_cosine'})
 audits={'hashes_unchanged':{k:sha(p)==hashes[k] for k,p in paths.items()},'finite_training_and_gradients':training_finite,'every_sample_active_scalar_budget_eight':all(allbudgets),'decode_formula_max_error':max(forwarderrors),'all_parameter_counts':sorted(set(r['parameter_count'] for r in records)),'whole_model_forwards':0,'test_evaluations':0};assert all(audits['hashes_unchanged'].values()) and all(allbudgets) and audits['all_parameter_counts']==[33024]
 torch.save({'states':states},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'stability':stability,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
