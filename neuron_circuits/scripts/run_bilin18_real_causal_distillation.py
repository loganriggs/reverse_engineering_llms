"""H943 compact preMLP4 update models trained with independent finite interventions."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from pathlib import Path
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_real_causal_distillation';MODES=['native','endpoints','effects'];SEEDS=[0,1];RANK=64
class Student(torch.nn.Module):
 def __init__(self,seed,mean):
  super().__init__();torch.manual_seed(94300+seed);self.left=torch.nn.Parameter(torch.randn(1152,RANK));self.right=torch.nn.Parameter(torch.randn(1152,RANK));self.out=torch.nn.Parameter(torch.randn(RANK,1152)*.001);self.bias=torch.nn.Parameter(mean.clone())
 def forward(self,x):
  u=F.rms_norm(x,(1152,));L=F.normalize(self.left,dim=0);R=F.normalize(self.right,dim=0);return ((u@L)*(u@R))@self.out+self.bias

def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'data':ART/'bilin18_relay_natural_data_captures.pt','data_protocol':ROOT/'configs/bilin18_relay_natural_data.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H943','hashes':hashes,'rank':RANK,'modes':MODES,'seeds':SEEDS,'steps':2000,'batch':128,'site':'raw preMLP4 -> RMS -> full bilinear MLP update','training':'H851 original2048 training documents, one of16 captured positions per document chosen by document_index mod16. Fixed Gaussian tangential perturbation at5% raw-state norm per document, independent of learner and task labels.','evaluation':'512 H851 development documents from separate natural-document pool; reused research development, not reserved or fresh test. Independent5% tangent noise plus opposite sign and10% dose. No task directions loaded in fitting or selection.','objectives':'native: centered-energy-normalized outputMSE; endpoints: average native+edited endpointMSE at same normalizer; effects: native outputMSE + effectMSE divided by TRAINING effect energy. This deliberately changes weighting, not endpoint information. Same initializations, minibatches, architecture, Adam .003 and final2000step checkpoint.','limits':'Compact update surrogate, not activation autoencoder or discovered sparse ontology. Local update scores do not establish outputKL or task preservation. Known rank is NOT assumed here;64 is a fixed compression budget. Full input access and normalization are part of student computation.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 data=torch.load(paths['data'],weights_only=False);allx=data['captures']['train',4];assert allx.shape==(32768,1152);indices=torch.arange(2048)*16+torch.arange(2048)%16;tx=allx[indices].float();vx=data['captures']['validation',4].float();assert vx.shape==(512,1152)
 def perturb(x,seed):
  gen=torch.Generator().manual_seed(seed);noise=torch.randn(x.shape,generator=gen);noise=noise-x*(noise*x).sum(1,keepdim=True)/x.square().sum(1,keepdim=True);return noise/noise.norm(dim=1,keepdim=True)*(.05*x.norm(dim=1,keepdim=True))
 td=perturb(tx,943);vd=perturb(vx,944);tangent=max(float(((tx*td).sum(1)/(tx.norm(dim=1)*td.norm(dim=1))).abs().max()),float(((vx*vd).sum(1)/(vx.norm(dim=1)*vd.norm(dim=1))).abs().max()));assert tangent<1e-6
 model=load_full_model('cuda');mlp=model.blocks[4].mlp.float().eval();assert not mlp.config.gated;assert mlp.Left.bias is None and mlp.Right.bias is None
 for p in mlp.parameters():p.requires_grad_(False)
 teacherparams=sum(p.numel() for p in mlp.parameters());L=mlp.Left.weight;R=mlp.Right.weight;D=mlp.Down.weight;bias=mlp.Down_bias
 @torch.no_grad()
 def truth(x):return torch.cat([mlp(F.rms_norm(b.cuda(),(1152,))).cpu() for b in x.split(128)])
 with torch.no_grad():
  z=tx[:16].cuda();u=F.rms_norm(z,(1152,));explicit=F.linear(F.linear(u,L)*F.linear(u,R),D,bias);matherr=float((explicit-mlp(u)).abs().max())
  # Actual finite difference through normalized endpoints, independently assembled bilinear expansion.
  w=F.rms_norm((tx[:16]+td[:16]).cuda(),(1152,));du=w-u;exp=F.linear(F.linear(u,L)*F.linear(du,R)+F.linear(du,L)*F.linear(u,R)+F.linear(du,L)*F.linear(du,R),D);direct=mlp(w)-mlp(u);effecterr=float((exp-direct).norm()/direct.norm())
 print('ALGEBRA',matherr,effecterr,flush=True);assert matherr<1e-5 and effecterr<1e-4
 ty=truth(tx);te=truth(tx+td)-ty;vy=truth(vx);panels={name:dict(x=vx,delta=mult*vd,y=vy,effect=truth(vx+mult*vd)-vy) for name,mult in [('random5',1.),('opposite5',-1.),('random10',2.)]}
 torch.save(dict(train_x=tx,train_delta=td,train_y=ty,train_effect=te,train_flat_indices=indices,panels=panels,validation_documents=data['validation_documents'],validation_positions=data['validation_positions']),ART/f'{S}_targets.pt');del model,mlp,L,R,D,bias;torch.cuda.empty_cache()
 xx=tx.cuda();dd=td.cuda();yy=ty.cuda();ee=te.cuda();mu=yy.mean(0);native_den=(yy-mu).square().mean();effect_den=ee.square().mean();records=[];curves=[];models={};finite=True
 for seed in SEEDS:
  for mode in MODES:
   m=Student(seed,mu).cuda();opt=torch.optim.Adam(m.parameters(),lr=.003);gen=torch.Generator().manual_seed(94300+seed)
   for step in range(2000):
    ii=torch.randint(len(xx),(128,),generator=gen).cuda();a=m(xx[ii]);native=(a-yy[ii]).square().mean()/native_den
    if mode=='native':loss=native
    else:
     b=m(xx[ii]+dd[ii])
     if mode=='endpoints':loss=.5*(native+(b-yy[ii]-ee[ii]).square().mean()/native_den)
     else:loss=native+(b-a-ee[ii]).square().mean()/effect_den
    assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();opt.step()
    if step%250==0 or step==1999:curves.append(dict(mode=mode,seed=seed,step=step+1,loss=float(loss.detach()),native_loss=float(native.detach())))
   m=m.cpu();models[mode,seed]=m.state_dict()
   with torch.no_grad():
    for name,p in panels.items():
     pred=m(p['x']);edit=m(p['x']+p['delta'])-pred;records.append(dict(mode=mode,seed=seed,panel=name,native_error_ratio=float((pred-p['y']).norm()/(p['y']-mu.cpu()).norm()),effect_error_ratio=float((edit-p['effect']).norm()/p['effect'].norm()),parameters=sum(q.numel() for q in m.parameters())))
   print('DONE',mode,seed,flush=True)
 audits={'hashes_unchanged':all(sha(v)==hashes[k] for k,v in paths.items()),'finite':all(np.isfinite(r['effect_error_ratio']) for r in records),'models':6,'teacher_parameters':teacherparams,'tangent_cosine_max_abs':tangent,'explicit_MLP_max_abs':matherr,'finite_effect_algebra_relative_error':effecterr,'teacher_local_evaluations':len(tx)*2+len(vx)*4+48,'original_whole_model_forwards':0,'reserved_test_evaluations':0,'no_task_directions_or_labels_loaded':True};assert audits['hashes_unchanged'] and audits['finite'];torch.save(dict(models=models,native_den=float(native_den),effect_den=float(effect_den),mean=mu.cpu()),ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,curves=curves,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(records,audits,flush=True)
if __name__=='__main__':main()
