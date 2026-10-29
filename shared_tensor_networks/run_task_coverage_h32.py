import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from batched_coverage import BatchedCoverage,losses,clip_each
from quartic_cross_gram import product_factors,cross_gram
from larger_hidden_transfer import values
root=Path('results/task_coverage_h32');root.mkdir(exist_ok=True);out=root/'panel';out.mkdir(exist_ok=False);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cases=[(n,t,init) for n in [2048,8192] for t in [132,528] for init in [252001,252002,'teacher']];records=[]
def normalize(A,B,W):
 F=product_factors(A,B,np.ones(32));G=cross_gram(F,F);norm=np.sqrt(np.einsum('tp,pq,tq->t',W,G,W));return W/norm[:,None]
def cosines(A,B,TA,TB):
 Q=(A[:,:,None]*B[:,None,:]+B[:,:,None]*A[:,None,:])/2;T=(TA[:,:,None]*TB[:,None,:]+TB[:,:,None]*TA[:,None,:])/2
 def inner(Q,T):return 2*np.einsum('iab,jab->ij',Q,T)+np.outer(np.trace(Q,axis1=1,axis2=2),np.trace(T,axis1=1,axis2=2))
 C=abs(inner(Q,T))/np.sqrt(np.outer(np.diag(inner(Q,Q)),np.diag(inner(T,T))));i,j=linear_sum_assignment(-C);return C[i,j]
for seed in range(251101,251106):
 rng=np.random.default_rng(seed);TA=rng.normal(size=(32,64))/np.sqrt(64);TB=rng.normal(size=(32,64))/np.sqrt(64);TW=normalize(TA,TB,rng.normal(size=(532,528))/np.sqrt(528));TF=product_factors(TA,TB,np.ones(32));TG=cross_gram(TF,TF)
 assert np.linalg.matrix_rank(np.r_[TA,TB])==64 and np.linalg.matrix_rank(TW[:528])==528
 drng=np.random.default_rng(seed+5000);x=drng.normal(size=(8192,64));noise=drng.normal(size=(8192,528));xa=drng.normal(size=(2048,64));an=drng.normal(size=(2048,4));y=values(TF,x)@TW[:528].T+.05*noise;ya=values(TF,xa)@TW[528:].T+.05*an
 np.savez_compressed(out/f'{seed}_data.npz',xa=xa,ya=ya,teacher_A=TA,teacher_B=TB,teacher_W=TW,adaptation_noise=an)
 for index in range(4):
  sl=slice(index*2048,(index+1)*2048);np.savez_compressed(out/f'{seed}_source_{index}.npz',x=x[sl],y=y[sl],noise=noise[sl])
 initials={}
 for init in [252001,252002]:
  irng=np.random.default_rng(init);A=irng.normal(size=(32,64))/np.sqrt(64);B=irng.normal(size=(32,64))/np.sqrt(64);W=normalize(A,B,irng.normal(size=(528,528))/np.sqrt(528));initials[init]=(A,B,W)
 initials['teacher']=(TA,TB,TW[:528])
 As,Bs,Ws=map(np.stack,zip(*(initials[i] for n,t,i in cases)))
 mask=np.zeros((12,8192,528),dtype=np.float32)
 for i,(n,t,init) in enumerate(cases):mask[i,:n,:t]=1
 net=BatchedCoverage(torch.tensor(As,dtype=torch.float32,device='cuda'),torch.tensor(Bs,dtype=torch.float32,device='cuda'),torch.tensor(Ws,dtype=torch.float32,device='cuda'))
 xx=torch.tensor(x,dtype=torch.float32,device='cuda');yy=torch.tensor(y,dtype=torch.float32,device='cuda');mm=torch.tensor(mask,device='cuda');opt=torch.optim.Adam(net.parameters(),lr=.003);clipped=np.zeros(12,dtype=int);curves=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 for step in range(3000):
  opt.zero_grad();ls=losses(net,xx,yy,mm);assert torch.isfinite(ls).all(),(seed,step,'nonfinite loss');ls.sum().backward();gn=clip_each(net);assert torch.isfinite(gn).all(),(seed,step,'nonfinite gradient');clipped+=(gn.detach().cpu().numpy()>10);opt.step()
  if (step+1)%200==0:curves.append(dict(step=step+1,preupdate_loss=ls.detach().cpu().tolist(),gradient_norm=gn.detach().cpu().tolist()));print(seed,'step',step+1,flush=True)
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 resources=dict(seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 (out/f'{seed}_resources.json').write_text(json.dumps(resources,indent=2)+'\n')
 Aall,Ball,Wall=[p.detach().cpu().numpy().astype(float) for p in [net.A,net.B,net.W]]
 for case,(n,t,init) in enumerate(cases):
  A,B,W=Aall[case],Ball[case],Wall[case];F=product_factors(A,B,np.ones(32));G=cross_gram(F,F);C=cross_gram(F,TF);E=np.einsum('tp,pq,tq->t',TW,TG,TW)
  source=np.einsum('tp,pq,tq->t',W[:t],G,W[:t])-2*np.sum((W[:t]@C)*TW[:t],axis=1)+E[:t]
  saved=dict(initial_A=As[case].astype(np.float32),initial_B=Bs[case].astype(np.float32),initial_W=Ws[case].astype(np.float32),A=A,B=B,W=W);trans={}
  for k in [512,2048]:
   for typ,FF,GG,CC in [('learned',F,G,C),('oracle',TF,TG,TG)]:
    coef=np.linalg.lstsq(values(FF,xa[:k]),ya[:k],rcond=1e-12)[0];error=np.sum(coef*(GG@coef),axis=0)-2*np.sum(coef*(CC@TW[528:].T),axis=0)+E[528:];trans[f'{typ}_{k}']=error.tolist();saved[f'{typ}_{k}']=coef
  train=np.mean((values(F,x[:n])@W[:t].T-y[:n,:t])**2);cos=cosines(A,B,TA,TB)
  name=f'{seed}_n{n}_t{t}_{init}';np.savez_compressed(out/f'{name}.npz',**saved)
  records.append(dict(name=name,teacher=seed,n=n,tasks=t,labels=n*t,initialization=str(init),active_parameters=4096+t*528,train_mse=float(train),population_mse=source.tolist(),component_cosines=cos.tolist(),transfer=trans,clipped_steps=int(clipped[case]),batch_seconds=seconds))
 (out/f'{seed}_curves.json').write_text(json.dumps(curves,indent=2)+'\n');(out/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(seed,'completed12fits',round(seconds,1),flush=True)
(root/'completed.json').write_text(json.dumps(dict(status='complete',fits=60,steps=3000,device='RTX5080 CUDA float32 TF32off'),indent=2)+'\n')
