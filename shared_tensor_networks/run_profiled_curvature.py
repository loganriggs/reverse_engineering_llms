import hashlib,json,time
from pathlib import Path
import numpy as np
from profiled_curvature import evaluate
from multitask import monomials,product_map
from experiment import gram
src=Path('results/normalized_readout/panel');out=Path('results/profiled_curvature');out.mkdir(exist_ok=False)
keys=monomials(8);G=gram(keys);records=[]
for seed in range(232101,232106):
 data=np.load(src/f'{seed}_data.npz');x,y=data['x'],data['y'];truth=data['truth'];xa=data['xa'];px=np.prod(xa[:,None,:]**np.array(keys)[None,:,:],axis=2)
 for init in [233001,233002]:
  initial=np.load(src/f'{seed}_{init}_reduced.npz')['initial']
  for method in ['full','projected']:
   theta=initial.copy();states=[theta.copy()];steps=[];grads=[];history=[];status='cap';start=time.perf_counter()
   for iteration in range(100):
    r,J,A,U,ab=evaluate(theta,x,y);loss=.5*r@r;g=J.T@r
    if max(abs(g))<1e-8:status='gradient';break
    matrix=J if method=='full' else A;H=matrix.T@matrix+.001*np.eye(64);delta=np.linalg.solve(H,-g)
    assert g@delta<0
    accepted=False
    for trial in range(12):
     alpha=2.**(-trial);candidate=theta+alpha*delta
     rn,_,_=evaluate(candidate,x,y,derivatives=False);newloss=.5*rn@rn
     if newloss<=loss+1e-4*alpha*(g@delta):accepted=True;break
    steps.append(delta);grads.append(g)
    predicted=-(alpha*g@delta+.5*alpha**2*np.sum((matrix@delta)**2))
    history.append(dict(iteration=iteration,loss=float(loss),candidate_loss=float(newloss),alpha=alpha,accepted=accepted,trials=trial+1,
      directional_derivative=float(g@delta),predicted_decrease=float(predicted),actual_decrease=float(loss-newloss),
      extra_curvature_trace_ratio=float(np.sum((J-A)**2)/np.sum(A*A))))
    if not accepted:status='line_search';break
    theta=candidate;states.append(theta.copy())
    if loss-newloss<=1e-10*max(1.,loss):status='relative_change';break
   elapsed=time.perf_counter()-start;r,U,ab=evaluate(theta,x,y,derivatives=False)
   F=product_map(ab[:4],ab[4:],keys);adapt=np.linalg.lstsq(px@F,data['ya'],rcond=None)[0]
   e=F@U-truth[:,:10];et=F@adapt-truth[:,10:]
   row=dict(seed=seed,init=init,method=method,status=status,curvature_calls=iteration+1,steps=len(history),seconds=elapsed,
       objective=float(.5*r@r),source_mse=float(np.sum(e*(G@e))/10),transfer_mse=float(np.sum(et*(G@et))/4))
   file=out/f'{seed}_{init}_{method}.npz';np.savez(file,states=np.array(states),steps=np.array(steps),gradients=np.array(grads),readout=U,features=F,transfer=adapt)
   (out/f'{seed}_{init}_{method}.json').write_text(json.dumps(history,indent=2)+'\n')
   row['checkpoint_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();row['data_sha256']=hashlib.sha256((src/f'{seed}_data.npz').read_bytes()).hexdigest()
   records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2)+'\n');print(row,flush=True)
