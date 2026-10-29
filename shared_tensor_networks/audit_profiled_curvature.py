import hashlib,itertools,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from profiled_curvature import evaluate
from multitask import monomials,product_map
from experiment import gram
from direct_bilinear_readout import quadratic_coefficients
root=Path('results/profiled_curvature');src=Path('results/normalized_readout/panel');rows=json.loads((root/'results.json').read_text());assert len(rows)==20
keys=monomials(8);G=gram(keys);qkeys=[tuple(int(k==i)+int(k==j) for k in range(8)) for i,j in itertools.combinations_with_replacement(range(8),2)];Gq=gram(qkeys)
records=[];torch.set_default_dtype(torch.float64)
for row in rows:
 name=f"{row['seed']}_{row['init']}_{row['method']}";file=root/f'{name}.npz';dfile=src/f"{row['seed']}_data.npz"
 assert hashlib.sha256(file.read_bytes()).hexdigest()==row['checkpoint_sha256'];assert hashlib.sha256(dfile.read_bytes()).hexdigest()==row['data_sha256']
 z=np.load(file);data=np.load(dfile);history=json.loads((root/f'{name}.json').read_text());x,y=data['x'],data['y']
 # QR/literal-forward objective independent of producer residual function.
 def replay(theta):
  raw=theta.reshape(8,8);norm=np.linalg.norm(raw,axis=1);ab=raw/norm[:,None]
  hi=(x@ab[:4].T)*(x@ab[4:].T);phi=np.stack([hi[:,i]*hi[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],axis=1)
  Z=np.concatenate([phi,np.sqrt(.001)*np.eye(10)]);Y=np.concatenate([y,np.zeros((10,10))]);Q,R=np.linalg.qr(Z,mode='reduced');U=np.linalg.solve(R,Q.T@Y)
  return .5*np.sum((Z@U-Y)**2)+.5*np.sum((norm-1)**2),ab,U
 max_equation=0.;grad_error=0.
 for j,rec in enumerate(history):
  theta=z['states'][j];loss,_,_=replay(theta);np.testing.assert_allclose(loss,rec['loss'],rtol=1e-9,atol=1e-8)
  r,J,A,_,_=evaluate(theta,x,y);g=J.T@r;M=J if row['method']=='full' else A;step=z['steps'][j]
  err=np.linalg.norm((M.T@M+.001*np.eye(64))@step+g)/max(1.,np.linalg.norm(g));assert err<1e-7;max_equation=max(max_equation,err)
  np.testing.assert_allclose(g,z['gradients'][j],rtol=1e-8,atol=1e-8)
  for trial in range(rec['trials']):
   alpha=2.**(-trial);candidate=theta+alpha*step;nl,_,_=replay(candidate)
   passes=nl<=loss+1e-4*alpha*g@step
   if trial<rec['trials']-1:assert not passes
   else:
    assert passes==rec['accepted'];np.testing.assert_allclose(nl,rec['candidate_loss'],rtol=1e-8,atol=1e-8)
  if rec['accepted']:np.testing.assert_array_equal(z['states'][j+1],theta+rec['alpha']*step)
 # Independent autograd gradient at each final checkpoint, without differentiating through readout optimum (envelope).
 theta=z['states'][-1];loss,ab,U=replay(theta);vt=torch.tensor(theta,requires_grad=True)
 raw=vt.reshape(8,8);norm=torch.linalg.vector_norm(raw,dim=1);unit=raw/norm[:,None];hi=(torch.tensor(x)@unit[:4].T)*(torch.tensor(x)@unit[4:].T)
 phi=torch.stack([hi[:,i]*hi[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],dim=1)
 L=.5*torch.sum((phi@torch.tensor(U)-torch.tensor(y))**2)+.5*torch.sum((norm-1)**2)
 tg=torch.autograd.grad(L,vt)[0].detach().numpy();r,J,_,_,_=evaluate(theta,x,y)
 grad_error=np.linalg.norm(J.T@r-tg)/max(1.,np.linalg.norm(tg));assert grad_error<1e-7,grad_error
 np.testing.assert_allclose(loss,row['objective'],rtol=1e-8,atol=1e-8)
 F=product_map(ab[:4],ab[4:],keys);e=F@U-data['truth'][:,:10]
 np.testing.assert_allclose(np.sum(e*(G@e))/10,row['source_mse'],rtol=1e-8,atol=1e-10)
 xa=data['xa'];ha=(xa@ab[:4].T)*(xa@ab[4:].T);fa=np.stack([ha[:,i]*ha[:,j] for i,j in itertools.combinations_with_replacement(range(4),2)],axis=1)
 Q,R=np.linalg.qr(fa,mode='reduced');transfer=np.linalg.solve(R,Q.T@data['ya']);e=F@transfer-data['truth'][:,10:]
 np.testing.assert_allclose(np.sum(e*(G@e))/4,row['transfer_mse'],rtol=1e-7,atol=1e-9)
 np.testing.assert_allclose(fa@transfer,fa@z['transfer'],rtol=1e-7,atol=1e-9)
 c=quadratic_coefficients(ab[:4],ab[4:]);t=quadratic_coefficients(data['teacher_A'],data['teacher_B']);c/=np.sqrt(np.sum(c*(Gq@c),axis=0));t/=np.sqrt(np.sum(t*(Gq@t),axis=0))
 cos=np.abs(c.T@Gq@t);ii,jj=linear_sum_assignment(-cos)
 records.append(dict(seed=row['seed'],init=row['init'],method=row['method'],max_step_equation_error=max_equation,autograd_gradient_error=grad_error,
  min_component_cosine=float(min(cos[ii,jj])),all_components_above_099=bool(min(cos[ii,jj])>.99),audited_steps=len(history)))
(root/'audit.json').write_text(json.dumps(records,indent=2)+'\n');print('Audited20 trajectories, steps',sum(r['audited_steps'] for r in records))
for m in ['full','projected']:print(m,'recovery',sum(r['all_components_above_099'] for r in records if r['method']==m))
