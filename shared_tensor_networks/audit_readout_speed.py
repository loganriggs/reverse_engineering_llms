import hashlib,itertools,json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from quartic_cross_gram import product_factors,cross_gram
from readout_speed import calc

ROOT=Path('results/readout_speed');SOURCE=Path('results/normalized_readout/panel')
PAIRS=list(itertools.combinations_with_replacement(range(4),2))

def literal(rows,x):
 z=(x@rows[:4].T)*(x@rows[4:].T)
 return np.stack([z[:,i]*z[:,j] for i,j in PAIRS],1)

def quadratic_match(rows,teacher):
 pairs=list(itertools.combinations_with_replacement(range(8),2))
 def coef(r):return np.array([r[:4,i]*r[4:,j]+(r[:4,j]*r[4:,i] if i!=j else 0) for i,j in pairs]).T
 gram=np.zeros((len(pairs),len(pairs)))
 for i,a in enumerate(pairs):
  for j,b in enumerate(pairs):
   counts=np.bincount(a+b,minlength=8)
   gram[i,j]=0 if np.any(counts%2) else np.prod([1 if c<4 else 3 for c in counts])
 a,b=coef(rows),coef(teacher);C=abs(a@gram@b.T)/np.sqrt(np.outer(np.diag(a@gram@a.T),np.diag(b@gram@b.T)))
 i,j=linear_sum_assignment(-C);return C[i,j]

def main():
 records=json.loads((ROOT/'results.json').read_text());assert len(records)==70
 checks=[];sourcehash={};cached={};initials={}
 for r in records:
  path=SOURCE/f'{r["seed"]}_data.npz';data=np.load(path);x,y=data['x'],data['y'];saved=np.load(ROOT/r['checkpoint']);rows,U=saved['rows'],saved['U']
  sourcehash[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  f0=literal(saved['initial_rows'],x);star=np.linalg.lstsq(np.r_[f0,np.sqrt(.001)*np.eye(10)],np.r_[y,np.zeros((10,10))],rcond=None)[0]
  np.testing.assert_allclose(star,saved['initial_U'],rtol=1e-10,atol=1e-10)
  key=(r['seed'],r['init']);initial=np.r_[saved['initial_rows'].ravel(),saved['initial_U'].ravel()]
  if key in initials:np.testing.assert_array_equal(initial,initials[key])
  else:initials[key]=initial
  log=saved['logs'];assert len(log)==r['steps'] and np.all(log[:,2]<=log[:,1]+1e-10)
  if len(log)>1:np.testing.assert_allclose(log[1:,1],log[:-1,2],atol=1e-12)
  np.testing.assert_allclose(log[:,3],.01*2.**(-log[:,4]),atol=0,rtol=1e-14)
  np.testing.assert_allclose(log[:,3].sum(),r['hidden_clock'],rtol=1e-12)
  for step,state in zip(saved['state_steps'],saved['states']):
   rr=state[:64].reshape(8,8);uu=state[64:].reshape(10,10);np.testing.assert_allclose(np.linalg.norm(rr,axis=1),1,atol=1e-12)
   loss=(np.sum((literal(rr,x)@uu-y)**2)+.001*np.sum(uu**2))/(2*len(x))
   target=log[0,1] if step==0 else log[step-1,2]
   np.testing.assert_allclose(loss,target,rtol=1e-9,atol=1e-11)
  np.testing.assert_allclose(saved['states'][-1],np.r_[rows.ravel(),U.ravel()],atol=0)
  if r['seed'] not in cached:
   TF=product_factors(data['teacher_A'],data['teacher_B'],np.ones(4));TG=cross_gram(TF,TF);W=data['teacher_W'];energies=np.einsum('tp,pq,tq->t',W,TG,W)
   cached[r['seed']]=(TF,W,energies)
  TF,W,energies=cached[r['seed']];F=product_factors(rows[:4],rows[4:],np.ones(4));G=cross_gram(F,F);C=cross_gram(F,TF)
  source=np.sum(U*(G@U),axis=0)-2*np.sum(U*(C@W[:10].T),axis=0)+energies[:10]
  transfer=np.linalg.lstsq(literal(rows,data['xa']),data['ya'],rcond=1e-12)[0]
  np.testing.assert_allclose(transfer,saved['transfer'],rtol=1e-7,atol=1e-8)
  error=np.sum(transfer*(G@transfer),axis=0)-2*np.sum(transfer*(C@W[10:].T),axis=0)+energies[10:]
  np.testing.assert_allclose(source.mean(),r['source_mse'],rtol=1e-7,atol=1e-9)
  np.testing.assert_allclose(error.mean(),r['transfer_mse'],rtol=1e-7,atol=1e-9)
  np.testing.assert_allclose(quadratic_match(rows,np.r_[data['teacher_A'],data['teacher_B']]),r['cosines'],rtol=1e-10,atol=1e-10)
  v=torch.tensor(np.r_[rows.ravel(),U.ravel()],dtype=torch.float64,requires_grad=True)
  raw=v[:64].reshape(8,8);unit=raw/torch.linalg.vector_norm(raw,dim=1)[:,None];xx=torch.tensor(x)
  z=(xx@unit[:4].T)*(xx@unit[4:].T);ff=torch.stack([z[:,i]*z[:,j] for i,j in PAIRS],1);uu=v[64:].reshape(10,10)
  obj=((ff@uu-torch.tensor(y)).square().sum()+.001*uu.square().sum())/(2*len(x))
  gradient=torch.autograd.grad(obj,v)[0].numpy();_,gr,gu=calc(rows,U,x,y)
  gradient_error=float(np.max(np.abs(gradient-np.r_[gr.ravel(),gu.ravel()])));assert gradient_error<1e-8
  checks.append(dict(checkpoint=r['checkpoint'],states=len(saved['states']),gradient_max_error=gradient_error,source_replay=float(source.mean()),transfer_replay=float(error.mean()),sha256=hashlib.sha256((ROOT/r['checkpoint']).read_bytes()).hexdigest()))
 result=dict(fits=len(checks),states=sum(c['states'] for c in checks),checks=checks,source_sha256=sourcehash,
             scope='Independent literal objectives, quadratic coefficient matching, Wick source/transfer errors, SVD adaptation, pairing, scalar monotonicity and clocks. Does not replay every internal line-search decision.')
 (ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('Audited',result['fits'],'fits and',result['states'],'states')

if __name__=='__main__':main()
