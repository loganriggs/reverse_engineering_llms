"""Exact modular minors for d4/h2, three full-product task readouts."""
import json,math
from pathlib import Path
import numpy as np
import sympy as sp

PRIME=2147483647
assert all(PRIME%d for d in range(2,math.isqrt(PRIME)+1))
PAIRS=[(0,0),(0,1),(1,1)]

def pivots(matrix):
 m=np.array(matrix,dtype=np.int64)%PRIME;ids=list(range(len(m)));selected=[];columns=[];r=0
 for c in range(m.shape[1]):
  options=np.flatnonzero(m[r:,c])
  if not len(options):continue
  k=r+int(options[0]);m[[r,k]]=m[[k,r]];ids[r],ids[k]=ids[k],ids[r]
  inv=pow(int(m[r,c]),-1,PRIME);assert int(m[r,c])*inv%PRIME==1
  m[r]=m[r]*inv%PRIME
  for j in range(r+1,len(m)):m[j]=(m[j]-int(m[j,c])*m[r])%PRIME
  selected.append(ids[r]);columns.append(c);r+=1
  if r==len(m):break
 return selected,columns

def determinant(matrix):
 m=np.array(matrix,dtype=np.int64)%PRIME;value=1
 for c in range(len(m)):
  choices=np.flatnonzero(m[c:,c])
  if not len(choices):return 0
  k=c+int(choices[0])
  if k!=c:m[[c,k]]=m[[k,c]];value=-value
  pivot=int(m[c,c]);value=value*pivot%PRIME;inv=pow(pivot,-1,PRIME)
  for row in range(c+1,len(m)):
   factor=int(m[row,c])*inv%PRIME;m[row]=(m[row]-factor*m[c])%PRIME
 return value%PRIME

def forward(v,x):
 A,B,W=v[:8].reshape(2,4),v[8:16].reshape(2,4),v[16:].reshape(3,3)
 z=(x@A.T)*(x@B.T);phi=np.stack([z[:,i]*z[:,j] for i,j in PAIRS],1)
 return phi@W.T,z,phi

def jacobian(v,x):
 A,B,W=v[:8].reshape(2,4),v[8:16].reshape(2,4),v[16:].reshape(3,3)
 _,z,phi=forward(v,x);J=np.zeros((len(x),3,25),dtype=np.int64)
 for which,other in [(0,B),(1,A)]:
  for i in range(2):
   for j in range(4):
    dz=x[:,j]*(x@other[i]);dp=np.stack([(dz*z[:,b] if a==i else 0)+(dz*z[:,a] if b==i else 0) if i in [a,b] else np.zeros(len(x),dtype=np.int64) for a,b in PAIRS],1)
    J[:,:,which*8+i*4+j]=dp@W.T
 for task in range(3):J[:,task,16+task*3:16+(task+1)*3]=phi
 return J.reshape(-1,25)

A=np.array([[1,0,0,0],[0,0,1,0]]);B=np.array([[0,1,0,0],[0,0,0,1]]);W=np.eye(3,dtype=np.int64);v=np.r_[A.ravel(),B.ravel(),W.ravel()]
gauges=[]
for i in range(2):
 g=np.zeros(25,dtype=np.int64);g[i*4:i*4+4]=A[i];g[8+i*4:8+i*4+4]=-B[i];gauges.append(g)
 g=np.zeros(25,dtype=np.int64);g[i*4:i*4+4]=A[i];g[16:]=(-W*np.array([int(a==i)+int(b==i) for a,b in PAIRS])[None,:]).ravel();gauges.append(g)
gauges=np.array(gauges).T;assert len(pivots(gauges)[1])==4
records=[]
for bound,seed in [(b,s) for b in [3,30] for s in [260101,260102,260103]]:
 xall=np.random.default_rng(seed).integers(-bound,bound+1,size=(15,4),dtype=np.int64)
 for n in [4,6,7,8,15]:
  x=xall[:n];J=jacobian(v,x);_,z,phi=forward(v,x)
  assert len(pivots(z)[1])==2 and len(pivots(phi)[1])==3
  for step in [1,2]:
   for k in range(25):
    plus=v.copy();minus=v.copy();plus[k]+=step;minus[k]-=step
    diff=(forward(plus,x)[0]-forward(minus,x)[0]).ravel();assert np.all(diff%(2*step)==0)
    np.testing.assert_array_equal(diff//(2*step),J[:,k])
  np.testing.assert_array_equal(J@gauges,np.zeros((3*n,4),dtype=np.int64))
  rr,cc=pivots(J);rw,cw=pivots(J[:,16:]);rank=len(cc);readout_rank=len(cw)
  upper=9+min(12,3*(n-3),2*(n-2));assert rank<=upper and readout_rank==9,(seed,n,rank,upper)
  rational_rank=int(sp.Matrix(J.tolist()).rank());assert rational_rank==rank
  assert int(sp.Matrix(J[:,16:].tolist()).rank())==readout_rank
  minor=determinant(J[np.ix_(rr,cc)]);readout_minor=determinant(J[:,16:][np.ix_(rw,cw)]);assert minor and readout_minor
  records.append(dict(seed=seed,input_bound=bound,n=n,input_points=x.tolist(),rank=rank,readout_rank=readout_rank,profiled_rank=rank-readout_rank,geometric_upper_bound=upper,attains_upper_bound=rank==upper,exact_rational_rank=rational_rank,
                      pivot_rows=rr,pivot_columns=cc,minor_mod_prime=int(minor),readout_minor_mod_prime=int(readout_minor),jacobian=J.tolist()))
out=Path('results/sampled_information_certificate');out.mkdir(parents=True,exist_ok=True)
(out/'certificate.json').write_text(json.dumps(dict(prime=PRIME,primality_trial_division=True,parameters=v.tolist(),gauge_vectors=gauges.tolist(),records=records),indent=2)+'\n')
print('Certified',len(records),'integer fixtures; profiled ranks',[(r['input_bound'],r['seed'],r['n'],r['profiled_rank'],r['attains_upper_bound']) for r in records])
