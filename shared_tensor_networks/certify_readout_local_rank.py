"""Exact integer-polynomial checks and modular local-rank certificate for d6/h3."""
from pathlib import Path
import numpy as np
from run_readout_recovery_h3 import maps,N,H,D,P,PAIRS
from run_svhn_standard import digest,write
PRIME=2147483647
from math import isqrt
assert all(PRIME % divisor for divisor in range(2,isqrt(PRIME)+1))

def modular_rank(a):
 a=[[int(v)%PRIME for v in row] for row in a];labels=list(range(len(a)));rank=0;rows=[];cols=[]
 for c in range(len(a[0])):
  pivot=next((i for i in range(rank,len(a)) if a[i][c]),None)
  if pivot is None:continue
  a[rank],a[pivot]=a[pivot],a[rank];labels[rank],labels[pivot]=labels[pivot],labels[rank]
  rows.append(labels[rank]);cols.append(c);inv=pow(a[rank][c],-1,PRIME)
  a[rank]=[(v*inv)%PRIME for v in a[rank]]
  for i in range(rank+1,len(a)):
   factor=a[i][c]
   if factor:a[i]=[(v-factor*t)%PRIME for v,t in zip(a[i],a[rank])]
  rank+=1
 return rank,rows,cols

def determinant(a):
 a=[[int(v)%PRIME for v in row] for row in a];det=1
 for c in range(len(a)):
  pivot=next(i for i in range(c,len(a)) if a[i][c])
  if pivot!=c:a[c],a[pivot]=a[pivot],a[c];det=-det
  q=a[c][c];det=det*q%PRIME;inv=pow(q,-1,PRIME)
  for i in range(c+1,len(a)):
   f=a[i][c]*inv%PRIME
   a[i]=[(v-f*t)%PRIME for v,t in zip(a[i],a[c])]
 return det%PRIME

root=Path(__file__).parent;out=root/'results/readout_recovery_h3';rng=np.random.default_rng(220101)
A=rng.integers(-3,4,size=(H,D));B=rng.integers(-3,4,size=(H,D));W=rng.integers(-3,4,size=(P,P))
assert modular_rank(np.concatenate([A,B]))[0]==D
records=[]
for arm,k in [('one',1),('three',3),('six',6),('interventions',1)]:
 v=np.concatenate([A.ravel(),B.ravel(),W[:k].ravel()]).astype(float);C,J,_,_=maps(v,arm);J=J.reshape(-1,len(v))
 assert np.array_equal(J,np.rint(J)) and abs(J).max()<2**40
 # Each scalar parameter enters output with degree at most two, hence central differences are exact.
 for i in range(len(v)):
  for step in [1.,2.]:
   plus=v.copy();minus=v.copy();plus[i]+=step;minus[i]-=step
   central=(maps(plus,arm)[0]-maps(minus,arm)[0]).ravel()/(2*step)
   np.testing.assert_array_equal(central,J[:,i])
 gauges=np.zeros((len(v),2*H),dtype=np.int64)
 for h in range(H):
  gauges[h*D:(h+1)*D,h]=A[h];gauges[N+h*D:N+(h+1)*D,h]=-B[h]
  gauges[h*D:(h+1)*D,H+h]=A[h]
  for t in range(k):
   for j,(a,b) in enumerate(PAIRS):gauges[2*N+t*P+j,H+h]=-W[t,j]*(int(a==h)+int(b==h))
 integer=J.astype(np.int64);np.testing.assert_array_equal(integer@gauges,0)
 assert modular_rank(gauges)[0]==2*H
 rank,rr,cc=modular_rank(integer);assert rank==len(v)-2*H
 det=determinant(integer[np.ix_(rr,cc)]);assert det!=0
 records.append(dict(arm=arm,parameters=len(v),rank=rank,gauge_dimension=2*H,minor_rows=rr,minor_columns=cc,minor_determinant_mod_prime=det,parameters_fixture=v.tolist(),coefficient_jacobian=integer.tolist()))
 print(arm,len(v),rank,det)
write(out/'local_rank_certificate.json',dict(status='passed',prime=PRIME,records=records,source_sha256=digest(Path(__file__)),map_source_sha256=digest(root/'run_readout_recovery_h3.py'),arithmetic='small integer fixture: polynomial additions/products exact in float64; central differences at two integer offsets independently match every derivative; modular minor certifies rank'))
