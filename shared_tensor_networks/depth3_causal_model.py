import math
import numpy as np

def moment(k):
 if k%2:return 0.
 return float(math.prod(range(1,k,2))**2)
G=np.array([[moment(8-i-j)*moment(i+j) for j in range(5)] for i in range(5)])
L=np.linalg.cholesky(G)
Ghidden=np.array([[moment(4-i-j)*moment(i+j) for j in range(3)] for i in range(3)])
Lhidden=np.linalg.cholesky(Ghidden)

def coefficients(v,ablate=False):
 A=v[:4].reshape(2,2);B=v[4:8].reshape(2,2);W=v[8:].reshape(3,3)
 q=np.array([np.convolve(A[i],B[i]) for i in range(2)])
 D=np.zeros((2,3,8))
 for i in range(2):
  for j in range(2):
   D[i,j:j+2,2*i+j]=B[i];D[i,j:j+2,4+2*i+j]=A[i]
 if ablate:q[0]=0;D[0]=0
 pairs=[(0,0),(0,1),(1,1)];p=np.array([np.convolve(q[i],q[j]) for i,j in pairs]);dp=np.zeros((3,5,8))
 for row,(i,j) in enumerate(pairs):
  for k in range(8):dp[row,:,k]=np.convolve(D[i,:,k],q[j])+np.convolve(q[i],D[j,:,k])
 J=np.zeros((3,5,17));J[:,:,:8]=np.einsum('tp,pik->tik',W,dp)
 for t in range(3):J[t,:,8+3*t:8+3*(t+1)]=p.T
 return W@p,J

def normalize(v):
 v=v.copy();c,_=coefficients(v);energy=np.einsum('ti,ij,tj->t',c,G,c)
 assert np.all(energy>1e-12);v[8:]=(v[8:].reshape(3,3)/np.sqrt(energy)[:,None]).ravel()
 return v

def teacher(seed):
 rng=np.random.default_rng(seed);R=np.eye(2)+.2*rng.normal(size=(2,2));assert abs(np.linalg.det(R))>.01
 A=[];B=[]
 for a,b in R:
  Q=np.array([[a,b],[b,-a]]);e,U=np.linalg.eigh(Q)
  positive=np.sqrt(e[1])*U[:,1];negative=np.sqrt(-e[0])*U[:,0]
  A.append(positive+negative);B.append(positive-negative)
 W=rng.normal(size=(3,3));assert np.linalg.matrix_rank(W)==3
 return normalize(np.r_[np.array(A).ravel(),np.array(B).ravel(),W.ravel()])

def feature_metrics(v,target):
 def features(p):return np.array([np.convolve(p[:4].reshape(2,2)[i],p[4:8].reshape(2,2)[i]) for i in range(2)])
 f=features(v)@Lhidden;t=features(target)@Lhidden
 cos=np.abs(f@t.T)/np.outer(np.linalg.norm(f,axis=1),np.linalg.norm(t,axis=1))
 ordered=np.diag(cos);matched=ordered if np.trace(cos)>=cos[0,1]+cos[1,0] else np.array([cos[0,1],cos[1,0]])
 def projector(a):u,s,vh=np.linalg.svd(a,full_matrices=False);rank=int(np.sum(s>s[0]*1e-10));return vh[:rank].T@vh[:rank]
 return dict(ordered_cosines=ordered.tolist(),matched_cosines=matched.tolist(),span_projector_distance=float(np.linalg.norm(projector(f)-projector(t))))
