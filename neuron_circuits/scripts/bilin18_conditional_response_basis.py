"""Training-statistic conditional reduced-rank feature construction."""
import numpy as np

def conditional_statistics(g,r,a,b):
    gaa=a.T@g@a;gab=a.T@g@b;ra=a.T@r
    s=b.T@g@b-gab.T@np.linalg.solve(gaa,gab)
    t=b.T@r-gab.T@np.linalg.solve(gaa,ra)
    return (s+s.T)/2,t

def directions(s,t,b,beta,rank):
    ev,q=np.linalg.eigh(s);floor=max(ev[-1]*1e-12,0);regularization=beta*np.trace(s)/len(s);inverse=1/np.sqrt(np.maximum(ev,floor)+regularization);whitener=(q*inverse)@q.T;u,sv,_=np.linalg.svd(whitener@t,full_matrices=False);raw=b@whitener@u[:,:rank];basis=np.linalg.qr(raw)[0]
    return basis,{'minimum_eigenvalue':float(ev[0]),'maximum_eigenvalue':float(ev[-1]),'floor':float(floor),'ridge':float(regularization),'singular_values':sv[:rank].tolist()}
