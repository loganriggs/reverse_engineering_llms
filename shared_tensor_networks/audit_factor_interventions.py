"""Independent quadrature replay of moment-based factor-intervention metrics."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np

def main():
    root=Path(__file__).parent;out=root/'results/fixed_degree_recovery';result=json.loads((out/'interventions.json').read_text());z,w=np.polynomial.hermite_e.hermegauss(5);ix=np.array(list(itertools.product(range(5),repeat=3)));x=z[ix];weights=np.prod(w[ix]/np.sqrt(2*np.pi),axis=1);worst=0
    for r in result['rows']:
        p=root/'results/fixed_degree_reuse'/f"t4_s{r['seed']}_independent_g4.npz";data=np.load(p);t=data['teacher'];s=data[r['stage']];tu=t/np.linalg.norm(t,axis=1,keepdims=True);su=s/np.linalg.norm(s,axis=1,keepdims=True)
        perms=list(itertools.permutations(range(4)));permutation=max(perms,key=lambda perm:np.mean(abs(np.sum(tu*su[list(perm)],axis=1))));su=su[list(permutation)];sign=np.sign(np.sum(tu*su,axis=1));su*=sign[:,None]
        ta=np.prod(np.linalg.norm(t,axis=1))/data['teacher_scale'];sa=np.prod(np.linalg.norm(s,axis=1))/data['student_scale']*np.prod(sign)
        comparisons=[]
        for i in range(4):
            yt=ta*np.prod(x@np.delete(tu,i,axis=0).T,axis=1);ys=sa*np.prod(x@np.delete(su,i,axis=0).T,axis=1);comparisons.append((float(weights@(yt-ys)**2/(weights@yt**2)),r['canonical_single_factor_deletion_relative_mse'][i]))
        for var,key in [([1,1,1],'standard_relative_mse'),([.5,1,2],'shifted_covariance_relative_mse')]:
            xx=x*np.sqrt(var);yt=ta*np.prod(xx@tu.T,axis=1);ys=sa*np.prod(xx@su.T,axis=1);comparisons.append((float(weights@(yt-ys)**2/(weights@yt**2)),r[key]))
        for actual,expected in comparisons:
            err=abs(actual-expected)/max(abs(expected),1e-15);assert err<1e-10;worst=max(worst,err)
    (out/'intervention_audit.json').write_text(json.dumps(dict(status='passed',comparisons=36,max_relative_error=worst,input_sha256=hashlib.sha256((out/'interventions.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n');print('Passed36independent quadrature comparisons; max relative error',worst)
if __name__=='__main__':main()
