"""Integer polynomial Jacobian witness for the shared d8,h4,T12 model."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from experiment import multiply


def modular_rank(A,prime):
    M=A.copy()%prime;row=0;pivots=[]
    for col in range(M.shape[1]):
        hits=np.flatnonzero(M[row:,col])
        if not len(hits):continue
        pivot=row+int(hits[0]);M[[row,pivot]]=M[[pivot,row]]
        M[row]=M[row]*pow(int(M[row,col]),-1,prime)%prime
        if row+1<len(M):M[row+1:]=(M[row+1:]-M[row+1:,col,None]*M[row][None,:])%prime
        pivots.append(col);row+=1
        if row==len(M):break
    return row,pivots


def main():
    root=Path(__file__).parent;out=root/'results/generic_rank';out.mkdir(parents=True,exist_ok=False)
    d,h,tasks=8,4,12
    unit=[tuple(int(j==k) for j in range(d)) for k in range(d)]
    keys=sorted({tuple(sum(i==j for i in inds) for j in range(d)) for inds in itertools.combinations_with_replacement(range(d),4)})
    lookup={k:i for i,k in enumerate(keys)};pairs=list(itertools.combinations_with_replacement(range(h),2))
    features=[multiply({unit[2*i]:1},{unit[2*i+1]:1}) for i in range(h)]
    products=[multiply(features[i],features[j]) for i,j in pairs]
    W=np.zeros((tasks,len(pairs)),dtype=np.int64);W[:10]=np.eye(10,dtype=np.int64);W[10]=1;W[11]=np.arange(1,11)
    parameters=2*h*d+tasks*len(pairs);J=np.zeros((tasks*len(keys),parameters),dtype=np.int64)
    # Parameter ordering A(h,d), B(h,d), W(T,p).
    for which in range(2):
        for i in range(h):
            other={unit[2*i+(1 if which==0 else 0)]:1}
            for k in range(d):
                dh=multiply({unit[k]:1},other);column=which*h*d+i*d+k
                for p,(left,right) in enumerate(pairs):
                    terms=[]
                    if left==i:terms.append(multiply(dh,features[right]))
                    if right==i:terms.append(multiply(features[left],dh))
                    for poly in terms:
                        for key,value in poly.items():
                            for t in range(tasks):J[t*len(keys)+lookup[key],column]+=W[t,p]*value
    for t in range(tasks):
        for p,poly in enumerate(products):
            for key,value in poly.items():J[t*len(keys)+lookup[key],2*h*d+t*len(pairs)+p]=value
    # Eight exact independent tangent vectors for individual A/B row scalings.
    gauge=np.zeros((parameters,2*h),dtype=np.int64)
    for which in range(2):
        for i in range(h):
            col=which*h+i;gauge[which*h*d+i*d+2*i+which,col]=1
            for p,(left,right) in enumerate(pairs):
                multiplicity=int(left==i)+int(right==i)
                for t in range(tasks):gauge[2*h*d+t*len(pairs)+p,col]=-multiplicity*W[t,p]
    assert np.all(J@gauge==0)
    assert modular_rank(gauge.T,1000003)[0]==8
    ranks={};pivot_lists={}
    for prime in [1000003,1000033]:
        rank,pivots=modular_rank(J.T,prime);assert rank==176;ranks[str(prime)]=rank;pivot_lists[str(prime)]=pivots
    # Exact integer witness has 184 parameters, 8 independent null directions,
    # and a nonzero 176-minor modulo a prime, hence real/rational rank exactly176.
    np.savez_compressed(out/'witness.npz',jacobian=J,gauge=gauge,keys=np.array(keys),head_weights=W)
    result=dict(status='passed',d=d,h=h,tasks=tasks,parameters=parameters,coefficient_outputs=J.shape[0],
                exact_gauge_dimension=8,modular_ranks=ranks,pivot_columns=pivot_lists,
                implication='nonzero polynomial Jacobian minor witnesses generic rank176; generic fiber dimension8',
                local_regular_quotient_rlct=88,rlct_conditions='realizable independent Gaussian output noise, nonsingular Gaussian inputs, generic teacher, smooth positive prior in a compact local chart',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                witness_sha256=hashlib.sha256((out/'witness.npz').read_bytes()).hexdigest())
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='pivot_columns'})


if __name__=='__main__':main()
