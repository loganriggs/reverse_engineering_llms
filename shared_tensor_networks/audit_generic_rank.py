"""Autograd evaluation and an independent exact minor determinant certificate."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import qr


def determinant_mod(matrix,p):
    M=matrix.tolist();n=len(M);det=1
    for i in range(n):
        pivot=next((j for j in range(i,n) if M[j][i]%p),None)
        if pivot is None:return 0
        if pivot!=i:M[i],M[pivot]=M[pivot],M[i];det=-det
        value=M[i][i]%p;det=det*value%p;inv=pow(value,-1,p)
        for j in range(i+1,n):
            factor=M[j][i]*inv%p
            for k in range(i+1,n):M[j][k]=(M[j][k]-factor*M[i][k])%p
            M[j][i]=0
    return det%p


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    root=Path(__file__).parent;out=root/'results/generic_rank';r=json.loads((out/'results.json').read_text())
    ev=np.load(out/'witness.npz');J=ev['jacobian'];G=ev['gauge'];keys=ev['keys'];W=ev['head_weights']
    rows=r['pivot_columns']['1000003'];_,_,columns=qr(J[rows].astype(float),pivoting=True,mode='economic');columns=columns[:176]
    minor=J[np.array(rows)[:,None],columns[None,:]]
    det=determinant_mod(minor,1000003);assert det!=0
    indices=[i*8+2*i for i in range(4)]+[32+i*8+2*i+1 for i in range(4)]
    np.testing.assert_array_equal(G[indices],np.eye(8,dtype=np.int64));assert np.all(J@G==0)
    A=np.eye(8)[[0,2,4,6]];B=np.eye(8)[[1,3,5,7]]
    theta=torch.from_numpy(np.concatenate([A.flatten(),B.flatten(),W.astype(float).flatten()])).requires_grad_()
    x=torch.randn(16,8,generator=torch.Generator().manual_seed(89000));i,j=torch.triu_indices(4,4)
    def fn(v):
        a=v[:32].reshape(4,8);b=v[32:64].reshape(4,8);w=v[64:].reshape(12,10)
        h=(x@a.T)*(x@b.T);return (h[:,i]*h[:,j])@w.T
    autodiff=torch.autograd.functional.jacobian(fn,theta,vectorize=True).detach().numpy()
    phi=np.column_stack([np.prod(x.numpy()**k,axis=1) for k in keys])
    exact=np.einsum('np,tpk->ntk',phi,J.reshape(12,330,184))
    error=float(np.linalg.norm(autodiff-exact)/np.linalg.norm(exact));assert error<1e-12
    result=dict(status='passed',minor_dimension=176,prime=1000003,minor_determinant_mod_prime=det,
                minor_rows=rows,minor_columns=columns.tolist(),independent_null_vectors=8,
                relative_autograd_jacobian_error=error,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                witness_sha256=hashlib.sha256((out/'witness.npz').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['minor_rows','minor_columns']})


if __name__=='__main__':main()
