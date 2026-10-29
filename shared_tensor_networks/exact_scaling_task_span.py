"""Exact Gaussian quartic feature Gram via105 Wick pairings, no input expansion."""
import itertools,json,math
from pathlib import Path
import numpy as np
import torch
from deep_multitask import DeepMultiTask


def pairings(indices):
    if not indices:
        yield [];return
    a=indices[0]
    for j in range(1,len(indices)):
        for rest in pairings(indices[1:j]+indices[j+1:]):yield [(a,indices[j])]+rest


def feature_gram(A,B,scale):
    i,j=np.triu_indices(len(A));F=np.stack([A[i]/scale[i,None],B[i],A[j]/scale[j,None],B[j]],axis=1)
    n=len(F);cov={}
    for a in range(8):
        for b in range(a+1,8):
            if b<4:cov[a,b]=np.sum(F[:,a]*F[:,b],axis=1)[:,None]
            elif a>=4:cov[a,b]=np.sum(F[:,a-4]*F[:,b-4],axis=1)[None,:]
            else:cov[a,b]=F[:,a]@F[:,b-4].T
    G=np.zeros((n,n));terms=list(pairings(tuple(range(8))));assert len(terms)==105
    for pairs in terms:
        v=np.ones((n,n))
        for pair in pairs:v*=cov[pair]
        G+=v
    return (G+G.T)/2


def main():
    root=Path(__file__).parent;out=root/'results/scaling_grid';rng=np.random.default_rng(146000)
    # Independent tensor Gauss-Hermite integration of degree8 products.
    a=rng.normal(size=(3,4));b=rng.normal(size=(3,4));scale=np.array([.7,1.4,2.])
    nodes,weights=np.polynomial.hermite.hermgauss(5);indices=np.array(list(itertools.product(range(5),repeat=4)))
    x=np.sqrt(2)*nodes[indices];w=np.prod(weights[indices]/np.sqrt(np.pi),axis=1)
    h=(x@a.T)*(x@b.T)/scale;i,j=np.triu_indices(3);f=h[:,i]*h[:,j]
    exact=feature_gram(a,b,scale);quadrature=f.T@(w[:,None]*f)
    discrepancy=float(np.max(np.abs(exact-quadrature))/np.max(np.abs(exact)));assert discrepancy<1e-12
    rows=[]
    for gi,d,h in [(0,16,8),(2,32,16)]:
        for rep in range(3):
            key=f'g{gi}_shared_r{rep}';model=DeepMultiTask(d,h,16,2,True,0);model.load_state_dict(torch.load(out/key/'teacher.pt',weights_only=True))
            a=model.A[0].detach().numpy();b=model.B[0].detach().numpy();scale=model.scale_0.numpy();W=model.W.detach().numpy()
            G=feature_gram(a,b,scale);K=W@G@W.T
            eig=np.linalg.eigvalsh(G);assert eig.min()>-1e-10*eig.max()
            beta=np.linalg.solve(K[:12,:12],K[:12,12:]);residual=K[12:,12:]-K[12:,:12]@beta
            floor=np.diag(residual)/np.diag(K[12:,12:]);assert np.all((floor>=-1e-10)&(floor<=1+1e-10))
            np.savez_compressed(out/key/'exact_oracle_gram.npz',feature_gram=G,task_gram=K,population_coefficients=beta)
            rows.append(dict(key=key,d=d,h=h,hidden_product_dimension=h*(h+1)//2,source_head_rank=int(np.linalg.matrix_rank(W[:12])),all_head_rank=int(np.linalg.matrix_rank(W)),oracle_relative_projection_floor=floor.tolist(),mean_floor=float(floor.mean()),source_gram_condition=float(np.linalg.cond(K[:12,:12]))))
    result=dict(status='passed',pairings=105,independent_quadrature_relative_error=discrepancy,
                scope='shared depth2 teachers only; exact arithmetic Gaussian moment identities evaluated in float64; same relative projection floor on sphere',rows=rows)
    (out/'exact_task_span.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
