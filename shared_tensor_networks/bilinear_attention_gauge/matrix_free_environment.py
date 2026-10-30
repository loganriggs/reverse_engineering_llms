"""Matrix-free bond environment and WTG boundaries for sampled pair functions.

Given sample rows u_n, v_n and nonnegative weights w_n, the represented
scalar is u_n^T X v_n. Any QK2 and OV factors are folded into w_n. No
(d^2 x d^2) environment or transfer matrix is materialized.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.linalg as sla


class SampledBondEnvironment:
    def __init__(self,u,v,w):
        self.u=np.asarray(u,dtype=np.float64)
        self.v=np.asarray(v,dtype=np.float64)
        self.w=np.asarray(w,dtype=np.float64)
        assert self.u.shape==self.v.shape
        assert self.w.shape==(len(self.u),)
        assert np.all(np.isfinite(self.u)) and np.all(np.isfinite(self.v))
        assert np.all(np.isfinite(self.w)) and np.all(self.w>=0)
        assert np.sum(self.w)>0
        self.d=self.u.shape[1]

    def gram_apply(self,x):
        score=np.sum((self.u@x)*self.v,axis=1)
        return self.u.T@((self.w*score)[:,None]*self.v)

    def transfer_apply(self,op,transpose=False):
        source,target=(self.u,self.v) if transpose else (self.v,self.u)
        contraction=np.sum((source@op)*source,axis=1)
        return target.T@((self.w*contraction)[:,None]*target)

    def relative_error(self,x):
        delta=np.eye(self.d)-x
        numerator=np.sum(delta*self.gram_apply(delta))
        denominator=np.sum(np.eye(self.d)*self.gram_apply(np.eye(self.d)))
        return float(numerator/denominator)

    def boundary(self,transpose=False,max_iter=200,tol=1e-10):
        op=np.eye(self.d)/self.d
        residual=float("inf")
        for iteration in range(1,max_iter+1):
            next_op=self.transfer_apply(op,transpose)
            next_op=(next_op+next_op.T)/2
            scale=np.trace(next_op)
            assert scale>0 and np.isfinite(scale)
            next_op/=scale
            residual=float(np.linalg.norm(next_op-op)/np.linalg.norm(next_op))
            op=next_op
            if residual<tol:
                break
        # A small iterate difference is not enough if the transfer is nearly
        # defective; verify the actual eigenoperator equation as well.
        mapped=self.transfer_apply(op,transpose)
        eigen_residual=float(np.linalg.norm(mapped-np.trace(mapped)*op)/
                             np.linalg.norm(mapped))
        return op,dict(iterations=iteration,iterate_delta=residual,
                       eigen_residual=eigen_residual)

    def weighted_trace_gauge(self,rank,max_iter=200,tol=1e-10):
        left,li=self.boundary(False,max_iter,tol)
        right,ri=self.boundary(True,max_iter,tol)
        if max(li["eigen_residual"],ri["eigen_residual"])>max(1e-7,10*tol):
            raise ValueError("WTG boundaries did not converge")
        def root(op):
            values,vectors=sla.eigh(op)
            if values.min()<=1e-12:
                raise ValueError("WTG boundary is singular on this domain")
            return (vectors*np.sqrt(values))@vectors.T
        l,r=root(left),root(right)
        wl,s,wrt=sla.svd(l@r)
        gx,gy=wl.T@l,r@wrt.T
        cut=sla.solve(gx,np.diag(np.r_[s[:rank],np.zeros(self.d-rank)]))@sla.inv(gy)
        return cut,dict(left=li,right=ri,singular=s.tolist(),
                        left_condition=float(np.linalg.cond(left)),
                        right_condition=float(np.linalg.cond(right)))


def check(seed=0):
    rng=np.random.default_rng(seed)
    n,d=300,4
    u=rng.normal(size=(n,d))
    v=rng.normal(size=(n,d))
    w=np.exp(rng.normal(scale=.3,size=n))/n
    env=SampledBondEnvironment(u,v,w)
    dense=np.einsum("nr,ns,nt,nu,n->rstu",u,v,u,v,w,optimize=True).reshape(d*d,d*d)
    x=rng.normal(size=(d,d))
    gram_diff=float(np.max(np.abs(env.gram_apply(x).ravel()-dense@x.ravel())))
    assert gram_diff<1e-10,gram_diff
    transfer=dense.reshape(d,d,d,d).transpose(0,2,1,3).reshape(d*d,d*d)
    op=rng.normal(size=(d,d));op=(op+op.T)/2
    transfer_diff=float(np.max(np.abs(env.transfer_apply(op).ravel()-transfer@op.ravel())))
    assert transfer_diff<1e-10,transfer_diff
    transpose_diff=float(np.max(np.abs(env.transfer_apply(op,True).ravel()-
                                       transfer.T@op.ravel())))
    assert transpose_diff<1e-10,transpose_diff
    from toy_factorial import weighted_trace_gauge
    dense_cut,_,_,_=weighted_trace_gauge(dense,2)
    implicit_cut,info=env.weighted_trace_gauge(2)
    cut_diff=float(np.max(np.abs(implicit_cut-dense_cut)))
    assert cut_diff<1e-8,cut_diff
    gauge=sla.expm(.12*rng.normal(size=(d,d)))
    invg=sla.inv(gauge)
    gauged=SampledBondEnvironment(u@gauge,v@invg.T,w)
    gauged_cut,_=gauged.weighted_trace_gauge(2)
    original_scores=np.sum((u@implicit_cut)*v,axis=1)
    gauged_scores=np.sum(((u@gauge)@gauged_cut)*(v@invg.T),axis=1)
    gauge_cut_diff=float(np.max(np.abs(original_scores-gauged_scores)))
    assert gauge_cut_diff<1e-8,gauge_cut_diff
    result=dict(seed=seed,n=n,d=d,gram_max_abs=gram_diff,
                transfer_max_abs=transfer_diff,transpose_max_abs=transpose_diff,
                cut_max_abs=cut_diff,gauge_cut_score_max_abs=gauge_cut_diff,
                relative_error=env.relative_error(implicit_cut),
                boundary=info)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--check",action="store_true")
    p.add_argument("--seed",type=int,default=0)
    args=p.parse_args()
    if args.check:
        print(json.dumps(check(args.seed),indent=2))
