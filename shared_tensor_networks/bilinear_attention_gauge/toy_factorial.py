"""Known-rule two-head attention: metric x legibility-objective pilot.

This is an exact first-layer pair-function toy, not a trained language model.
Each head has two competing QK1 directions; rank one forces a choice. The
query/key positions are distinct, so every token pair is realizable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla


VOCAB = ("D0", "D1", "N0", "N1", "V0", "V1", "A0", "A1", "R0", "R1", "SEP", "PAD")
V = len(VOCAB)
HEADS = 2
D = 2
OUT = 2


def toy(seed: int = 0):
    rng = np.random.default_rng(seed)
    # Each column is a known query-key rule. Both heads have a common and a
    # rare rule; the second head uses different query/key categories.
    rules = [((4, 5), (2, 3), (8, 9), (0, 1)),
             ((2, 3), (0, 1), (8, 9), (6, 7))]
    heads = []
    for h, (common_q, common_k, rare_q, rare_k) in enumerate(rules):
        u = rng.normal(scale=.025, size=(V, D))
        v = rng.normal(scale=.025, size=(V, D))
        u[list(common_q), 0] += 1
        v[list(common_k), 0] += 1
        u[list(rare_q), 1] += 1.8
        v[list(rare_k), 1] += 1.8
        # QK2 is a separate, nonconstant selection path. Its intervention is
        # replacement by the constant score 1, not an edit to QK1.
        q2 = rng.normal(scale=.08, size=(V, 3))
        k2 = rng.normal(scale=.08, size=(V, 3))
        q2[:, 0] += 1
        k2[:, 0] += 1
        q2[list(common_q), 1] += .8
        k2[list(common_k), 1] += .8
        q2[list(rare_q), 2] += 1.2
        k2[list(rare_k), 2] += 1.2
        b = q2 @ k2.T
        c = rng.normal(scale=.03, size=(V, OUT))
        c[list(common_k), h] += 1
        c[list(rare_k), 1-h] += 1
        heads.append(dict(u=u, v=v, b=b, c=c,
                          common_q=common_q, common_k=common_k,
                          rare_q=rare_q, rare_k=rare_k))
    return heads


def metric(name: str):
    if name == "uniform":
        p = np.ones(V)/V
    elif name == "usage":
        p = np.array([.17, .17, .15, .15, .11, .11, .055, .055,
                      .005, .005, .005, .005])
        p /= p.sum()
    elif name == "hybrid":
        return .5*metric("uniform") + .5*metric("usage")
    else:
        raise ValueError(name)
    return np.outer(p,p)


def table(head, x, intervention=False):
    score = head["u"] @ x @ head["v"].T
    b = np.ones_like(head["b"]) if intervention else head["b"]
    return (score*b)[:, :, None]*head["c"][None, :, :]


def environment(head, measure, intervention=False):
    b = np.ones_like(head["b"]) if intervention else head["b"]
    weights=b*b*measure*np.sum(head["c"]**2,axis=-1)[None,:]
    return np.einsum("ar,at,bs,bu,ab->rstu",head["u"],head["u"],
                     head["v"],head["v"],weights,optimize=True).reshape(D*D,D*D)


def weighted_trace_gauge(env, rank):
    """Small real-bond WTG, same equations as run.py without loading PyTorch."""
    d=int(np.sqrt(env.shape[0]))
    transfer=env.reshape(d,d,d,d).transpose(0,2,1,3).reshape(d*d,d*d)
    def dominant(t):
        vals,vecs=sla.eig(t)
        op=vecs[:,np.argmax(np.abs(vals))].real.reshape(d,d)
        op=(op+op.T)/2
        if np.trace(op)<0: op=-op
        return op/np.trace(op)
    left,right=dominant(transfer),dominant(transfer.T)
    for op in (left,right):
        if np.min(sla.eigvalsh(op))<=1e-12:
            raise ValueError("Nonpositive WTG boundary")
    def root(op):
        vals,vecs=sla.eigh(op)
        return (vecs*np.sqrt(vals))@vecs.T
    l,r=root(left),root(right)
    wl,s,wrt=sla.svd(l@r)
    gx,gy=wl.T@l,r@wrt.T
    cut=sla.solve(gx,np.diag(np.r_[s[:rank],np.zeros(d-rank)]))@sla.inv(gy)
    return cut,s,gx,gy


def env_error(env,x):
    delta=(x-np.eye(D)).ravel()
    identity=np.eye(D).ravel()
    return float(np.sum(env*np.outer(delta,delta))/
                 np.sum(env*np.outer(identity,identity)))


def relative_loss(head, x, measure, intervention=False):
    original = table(head,np.eye(D),intervention)
    altered = table(head,x,intervention)
    return float(np.sum(measure[:,:,None]*(altered-original)**2)
                 /np.sum(measure[:,:,None]*original**2))


def rule_recall(head, x, which):
    q = head[which+"_q"]
    k = head[which+"_k"]
    original = table(head,np.eye(D))[np.ix_(q,k)]
    altered = table(head,x)[np.ix_(q,k)]
    return float(np.sum((altered-original)**2)/np.sum(original**2))


def fit(head, measure, objective, seed, lam=.02):
    env = environment(head,measure)
    _, s, gx, gy = weighted_trace_gauge(env,1)
    canonical = np.linalg.solve(gx,np.diag([s[0],0.])) @ np.linalg.inv(gy)
    uu, ss, vh = np.linalg.svd(canonical,full_matrices=False)
    initial = np.r_[uu[:,0]*np.sqrt(ss[0]), vh[0]*np.sqrt(ss[0])]
    rng = np.random.default_rng(seed+817)
    original = table(head,np.eye(D))
    normalizer = np.sum(measure[:,:,None]*original**2)
    cf_env = environment(head,measure,True)
    cf_scale = float(np.sum(cf_env*np.outer(np.eye(D).ravel(),np.eye(D).ravel())))
    identity = np.eye(D).ravel()
    scale = float(np.sum(env*np.outer(identity,identity)))

    def loss_and_grad(z):
        a, bb = z[:D], z[D:]
        x=np.outer(a,bb)
        delta=(x-np.eye(D)).ravel()
        main=np.sum(env*np.outer(delta,delta))/scale
        grad=(2*(env@delta)/scale).reshape(D,D)
        if objective == "fidelity":
            prior=0.
            prior_grad=np.zeros_like(x)
        elif objective == "native_l1":
            prior=np.sum(np.sqrt(x*x+1e-8))/D # gauge-dependent weight cost
            prior_grad=x/np.sqrt(x*x+1e-8)/D
        elif objective == "canonical_l1":
            xc=gx@x@gy
            prior=np.sum(np.sqrt(xc*xc+1e-8))/max(np.sum(np.abs(gx@gy)),1e-12)
            prior_grad=gx.T@(xc/np.sqrt(xc*xc+1e-8))@gy.T/max(np.sum(np.abs(gx@gy)),1e-12)
        elif objective == "pair_l1":
            t=table(head,x)
            magnitude=np.sqrt(np.sum(t*t,axis=-1)+1e-8)
            prior=np.sum(measure*magnitude)/np.sqrt(normalizer)
            weighted=measure*np.sum(t*(head["b"][:,:,None]*head["c"][None,:,:]),axis=-1)/magnitude
            prior_grad=head["u"].T@weighted@head["v"]/np.sqrt(normalizer)
        elif objective == "intervention":
            prior=np.sum(cf_env*np.outer(delta,delta))/cf_scale
            prior_grad=(2*(cf_env@delta)/cf_scale).reshape(D,D)
        else:
            raise ValueError(objective)
        grad += lam*prior_grad
        # Removes the irrelevant a -> ca, b -> b/c flat direction.
        balance=np.dot(a,a)-np.dot(bb,bb)
        penalty=1e-7*balance**2
        g_a=grad@bb+4e-7*balance*a
        g_b=grad.T@a-4e-7*balance*bb
        return float(main+lam*prior+penalty),np.r_[g_a,g_b]

    probe_z=np.array([.7,-.2,.3,.9])
    analytic=loss_and_grad(probe_z)[1]
    numeric=[]
    for k in range(2*D):
        direction=np.eye(2*D)[k]*1e-6
        numeric.append((loss_and_grad(probe_z+direction)[0]-
                        loss_and_grad(probe_z-direction)[0])/(2e-6))
    gradient_check=float(np.max(np.abs(analytic-numeric)))
    assert gradient_check<1e-6,(objective,gradient_check)

    starts=[initial]+[rng.normal(size=2*D) for _ in range(3)]
    best=(float("inf"),None,0,float("inf"))
    for start in starts:
        z=start.copy()
        first=np.zeros_like(z)
        second=np.zeros_like(z)
        for step in range(1,2001):
            value,gradient=loss_and_grad(z)
            gnorm=float(np.linalg.norm(gradient))
            if value<best[0]:
                best=(value,z.copy(),step,gnorm)
            if gnorm<1e-7:
                break
            first=.9*first+.1*gradient
            second=.999*second+.001*gradient**2
            z-=.03*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
            z=np.clip(z,-10,10)
    x=np.outer(best[1][:D],best[1][D:])
    return x,canonical,dict(success=best[3]<1e-4,iterations=best[2],
                             objective_loss=best[0],gradient_norm=best[3],
                             gradient_check_max_abs=gradient_check)


def run(seed=0):
    heads=toy(seed)
    probe=heads[0]
    gauge=np.array([[2.,.8],[0.,.5]])
    invg=np.linalg.inv(gauge)
    probe_x=np.array([[1.,.25],[.1,0.]])
    gauged=dict(probe,u=probe["u"]@gauge,v=probe["v"]@invg.T)
    gauged_x=invg@probe_x@gauge
    gauge_check=dict(function_max_abs=float(np.max(np.abs(table(probe,probe_x)-
                                                       table(gauged,gauged_x)))),
                     bond_entry_l1_before=float(np.sum(np.abs(probe_x))),
                     bond_entry_l1_after=float(np.sum(np.abs(gauged_x))),
                     pair_l1_before=float(np.sum(np.linalg.norm(table(probe,probe_x),axis=-1))),
                     pair_l1_after=float(np.sum(np.linalg.norm(table(gauged,gauged_x),axis=-1))))
    assert gauge_check["function_max_abs"]<1e-12
    _,_,g1l,g1r=weighted_trace_gauge(environment(probe,metric("uniform")),1)
    _,_,g2l,g2r=weighted_trace_gauge(environment(gauged,metric("uniform")),1)
    canonical_l1=lambda gl,xx,gr: np.sum(np.abs(gl@xx@gr))/np.sum(np.abs(gl@gr))
    gauge_check["canonical_relative_l1_before"]=float(canonical_l1(g1l,probe_x,g1r))
    gauge_check["canonical_relative_l1_after"]=float(canonical_l1(g2l,gauged_x,g2r))
    assert abs(gauge_check["canonical_relative_l1_before"]-
               gauge_check["canonical_relative_l1_after"])<1e-10
    records=[]
    for name in ("uniform","usage","hybrid"):
        m=metric(name)
        for objective in ("fidelity","native_l1","canonical_l1","pair_l1","intervention"):
            xs=[]
            for h,head in enumerate(heads):
                x,canonical,info=fit(head,m,objective,seed)
                xs.append(x)
                records.append(dict(metric=name,objective=objective,head=h,
                    train_mse=relative_loss(head,x,m),
                    uniform_mse=relative_loss(head,x,metric("uniform")),
                    usage_mse=relative_loss(head,x,metric("usage")),
                    intervention_mse=relative_loss(head,x,metric("uniform"),True),
                    common_rule_mse=rule_recall(head,x,"common"),
                    rare_rule_mse=rule_recall(head,x,"rare"),
                    canonical_train_mse=relative_loss(head,canonical,m),
                    **info))
            target=sum(table(head,np.eye(D)) for head in heads)
            approx=sum(table(head,x) for head,x in zip(heads,xs))
            for eval_name in ("uniform","usage"):
                em=metric(eval_name)
                full=float(np.sum(em[:,:,None]*(target-approx)**2)
                           /np.sum(em[:,:,None]*target**2))
                records.append(dict(metric=name,objective=objective,head="sum",
                                    eval_metric=eval_name,full_mse=full))
    return dict(seed=seed,vocab=VOCAB,gauge_check=gauge_check,
                scope="exact two-position, two-head known-rule toy; rank-one per head",
                notes="Bond-entry L1 depends on the chosen QK1 coordinates; pair L1 is function based.",
                records=records)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=run(args.seed)
    output=args.output or Path(__file__).with_name(f"toy_factorial_seed{args.seed}.json")
    output.write_text(json.dumps(result,indent=2)+"\n")
    for r in result["records"]:
        if r["head"] == "sum":
            print(r)
