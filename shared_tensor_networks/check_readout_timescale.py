"""Finite-time two-timescale flow check; protocol in READOUT_TIMESCALE_PLAN.md."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
from normalized_readout import normalized_design

OUT = Path('results/readout_timescale')
RIDGE = .1
HIDDEN, DIM, TASKS, N = 2, 3, 3, 32
K, P = 12, 3
GRID = np.linspace(0, .5, 101)


def state(theta, U, x, y):
    phi, D, radial, Jrad, _ = normalized_design(theta, x, HIDDEN)
    phi, D, y = phi/np.sqrt(N), D/np.sqrt(N), y/np.sqrt(N)
    H = phi.T@phi + RIDGE*np.eye(P)
    star = np.linalg.solve(H, phi.T@y)
    residual = phi@U-y
    gt = np.einsum('nt,npk,pt->k', residual, D, U)+Jrad.T@radial
    gu = phi.T@residual+RIDGE*U
    loss = .5*(np.sum(residual**2)+RIDGE*np.sum(U**2)+radial@radial)
    return gt, gu, star, H, phi, D, y, loss


def star_direction(theta, v, x, y):
    _, _, star, H, phi, D, yy, _ = state(theta, np.zeros((P,TASKS)), x,y)
    dp = np.einsum('npk,k->np', D, v)
    return np.linalg.solve(H, dp.T@(yy-phi@star)-phi.T@dp@star)


def integrate(fun, initial, tight=False):
    sol = solve_ivp(fun, (0,.5), initial, method='Radau', t_eval=GRID,
                    rtol=1e-11 if tight else 1e-9, atol=1e-13 if tight else 1e-11)
    assert sol.success and np.isfinite(sol.y).all(), sol.message
    return sol


def independent_gradient(theta,U,x,y):
    v=torch.tensor(np.r_[theta,U.ravel()],dtype=torch.float64,requires_grad=True)
    raw=v[:K].reshape(4,3);norm=torch.linalg.vector_norm(raw,dim=1);unit=raw/norm[:,None]
    xx=torch.tensor(x,dtype=torch.float64)
    z=(xx@unit[:2].T)*(xx@unit[2:].T)
    phi=torch.stack([z[:,i]*z[:,j] for i,j in itertools.combinations_with_replacement(range(2),2)],1)
    u=v[K:].reshape(3,3)
    loss=.5*((phi@u-torch.tensor(y)).square().sum()/N+RIDGE*u.square().sum()+(norm-1).square().sum())
    return torch.autograd.grad(loss,v)[0].numpy()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[];checks=[]
    for seed in range(247101,247104):
        rng=np.random.default_rng(seed);x=rng.normal(size=(N,DIM))
        teacher=rng.normal(size=(4,3));teacher/=np.linalg.norm(teacher,axis=1)[:,None]
        W=rng.normal(size=(P,TASKS));phi=normalized_design(teacher.ravel(),x,2)[0];y=phi@W
        theta=rng.normal(size=(4,3));theta/=np.linalg.norm(theta,axis=1)[:,None];theta=theta.ravel()
        star=state(theta,np.zeros((P,TASKS)),x,y)[2];offset=rng.normal(size=star.shape)
        def reduced(t,v):
            u=state(v,np.zeros((P,TASKS)),x,y)[2]
            return -state(v,u,x,y)[0]
        ref=integrate(reduced,theta);ref_tight=integrate(reduced,theta,True)
        ref_error=np.max(np.abs(ref.y-ref_tight.y));assert ref_error<1e-7
        np.savez_compressed(OUT/f'data_{seed}.npz',x=x,y=y,teacher=teacher,W=W,theta=theta,offset=offset,
                            times=GRID,reduced=ref.y,reduced_tight=ref_tight.y)
        for init in ['optimal','perturbed']:
            initial=np.r_[theta,(star+(offset if init=='perturbed' else 0)).ravel()]
            for eps in [.1,.03,.01,.003,.001]:
                def flow(t,v):
                    gt,gu,*_=state(v[:K],v[K:].reshape(P,TASKS),x,y)
                    return np.r_[-gt,(-gu/eps).ravel()]
                sol=integrate(flow,initial);tight=integrate(flow,initial,True)
                replay=np.max(np.abs(sol.y-tight.y));assert replay<1e-7,replay
                tracking=[];losses=[];forcing=[];identities=[]
                for v in sol.y.T:
                    gt,gu,us,H,*_,loss=state(v[:K],v[K:].reshape(P,TASKS),x,y)
                    du=star_direction(v[:K],-gt,x,y)
                    e=v[K:].reshape(P,TASKS)-us
                    identities.append(np.linalg.norm(gu-H@e))
                    tracking.append(np.linalg.norm(e));forcing.append(np.linalg.norm(du));losses.append(loss)
                assert max(identities)<1e-9
                assert max(np.diff(losses))<1e-8
                for vi in [0,-1]:
                    v=sol.y[:,vi];u=v[K:].reshape(P,TASKS);gt,gu,*_=state(v[:K],u,x,y)
                    analytic=np.r_[gt,gu.ravel()];ind=independent_gradient(v[:K],u,x,y)
                    gerr=np.linalg.norm(analytic-ind)/max(1,np.linalg.norm(ind));assert gerr<1e-9
                    direction=-gt;delta=1e-5/max(1,np.linalg.norm(direction))
                    plus=state(v[:K]+delta*direction,u,x,y)[2]
                    minus=state(v[:K]-delta*direction,u,x,y)[2]
                    exact=star_direction(v[:K],direction,x,y)
                    derr=np.linalg.norm((plus-minus)/(2*delta)-exact)/max(1,np.linalg.norm(exact));assert derr<1e-7
                    checks.append(dict(seed=seed,initialization=init,epsilon=eps,endpoint=vi,gradient_error=gerr,star_derivative_error=derr))
                name=f'{seed}_{init}_{eps:g}.npz'
                np.savez_compressed(OUT/name,times=GRID,path=sol.y,tight_path=tight.y,tracking=tracking,loss=losses,forcing=forcing)
                records.append(dict(seed=seed,initialization=init,epsilon=eps,trajectory=name,
                    hidden_endpoint_distance=float(np.linalg.norm(sol.y[:K,-1]-ref.y[:,-1])),
                    hidden_max_distance=float(np.linalg.norm(sol.y[:K]-ref.y,axis=0).max()),
                    final_tracking=float(tracking[-1]),initial_tracking=float(tracking[0]),
                    max_sampled_forcing=float(max(forcing)),final_loss=float(losses[-1]),
                    replay_max_abs=float(replay),reference_replay_max_abs=float(ref_error),nfev=sol.nfev))
                print(records[-1],flush=True)
        (OUT/'results.json').write_text(json.dumps(records,indent=2)+'\n')
        (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert len(records)==30 and len(checks)==60
    (OUT/'completed.json').write_text(json.dumps(dict(fits=len(records),gradient_and_derivative_checks=len(checks)))+'\n')


if __name__=='__main__':main()
