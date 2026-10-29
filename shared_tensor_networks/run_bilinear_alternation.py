"""Layerwise alternating gradients on actual depth-two bilinear networks."""
import itertools,json,time
from pathlib import Path
import numpy as np
import torch
from experiment import Network
from run_svhn_standard import digest,write


def evaluate(p,x):
    left=(x@p['A'].T)*(x@p['B'].T)
    right=(x@p['AR'].T)*(x@p['BR'].T) if 'AR' in p else left
    return ((left@p['C'].T)*(right@p['D'].T))@p['w']


def gradients(p,x,weights,target,scale):
    ax,bx=x@p['A'].T,x@p['B'].T;left=ax*bx
    if 'AR' in p:
        ar,br=x@p['AR'].T,x@p['BR'].T;right=ar*br
    else:right=left
    u,v=left@p['C'].T,right@p['D'].T
    residual=weights*((u*v)@p['w']/scale-target)/scale
    du,dv=residual[:,None]*v*p['w'],residual[:,None]*u*p['w']
    g=dict(C=du.T@left,D=dv.T@right,w=(residual[:,None]*u*v).sum(0))
    dl,dr=du@p['C'],dv@p['D']
    if 'AR' in p:
        g['AR']=(dr*br).T@x;g['BR']=(dr*ar).T@x
    else:dl=dl+dr
    g['A']=(dl*bx).T@x;g['B']=(dl*ax).T@x
    return g


def blocks(p):
    return [['A','B']+(['AR','BR'] if 'AR' in p else []),['C','D'],['w']]


def update(p,x,weights,target,scale,lr,method,metric):
    p={k:v.copy() for k,v in p.items()}
    groups=blocks(p)
    schedule=[sum(groups,[])] if method=='joint' else groups if method=='forward' else groups[::-1]
    for group in schedule:
        g=gradients(p,x,weights,target,scale)
        for k in group:
            multiplicity=2 if metric=='corrected' and k in ['A','B'] else 1
            p[k]-=lr*g[k]/multiplicity
    return p


def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/bilinear_alternation';out.mkdir(exist_ok=True)
    protocol=dict(d=3,h=3,q=3,seeds=[211001,211002,211003],rates=[.01,.003],block_time=3,
        teacher_families=['tied','untied'],students=['tied_raw','tied_corrected','untied_raw'],methods=['joint','forward','reverse'],
        initialization='independent A/B rows; untied occurrences copied from tied model for matched initial functions',
        normalization='teacher and initial student each RMS1; frozen scales',
        objective='half exact Gaussian population MSE,5point-per-axis quadrature',
        blocks=['all bottom A/B occurrences','top C/D','readout w'],
        budget='same per-block time3,3 block components/sweep; alternating3 full gradients vs joint1; report work separately',
        source_sha256=digest(Path(__file__)),dependencies={'experiment.py':digest(root/'experiment.py')})
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:write(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)))
    x,weights=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1);checks=[]
    for teacher_kind in protocol['teacher_families']:
        for seed in protocol['seeds']:
            teacher=Network(3,3,3,teacher_kind=='tied',seed+100)
            tp={k:v.detach().numpy().copy() for k,v in teacher.named_parameters()}
            raw=evaluate(tp,x);tscale=np.sqrt(weights@raw**2);target=raw/tscale
            base=Network(3,3,3,True,seed+200);init={k:v.detach().numpy().copy() for k,v in base.named_parameters()}
            scale=np.sqrt(weights@evaluate(init,x)**2)
            for student in protocol['students']:
                initial={k:v.copy() for k,v in init.items()}
                if student=='untied_raw':initial.update(AR=initial['A'].copy(),BR=initial['B'].copy())
                # Independent implementation gate before any training in this case.
                m=Network(3,3,3,student!='untied_raw',0)
                with torch.no_grad():
                    for k,p in m.named_parameters():p.copy_(torch.tensor(initial[k]))
                loss=.5*(torch.tensor(weights)*(m(torch.tensor(x))/scale-torch.tensor(target))**2).sum()
                ag=torch.autograd.grad(loss,tuple(m.parameters()));analytic=gradients(initial,x,weights,target,scale)
                error=max(float(np.linalg.norm(g.numpy()-analytic[k])/max(1,np.linalg.norm(g.numpy()))) for (k,p),g in zip(m.named_parameters(),ag))
                assert error<1e-10;checks.append(error)
                for lr in protocol['rates']:
                    for method in protocol['methods']:
                        name=f'{teacher_kind}_{seed}_{student}_{lr}_{method}';dest=out/f'{name}.json'
                        if dest.exists():continue
                        p={k:v.copy() for k,v in initial.items()};curves=[];failure=None;steps=round(3/lr);start=time.monotonic()
                        metric='corrected' if student=='tied_corrected' else 'raw'
                        for k in range(steps+1):
                            if k in {0,1,round(.1/lr),round(1/lr),steps}:
                                mse=float(weights@(evaluate(p,x)/scale-target)**2)
                                if not np.isfinite(mse):failure=dict(step=k,reason='nonfinite loss');break
                                curves.append(dict(sweep=k,time=k*lr,mse=mse))
                            if k==steps:break
                            p=update(p,x,weights,target,scale,lr,method,metric)
                            if not all(np.isfinite(v).all() for v in p.values()):failure=dict(step=k+1,reason='nonfinite parameters');break
                        payload={f'teacher_{k}':v for k,v in tp.items()}
                        payload.update({f'initial_{k}':v for k,v in initial.items()});payload.update({f'final_{k}':v for k,v in p.items()})
                        np.savez_compressed(out/f'{name}.npz',**payload,teacher_scale=tscale,student_scale=scale)
                        write(dest,dict(name=name,teacher=teacher_kind,seed=seed,student=student,lr=lr,method=method,metric=metric,
                            curves=curves,failure=failure,full_gradient_evaluations=(1 if method=='joint' else 3)*steps,
                            seconds=time.monotonic()-start,parameters=sum(v.size for v in initial.values())))
                        print(name,curves[-1],failure,flush=True)
    write(out/'completed.json',dict(status='complete',fits=108,max_initial_gradient_error=max(checks)))


if __name__=='__main__':main()
