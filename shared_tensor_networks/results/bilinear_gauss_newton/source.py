"""Fixed-damping Gauss-Newton with Armijo search on archived bilinear tasks."""
import itertools,time
from pathlib import Path
import numpy as np
from run_bilinear_alternation import evaluate
from run_svhn_standard import digest,write


def pack(p):return np.concatenate([v.ravel() for v in p.values()])
def unpack(v,template):
    p={};start=0
    for k,a in template.items():p[k]=v[start:start+a.size].reshape(a.shape).copy();start+=a.size
    return p


def jacobian(p,x,scale):
    ax,bx=x@p['A'].T,x@p['B'].T;left=ax*bx
    if 'AR' in p:ar,br=x@p['AR'].T,x@p['BR'].T;right=ar*br
    else:right=left
    u,v=left@p['C'].T,right@p['D'].T
    du,dv=v*p['w'],u*p['w'];dl,dr=du@p['C'],dv@p['D']
    j=dict(C=du[:,:,None]*left[:,None,:],D=dv[:,:,None]*right[:,None,:],w=u*v)
    if 'AR' in p:j.update(AR=(dr*br)[:,:,None]*x[:,None,:],BR=(dr*ar)[:,:,None]*x[:,None,:])
    else:dl+=dr
    j.update(A=(dl*bx)[:,:,None]*x[:,None,:],B=(dl*ax)[:,:,None]*x[:,None,:])
    return np.concatenate([j[k].reshape(len(x),-1) for k in p],axis=1)/scale


def main():
    root=Path(__file__).parent;out=root/'results/bilinear_gauss_newton';out.mkdir(exist_ok=True)
    protocol=dict(seeds=[211001,211002,211003],teacher_families=['tied','untied'],students=['tied_raw','tied_corrected','untied_raw'],
        damping=[.001,.1],max_steps=100,mse_stop=1e-12,armijo=1e-4,line_search_alphas=[2.**-k for k in range(12)],
        metric='corrected tied damping M has2 on A/B,1 elsewhere; raw usesI',
        objective='half exact Gaussian population MSE, same archived initialization/scales as layerwise panel',
        source_sha256=digest(Path(__file__)),dependencies={'run_bilinear_alternation.py':digest(root/'run_bilinear_alternation.py')})
    write(out/'protocol.json',protocol);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    z,w1=np.polynomial.hermite_e.hermegauss(5);ids=np.array(list(itertools.product(range(5),repeat=3)));x,w=z[ids],np.prod(w1[ids]/np.sqrt(2*np.pi),axis=1)
    for teacher in protocol['teacher_families']:
        for seed in protocol['seeds']:
            for student in protocol['students']:
                source=root/f'results/bilinear_alternation_long/{teacher}_{seed}_{student}_0.003_joint.npz';d=np.load(source)
                initial={k[8:]:d[k] for k in d.files if k.startswith('initial_')}
                tp={k[8:]:d[k] for k in d.files if k.startswith('teacher_') and k!='teacher_scale'}
                scale=float(d['student_scale']);target=evaluate(tp,x)/float(d['teacher_scale'])
                metric=np.concatenate([np.full(a.size,2. if student=='tied_corrected' and k in ['A','B'] else 1.) for k,a in initial.items()])
                for damping in protocol['damping']:
                    name=f'{teacher}_{seed}_{student}_{damping}';dest=out/f'{name}.json'
                    if dest.exists():continue
                    p={k:v.copy() for k,v in initial.items()};states=[pack(p)];curves=[];steps=[];failure=None;calls=0;start=time.monotonic()
                    for iteration in range(101):
                        residual=np.sqrt(w)*(evaluate(p,x)/scale-target);mse=float(residual@residual);calls+=1
                        curves.append(dict(iteration=iteration,mse=mse))
                        if mse<1e-12 or iteration==100:break
                        J=np.sqrt(w)[:,None]*jacobian(p,x,scale);g=J.T@residual;H=J.T@J+damping*np.diag(metric)
                        direction=np.linalg.solve(H,-g);slope=float(g@direction)
                        if not np.isfinite(direction).all() or slope>=0:
                            failure=dict(iteration=iteration,reason='invalid descent direction');break
                        accepted=False
                        for trial,alpha in enumerate(protocol['line_search_alphas'],1):
                            candidate=unpack(pack(p)+alpha*direction,p);rnew=np.sqrt(w)*(evaluate(candidate,x)/scale-target);new_mse=float(rnew@rnew);calls+=1
                            if np.isfinite(new_mse) and .5*new_mse<=.5*mse+1e-4*alpha*slope:
                                accepted=True;break
                        if not accepted:failure=dict(iteration=iteration,reason='line search exhausted');break
                        steps.append(dict(alpha=alpha,trials=trial,slope=slope,linear_system_residual=float(np.linalg.norm(H@direction+g)),mse_before=mse,mse_after=new_mse))
                        p=candidate;states.append(pack(p))
                    np.savez_compressed(out/f'{name}.npz',states=np.stack(states),metric=metric)
                    write(dest,dict(name=name,teacher=teacher,seed=seed,student=student,damping=damping,input_file=str(source.relative_to(root)),input_sha256=digest(source),
                        curves=curves,steps=steps,failure=failure,forward_evaluations=calls,jacobian_evaluations=len(steps)+(1 if failure else 0),seconds=time.monotonic()-start))
                    print(name,curves[-1],failure,flush=True)
    write(out/'completed.json',dict(status='complete',fits=36))


if __name__=='__main__':main()
