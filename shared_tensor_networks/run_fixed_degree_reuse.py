"""Fixed quartic degree; exact Gaussian population loss; symmetry-aware controls."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch

def quadrature():
    z,w=np.polynomial.hermite_e.hermegauss(5);indices=np.array(list(itertools.product(range(5),repeat=3)))
    return z[indices],np.prod(w[indices]/np.sqrt(2*np.pi),axis=1)
def polynomial(x,a):return (x@a.T).pow(4//len(a)).prod(-1)
def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/fixed_degree_reuse';out.mkdir(exist_ok=True)
    protocol=dict(d=3,degree=4,teacher_groups=[1,2,4],student_groups=[1,2,4],seeds=[170001,170002,170003],initialization=['symmetric','perturbed','independent'],steps=1000,lr=.03,optimizer='plain SGD divided by factor multiplicity4/g; no clipping/momentum/decay',objective='half exact Gaussian population MSE via5point-per-axis Gauss-Hermite, no sampled train/test split',normalization='teacher Gaussian RMS1; each student fixed initial Gaussian RMS1; symmetric states match functions across group counts',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    xx,ww=quadrature();x=torch.tensor(xx,dtype=torch.float64);w=torch.tensor(ww,dtype=torch.float64);np.savez_compressed(out/'quadrature.npz',x=xx,w=ww)
    for tg in [1,2,4]:
        for seed in protocol['seeds']:
            rng=torch.Generator().manual_seed(seed+tg*100);teacher=torch.randn(tg,3,generator=rng,dtype=torch.float64);teacher/=teacher.norm(dim=1,keepdim=True);raw=polynomial(x,teacher);tscale=(w@raw.square()).sqrt();target=raw/tscale
            for mode in protocol['initialization']:
                for sg in [1,2,4]:
                    name=f't{tg}_s{seed}_{mode}_g{sg}';dest=out/f'{name}.json'
                    if dest.exists():continue
                    rng=torch.Generator().manual_seed(seed+1000);base=torch.randn(1,3,generator=rng,dtype=torch.float64);base/=base.norm();all_noise=torch.randn(4,3,generator=rng,dtype=torch.float64)
                    init=base.repeat(sg,1) if mode=='symmetric' else base.repeat(sg,1)+.05*all_noise[:sg] if mode=='perturbed' else all_noise[:sg].clone()
                    scale=(w@polynomial(x,init).square()).sqrt();a=init.clone().requires_grad_();curves=[];failure=None;start=time.monotonic()
                    for step in range(1001):
                        pred=polynomial(x,a)/scale;loss=.5*(w@(pred-target).square())
                        if not torch.isfinite(loss):failure=dict(step=step,reason='nonfinite loss');break
                        if step%50==0:curves.append(dict(step=step,population_mse=2*float(loss.detach()),factor_spread=float((a-a.mean(0)).norm().detach())))
                        if step==1000:break
                        grad=torch.autograd.grad(loss,a)[0]
                        if not torch.isfinite(grad).all():failure=dict(step=step,reason='nonfinite gradient');break
                        with torch.no_grad():a.add_(grad,alpha=-.03/(4//sg))
                    np.savez_compressed(out/f'{name}.npz',teacher=teacher.numpy(),teacher_scale=float(tscale),initial=init.numpy(),student=a.detach().numpy(),student_scale=float(scale))
                    row=dict(name=name,teacher_groups=tg,student_groups=sg,seed=seed,initialization=mode,parameters=sg*3,curves=curves,failure=failure,seconds=time.monotonic()-start)
                    dest.write_text(json.dumps(row,indent=2)+'\n');print(name,'mse',curves[-1]['population_mse'] if curves else None,'failure',failure,flush=True)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',fits=81))+'\n')
if __name__=='__main__':main()
