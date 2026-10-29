"""Nested finite-data/noise study with exact quartic population evaluation."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from run_fixed_degree_reuse import polynomial,quadrature
from assess_fixed_degree_recovery import assess

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/reuse_sample_noise';out.mkdir(exist_ok=True)
    protocol=dict(d=3,degree=4,teacher_groups=[1,2,4],student_groups=[1,2,4],seeds=[180001,180002,180003],ntrain=[32,128,512],noise_std=[0,.1],steps=1000,lr=.03,optimizer='multiplicity-corrected full-batch plainSGD, no clipping/momentum/decay',initialization='independent factors, frozen exact initial Gaussian RMS1',sampling='nested prefixes of512Gaussian samples; common independent standardGaussian noise scaled bysigma; teacher/data/init streams separate',evaluation='exact Gaussian population relativeMSE against noiseless teacher; factors up to sign/permutation/scale',baseline='unregularized degree4homogeneous15coefficient leastsquares, fixed rcond=None',source_sha256=digest(Path(__file__)),dependencies={n:digest(root/n) for n in ['run_fixed_degree_reuse.py','assess_fixed_degree_recovery.py']})
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    else:(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source.py').write_bytes(Path(__file__).read_bytes())
    xx,ww=quadrature();qx=torch.tensor(xx);qw=torch.tensor(ww);powers=np.array([a for a in itertools.product(range(5),repeat=3) if sum(a)==4]);np.savez_compressed(out/'evaluation.npz',x=xx,w=ww,powers=powers)
    for tg in [1,2,4]:
        for seed in protocol['seeds']:
            rng=torch.Generator().manual_seed(seed+tg*100);teacher=torch.randn(tg,3,generator=rng,dtype=torch.float64);teacher/=teacher.norm(dim=1,keepdim=True);tscale=(qw@polynomial(qx,teacher).square()).sqrt();qtarget=polynomial(qx,teacher)/tscale
            rng=torch.Generator().manual_seed(seed+10000);x=torch.randn(512,3,generator=rng,dtype=torch.float64);noise=torch.randn(512,generator=rng,dtype=torch.float64);clean=polynomial(x,teacher)/tscale
            np.savez_compressed(out/f'data_t{tg}_s{seed}.npz',x=x.numpy(),noise=noise.numpy(),clean=clean.numpy(),teacher=teacher.numpy(),teacher_scale=float(tscale))
            for n in protocol['ntrain']:
                for sigma in protocol['noise_std']:
                    key=f't{tg}_s{seed}_n{n}_noise{sigma:g}';target=clean[:n]+sigma*noise[:n];features=np.prod(x[:n].numpy()[:,None,:]**powers[None,:,:],axis=2);coefs,res,rank,singular=np.linalg.lstsq(features,target.numpy(),rcond=None);qfeatures=np.prod(xx[:,None,:]**powers[None,:,:],axis=2);ols_mse=float(ww@(qfeatures@coefs-qtarget.numpy())**2)
                    np.savez_compressed(out/f'{key}_ols.npz',coefficients=coefs,singular_values=singular)
                    (out/f'{key}_ols.json').write_text(json.dumps(dict(teacher_groups=tg,seed=seed,ntrain=n,noise_std=sigma,rank=int(rank),population_mse=ols_mse,train_mse=float(np.mean((features@coefs-target.numpy())**2)),condition=float(singular[0]/singular[-1])),indent=2)+'\n')
                    for sg in [1,2,4]:
                        name=f'{key}_g{sg}';dest=out/f'{name}.json'
                        if dest.exists():continue
                        rng=torch.Generator().manual_seed(seed+20000);initial=torch.randn(4,3,generator=rng,dtype=torch.float64)[:sg].clone();scale=(qw@polynomial(qx,initial).square()).sqrt();a=initial.clone().requires_grad_();curves=[];failure=None;start=time.monotonic()
                        for step in range(1001):
                            loss=.5*(polynomial(x[:n],a)/scale-target).square().mean()
                            if not torch.isfinite(loss):failure=dict(step=step,reason='nonfinite loss');break
                            if step%100==0:curves.append(dict(step=step,train_mse=float(2*loss.detach())))
                            if step==1000:break
                            gradient=torch.autograd.grad(loss,a)[0]
                            if not torch.isfinite(gradient).all():failure=dict(step=step,reason='nonfinite gradient');break
                            with torch.no_grad():a.add_(gradient,alpha=-.03/(4//sg))
                        pop=None;recovery=None
                        if failure is None:
                            with torch.no_grad():value=qw@(polynomial(qx,a)/scale-qtarget).square()
                            if not torch.isfinite(value):failure=dict(step=1000,reason='nonfinite population evaluation')
                            else:pop=float(value);recovery=assess(teacher.numpy(),a.detach().numpy(),float(tscale),float(scale))
                        np.savez_compressed(out/f'{name}.npz',initial=initial.numpy(),student=a.detach().numpy(),student_scale=float(scale))
                        row=dict(name=name,teacher_groups=tg,student_groups=sg,seed=seed,ntrain=n,noise_std=sigma,population_mse=pop,recovery=recovery,curves=curves,failure=failure,seconds=time.monotonic()-start)
                        dest.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');print(name,'population',pop,'failure',failure,flush=True)
    (out/'completed.json').write_text(json.dumps(dict(status='complete',fits=162,baselines=54))+'\n')
if __name__=='__main__':main()
