"""Bounded curvature diagnostic for the two observed noiseless ALS plateaus."""
import json
from pathlib import Path
import numpy as np
import torch

def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/reuse_als';rows=[]
    for seed in [180002,180003]:
        key=f't1_s{seed}_n128_noise0_g4';archive=np.load(out/f'{key}.npz');data=np.load(root/'results/reuse_sample_noise'/f'data_t1_s{seed}.npz');a=archive['states'][-1];norm=np.linalg.norm(a,axis=1);normalized=a/norm[:,None];scale=float(archive['student_scale']/np.prod(norm));x=torch.tensor(data['x'][:128]);target=torch.tensor(data['clean'][:128]);theta=torch.tensor(normalized.flatten(),requires_grad=True)
        def objective(v):return .5*((x@v.reshape(4,3).T).prod(-1)/scale-target).square().mean()
        loss=objective(theta);grad=torch.autograd.grad(loss,theta)[0];hess=torch.autograd.functional.hessian(objective,theta);vals,vecs=torch.linalg.eigh(hess)
        gauges=torch.zeros(12,3,dtype=torch.float64)
        for i in range(3):gauges[3*i:3*i+3,i]=theta.detach()[3*i:3*i+3];gauges[9:12,i]=-theta.detach()[9:12]
        basis=torch.linalg.qr(gauges,mode='complete')[0][:,3:];transverse=basis.T@hess@basis;tv,tvec=torch.linalg.eigh(transverse);v=basis@tvec[:,0];assert float((hess@gauges).norm())<1e-10;assert float(tv.min())>0
        zero=torch.tensor(np.tile(data['teacher'][0],(4,1))*(scale/float(data['teacher_scale']))**.25).flatten();zero_loss=float(objective(zero).detach());assert zero_loss<1e-25
        checks=[]
        for eps in [1e-4,1e-3,1e-2]:
            plus=float(objective(theta+eps*v).detach());minus=float(objective(theta-eps*v).detach());checks.append(dict(eps=eps,plus_half_mse=plus,minus_half_mse=minus,central_second_difference=(plus+minus-2*float(loss.detach()))/eps**2))
        row=dict(seed=seed,half_train_mse=float(loss.detach()),gradient_norm=float(grad.norm()),hessian_eigenvalues=vals.detach().tolist(),transverse_eigenvalues=tv.detach().tolist(),gauge_hessian_residual=float((hess@gauges).norm()),explicit_zero_loss_solution=zero_loss,finite_difference_direction='smallest positive transverse-curvature eigenvector',finite_difference_checks=checks,normalization='unit factor norms, compensate frozen scale to preserve function',selection='two one-factor-teacher N128 noiseless fits whose final10sweep train changes were at rounding scale; diagnostic not a predeclared sweep')
        rows.append(row);print(json.dumps(row))
    (out/'plateau_curvature.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
