"""Exact empirical-to-population norm bounds and conditional OLS noise risk."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,solve_triangular
from audit_fixed_degree_reuse import moment

def main():
    out=Path(__file__).parent/'results/reuse_sample_noise';powers=np.load(out/'evaluation.npz')['powers'];G=np.array([[moment(tuple(a+b)) for b in powers] for a in powers],dtype=float);L=np.linalg.cholesky(G);rows=[];bounds=0
    for seed in [180001,180002,180003]:
        data=np.load(out/f'data_t1_s{seed}.npz')
        for n in [32,128,512]:
            X=np.prod(data['x'][:n,None,:]**powers[None,:,:],axis=2);A=X.T@X/n;evals=eigh(G,A,eigvals_only=True);upper=float(evals[-1]);lower=float(evals[0]);white=solve_triangular(L,X.T,lower=True).T;sv=np.linalg.svd(white,compute_uv=False);assert abs(upper-n/sv[-1]**2)<1e-8*upper
            expected=.1**2*float(np.trace(np.linalg.solve(X.T@X,G)));assert abs(expected-.1**2*sum(evals)/n)<1e-10
            for tg in [1,2,4]:
                for sg in [1,2,4]:
                    r=json.loads((out/f't{tg}_s{seed}_n{n}_noise0_g{sg}.json').read_text());train=r['curves'][-1]['train_mse'];pop=r['population_mse'];assert lower*train-1e-10<=pop<=upper*train+1e-10;bounds+=1
            rows.append(dict(seed=seed,ntrain=n,lower_population_over_empirical_mse=lower,upper_population_over_empirical_mse=upper,conditional_expected_ols_population_mse_noise_std_point1=expected))
    (out/'conditioning.json').write_text(json.dumps(dict(status='passed',rows=rows,verified_noiseless_fit_bounds=bounds,gram=G.tolist()),indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
