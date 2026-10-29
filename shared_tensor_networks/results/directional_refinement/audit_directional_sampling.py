"""Independent density, aggregate and exact-circle deep-network checks."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from directional_sampling import DirectionalMixture,fit_axes
from deep_multitask import DeepMultiTask


def main():
    torch.set_num_threads(1);root=Path(__file__).parent;out=root/'results/directional_refinement'
    result=json.loads((out/'results.json').read_text());worst=0.
    for r in result['results']:
        for row in r['rows']:
            v=np.load(out/f"depth{r['depth']}_batch{row['batch']}.npz");a=v['weighted_values'];w=v['weights']
            assert a.shape==(32768,2) and np.all(np.isfinite(a)) and np.all((w>0)&(w<=5+1e-12))
            np.testing.assert_allclose(a.mean(axis=0),row['means'],rtol=1e-13)
            np.testing.assert_allclose(a.sum(axis=0)**2/(a*a).sum(axis=0),row['ess'],rtol=1e-13)
        means=np.array([row['means'] for row in r['rows']]);np.testing.assert_allclose(means.std(axis=0,ddof=1)/means.mean(axis=0),r['batch_cv'])
        axes=np.load(out/f"axes_depth{r['depth']}.npy");proposal=DirectionalMixture(axes,(32,128))
        u=np.random.default_rng(114000+r['depth']).normal(size=(100,16));u/=np.linalg.norm(u,axis=1,keepdims=True)
        density=np.full(100,.2)
        for axis in axes:
            axis=axis/np.linalg.norm(axis)
            for tau in (32,128):
                S=np.eye(16)+tau*np.outer(axis,axis)
                density+=.8/128*np.linalg.det(S)**(-.5)*np.einsum('ni,ij,nj->n',u,np.linalg.inv(S),u)**(-8)
        err=np.max(np.abs(density-np.exp(proposal.log_density(u)))/density);worst=max(worst,float(err));assert err<1e-10
    # On a 2D circle, degree32 squared error is a trigonometric polynomial
    # with maximum Fourier frequency64. Any equally spaced N>64 integrates
    # it exactly in exact arithmetic. Check both N256 and N512 numerically.
    def circle(n):
        theta=np.arange(n)*2*np.pi/n
        return torch.from_numpy(np.sqrt(2)*np.column_stack([np.cos(theta),np.sin(theta)]))
    teacher=DeepMultiTask(2,4,3,5,True,115000);student=DeepMultiTask(2,4,3,5,True,115001)
    teacher.calibrate(circle(256));student.calibrate(circle(256))
    def energy(x):
        with torch.no_grad():
            y=teacher(x);pred=student(x)
            return np.column_stack([y.square().mean(dim=1).numpy(),(pred-y).square().mean(dim=1).numpy()])
    truth=energy(circle(256)).mean(axis=0);truth2=energy(circle(512)).mean(axis=0)
    np.testing.assert_allclose(truth,truth2,rtol=1e-12,atol=1e-12)
    pilot=torch.from_numpy(np.random.default_rng(115002).normal(size=(16,2)))
    proposal=DirectionalMixture(fit_axes(teacher,student,pilot),stretches=(16,64))
    rng=np.random.default_rng(115003);estimates=[]
    for _ in range(16):
        u,w=proposal.sample(8192,rng);estimates.append((energy(torch.from_numpy(u*np.sqrt(2)))*w[:,None]).mean(axis=0))
    estimates=np.array(estimates);se=estimates.std(axis=0,ddof=1)/4
    assert np.all(np.abs(estimates.mean(axis=0)-truth)<np.maximum(.025*truth,6*se))
    audit=dict(status='passed',aggregate_rows=24,max_independent_density_relative_error=worst,
               circle_depth=5,circle_degree=32,exact_circle_truth=truth.tolist(),weighted_mean=estimates.mean(axis=0).tolist(),weighted_batch_means=estimates.tolist(),
               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))


if __name__=='__main__':main()
