"""Analytic Gaussian coefficient moments and saved-data objective replay."""
import hashlib,json
from pathlib import Path
import numpy as np
from audit_fixed_degree_reuse import coefficients,inner

def main():
    out=Path(__file__).parent/'results/reuse_sample_noise';assert json.loads((out/'completed.json').read_text())['fits']==162
    protocol=json.loads((out/'protocol.json').read_text());assert hashlib.sha256((out/'source.py').read_bytes()).hexdigest()==protocol['source_sha256'];powers=np.load(out/'evaluation.npz')['powers'];records=[];failures=[];worst=0.
    for f in sorted(out.glob('t*_s*_n*_noise*_ols.json')):
        r=json.loads(f.read_text());d=np.load(out/f"data_t{r['teacher_groups']}_s{r['seed']}.npz");n=r['ntrain'];sigma=r['noise_std'];target=d['clean'][:n]+sigma*d['noise'][:n];tc=coefficients(d['teacher'],float(d['teacher_scale']));assert abs(inner(tc,tc)-1)<1e-10
        features=np.prod(d['x'][:n,None,:]**powers[None,:,:],axis=2);truth=np.array([tc[tuple(p)] for p in powers]);np.testing.assert_allclose(features@truth,d['clean'][:n],atol=1e-10,rtol=1e-10)
        z=np.load(f.with_suffix('.npz'));c={tuple(p):v for p,v in zip(powers,z['coefficients'])};diff={p:c[p]-tc[p] for p in tc};mse=inner(diff,diff);assert abs(mse-r['population_mse'])<1e-9*max(1,mse)
        residual=features@z['coefficients']-target;assert np.linalg.norm(features.T@residual)<1e-8*max(1,np.linalg.norm(features)*np.linalg.norm(target));assert abs(np.mean(residual**2)-r['train_mse'])<1e-10
        key=f.stem[:-4]
        for sg in [1,2,4]:
            path=out/f'{key}_g{sg}.json';row=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'));ic=coefficients(a['initial'],float(a['student_scale']));assert abs(inner(ic,ic)-1)<1e-10
            if row['failure']:failures.append(dict(name=row['name'],failure=row['failure']));continue
            sc=coefficients(a['student'],float(a['student_scale']));delta={p:sc[p]-tc[p] for p in tc};actual=inner(delta,delta);ref=row['population_mse'];err=abs(actual-ref)/max(1,abs(ref));assert err<1e-9;worst=max(worst,err)
            pred=features@np.array([sc[tuple(p)] for p in powers]);train=np.mean((pred-target)**2);assert abs(train-row['curves'][-1]['train_mse'])<1e-9*max(1,train)
            records.append(dict(name=row['name'],checkpoint_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()))
    assert len(records)+len(failures)==162
    result=dict(status='passed',audited_successful_fits=len(records),retained_failures=failures,baselines=54,max_relative_population_discrepancy=worst,records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
