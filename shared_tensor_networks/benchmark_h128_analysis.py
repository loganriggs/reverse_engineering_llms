"""Full-shape analysis feasibility; random fixture, not a trained-model result."""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
from quartic_cross_gram import product_factors,cross_gram
from coverage_evaluation import features,population_errors
from coverage_audit import phi,svd_refit

def main():
    out=Path('results/h128_analysis_preflight');out.mkdir(exist_ok=False)
    rng=np.random.default_rng(265201);A=rng.normal(size=(128,256))/16;B=rng.normal(size=(128,256))/16
    timings={};begin=time.perf_counter();F=product_factors(A,B,np.ones(128));G=cross_gram(F,F);timings['gram']=time.perf_counter()-begin
    np.testing.assert_allclose(G,G.T,rtol=1e-12,atol=1e-12);assert np.isfinite(G).all()
    W=rng.normal(size=(4,8256))/8256**.5;W/=np.sqrt(np.sum((W@G)*W,axis=1))[:,None]
    print('Exact Gram completed',timings['gram'],flush=True)
    x=rng.normal(size=(32768,256));begin=time.perf_counter();X=features(A,B,x);timings['features']=time.perf_counter()-begin
    np.testing.assert_allclose(X[:31],phi(A,B,x[:31]),rtol=1e-12,atol=1e-12)
    y=X@W.T+.05*rng.normal(size=(len(x),4));records=[]
    for n in [8192,32768]:
        begin=time.perf_counter();coef,_,rank,s=np.linalg.lstsq(X[:n],y[:n],rcond=1e-12);timings[f'lstsq_{n}']=time.perf_counter()-begin
        print('Least squares completed',n,timings[f'lstsq_{n}'],flush=True)
        begin=time.perf_counter();independent_rank=svd_refit(X[:n],y[:n],coef);timings[f'svd_audit_{n}']=time.perf_counter()-begin
        assert rank==independent_rank
        err=population_errors(coef,G,G,W,np.ones(4));delta=coef-W.T
        direct=np.sum(delta*(G@delta),axis=0)
        np.testing.assert_allclose(err,direct,rtol=1e-7,atol=1e-8);assert np.isfinite(err).all() and min(err)>-1e-8
        records.append(dict(n=n,rank=int(rank),population_error_replay=True,independent_svd_agreement=True))
        (out/'progress.json').write_text(json.dumps(dict(records=records,times_seconds=timings),indent=2)+'\n')
        print('Audit completed',n,timings[f'svd_audit_{n}'],flush=True)
    result=dict(scope='Known random dictionary; analysis resource/correctness check only, not scientific learning outcomes',d=256,h=128,products=8256,records=records,times_seconds=timings,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sources={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['benchmark_h128_analysis.py','coverage_evaluation.py','coverage_audit.py','quartic_cross_gram.py']})
    (out/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
