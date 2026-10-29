"""Independent polynomial coefficients and Gaussian moments, not quadrature replay."""
import hashlib,json,math
from pathlib import Path
import numpy as np

def coefficients(a,scale):
    c={(0,0,0):1.}
    for row in np.repeat(a,4//len(a),axis=0):
        nxt={}
        for power,v in c.items():
            for i,ai in enumerate(row):
                p=list(power);p[i]+=1;p=tuple(p);nxt[p]=nxt.get(p,0)+v*ai
        c=nxt
    return {p:v/scale for p,v in c.items()}
def moment(p):return math.prod(math.prod(range(1,n,2)) if n%2==0 else 0 for n in p)
def inner(a,b):return math.fsum(av*bv*moment(tuple(x+y for x,y in zip(ap,bp))) for ap,av in a.items() for bp,bv in b.items())
def main():
    out=Path(__file__).parent/'results/fixed_degree_reuse';assert json.loads((out/'completed.json').read_text())['fits']==81
    protocol=json.loads((out/'protocol.json').read_text());assert hashlib.sha256((out/'source.py').read_bytes()).hexdigest()==protocol['source_sha256'];rows=[];failures=[]
    for f in sorted(out.glob('t*_s*.json')):
        r=json.loads(f.read_text());a=np.load(f.with_suffix('.npz'));tc=coefficients(a['teacher'],float(a['teacher_scale']));assert abs(inner(tc,tc)-1)<1e-10
        ic=coefficients(a['initial'],float(a['student_scale']));assert abs(inner(ic,ic)-1)<1e-10
        if r['failure']:failures.append(r['name']);continue
        sc=coefficients(a['student'],float(a['student_scale']));diff={p:sc.get(p,0)-tc.get(p,0) for p in set(sc)|set(tc)};mse=inner(diff,diff);ref=r['curves'][-1]['population_mse'];assert abs(mse-ref)<1e-9*max(1,ref),(r['name'],mse,ref)
        rows.append(dict(name=r['name'],exact_moment_mse=mse,artifact_sha256=hashlib.sha256(f.with_suffix('.npz').read_bytes()).hexdigest()))
    worst=0.
    for tg in [1,2,4]:
        for seed in protocol['seeds']:
            base=np.load(out/f't{tg}_s{seed}_symmetric_g1.npz')['student'][0]
            ref=json.loads((out/f't{tg}_s{seed}_symmetric_g1.json').read_text())['curves']
            for sg in [2,4]:
                final=np.load(out/f't{tg}_s{seed}_symmetric_g{sg}.npz')['student'];err=float(np.max(abs(final-base)));assert err<1e-8;worst=max(worst,err)
                curves=json.loads((out/f't{tg}_s{seed}_symmetric_g{sg}.json').read_text())['curves'];np.testing.assert_allclose([r['population_mse'] for r in curves],[r['population_mse'] for r in ref],rtol=1e-8,atol=1e-9)
    result=dict(status='passed',audited_fits=len(rows),failures=failures,max_symmetric_trajectory_parameter_difference=worst,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
