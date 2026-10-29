"""Independent NumPy scoring of every M74 finite-difference preflight output."""
import json
import numpy as np
import torch
from downstream_metric_helpers import P,V,sha

def energy(base,tangent):
    z=np.asarray(base,dtype=np.float64);z-=z.max(-1,keepdims=True)
    p=np.exp(z);p/=p.sum(-1,keepdims=True)
    t=np.asarray(tangent,dtype=np.float64);t=t-(p*t).sum(-1,keepdims=True)
    return float((p*t*t).sum(-1).mean())

def main():
    torch.set_num_threads(1);rp=V/'downstream_metric_preflight.json';result=json.loads(rp.read_text());execution=json.loads((V/'downstream_metric_execution.json').read_text())
    assert result['status']=='passed' and result['execution_sha256']==sha(V/'downstream_metric_execution.json')
    assert result['source_sha256']==sha(P/'check_downstream_metric.py') and result['helper_sha256']==sha(P/'downstream_metric_helpers.py')
    maximum=0.;checks=0
    def check(a,b):
        nonlocal maximum
        assert np.isclose(a,b,rtol=1e-8,atol=1e-10),(a,b);maximum=max(maximum,abs(a-b))
    assert len(result['rows'])==len(result['files'])==38
    seen=set()
    for row,file in zip(result['rows'],result['files']):
        assert sha(file['path'])==file['sha256'];s=torch.load(file['path'],weights_only=True)
        key=(s['document'],s['model']['kind'],s['model']['seed'],s['model']['arm']);assert key not in seen;seen.add(key)
        for k,value in s['model'].items():assert row[k]==value
        assert row['document']==s['document'];t=s['tangent'].double().numpy();base=s['native'].numpy();E=energy(base,t);check(E,row['Fisher_energy'])
        for saved,test in zip(s['tests'],row['tests']):
            h=saved['step'];assert h==test['step'];fd=(saved['plus'].double().numpy()-saved['minus'].double().numpy())/(2*h)
            error=float(np.linalg.norm(fd-t)/np.linalg.norm(t));fe=energy(base,fd);ferr=abs(fe-E)/E
            # Producer's tangent norm is float32; independent NumPy norm is float64.
            assert np.isclose(error,test['relative_derivative_error'],rtol=1e-6,atol=1e-10);maximum=max(maximum,abs(error-test['relative_derivative_error']));check(fe,test['Fisher_energy']);check(ferr,test['relative_Fisher_error'])
            assert error<execution['preflight_max_relative_derivative_error'] and ferr<execution['preflight_max_relative_Fisher_error'];checks+=1
    assert checks==114 and result['forward_count']==268
    op=V/'downstream_metric_preflight_audit.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',result_sha256=sha(rp),source_sha256=sha(__file__),cases=38,finite_difference_checks=checks,max_metric_absolute_error=maximum),indent=2)+'\n');print('PASSED',checks,maximum)
if __name__=='__main__':main()
