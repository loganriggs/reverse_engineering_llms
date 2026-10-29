"""M77 all downstream vectors scored independently with NumPy."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V
from audit_pca_factorial import lp,margin
P=V.parent

def main():
    torch.set_num_threads(1);rp=V/'native_input_operator_results.json';result=json.loads(rp.read_text())
    assert result['source_sha256']==sha(P/'probe_native_input_operators.py') and result['local_sha256']==sha(V/'native_input_changes.json') and result['local_audit_sha256']==sha(V/'native_input_changes_audit.json')
    execution=json.loads((V/'native_input_operator_execution.json').read_text());assert result['execution_sha256']==sha(V/'native_input_operator_execution.json')
    native=json.loads((V/'whole_task_native.json').read_text());assert sha(native['samples_path'])==native['samples_sha256'];saved=torch.load(native['samples_path'],weights_only=True);targets={r['sample_key']:r for r in native['records']}
    local=json.loads((V/'native_input_changes.json').read_text());lrows={(r['pair_id'],r['endpoint'],r['region'],r['predictor']):r for r in local['rows']}
    rows={(r['key']+'|1|'+r['region'],r['predictor']):r for r in result['rows']};assert len(rows)==984
    maximum=0.;midpoint=0.;count=0;calls=112+sum(r['status']=='evaluated' and not r['noop'] for r in native['records'])
    def check(a,b):
        nonlocal maximum
        assert np.isclose(a,b,rtol=1e-8,atol=1e-9),(a,b);maximum=max(maximum,abs(a-b))
    for file in result['files']:
        assert sha(file['path'])==file['sha256'];s=torch.load(file['path'],weights_only=True)
        for key,logits in s['changed'].items():
            sk,predictor=key.rsplit('|',1);row=rows[sk,predictor];target=targets[sk];basekey=row['key'];b=saved['baselines'][basekey];z=b['logits'].numpy();bl=lp(z);teacher=lp(saved['changed'][sk])-bl;response=lp(logits)-bl;error=float((np.exp(bl)*(response-teacher)**2).sum());effect=margin(logits,b['groups'],b['variant'])-margin(z,b['groups'],b['variant']);check(error,row['response_error']);check(float((np.exp(bl)*teacher**2).sum()),row['target_energy']);check((effect-target['margin_change'])**2,row['margin_error']);check(target['margin_change']**2,row['margin_target_energy']);check(lrows[row['pair_id'],row['endpoint'],row['region'],predictor]['predicted_delta_energy'],row['delta_energy']);calls+=not row['noop'];count+=1
            if predictor=='midpoint_J_exact_control':
                midpoint=max(midpoint,error);assert error<=execution['midpoint_response_error_atol']+execution['midpoint_response_error_rtol']*row['target_energy']
            if row['region']=='identity':assert error==row['margin_error']==row['delta_energy']==0
    invalid={(r['pair_id'],r['endpoint'],r['region'],r['predictor']) for r in result['ineligible']};expected={(r['pair_id'],r['endpoint'],r['region'],p) for r in native['records'] if r['status']!='evaluated' for p in execution['predictors']};assert invalid==expected and len(invalid)==24
    assert count==984 and calls==976 and result['forward_count']==976 and result['teacher_replays']==328
    out=dict(status='passed',source_sha256=sha(__file__),metric_helper_sha256=sha(P/'audit_pca_factorial.py'),result_sha256=sha(rp),response_vectors=count,ineligible_cases=24,max_metric_absolute_error=maximum,max_midpoint_response_error=midpoint,scope='All full-vocabulary saved response vectors and target/delta/margin metrics independently rescored; exact midpoint numerical tolerance and all identities/ineligibilities checked. Local predictions independently audited separately.')
    op=V/'native_input_operator_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
