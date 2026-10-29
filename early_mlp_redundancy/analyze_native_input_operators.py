"""M77 all-family/format and contrast-qualified summaries without selection."""
import json
from collections import defaultdict
import numpy as np
from fit_pca_encoder import sha,V

def main():
    rp=V/'native_input_operator_results.json';r=json.loads(rp.read_text());a=json.loads((V/'native_input_operator_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp)
    local=json.loads((V/'native_input_changes.json').read_text());qualified={s['family'] for s in json.loads((V/'whole_task_native_comparison.json').read_text())['summaries'] if s['competent']}
    geometry={(x['pair_id'],x['endpoint'],x['region'],x['predictor']):x for x in local['rows']};groups=defaultdict(list)
    for row in r['rows']:
        scopes=['all','family:'+row['family'],'format:'+row['family']+':'+row['format']]+(['contrast_qualified'] if row['family'] in qualified else [])
        for scope in scopes:groups[scope,row['region'],row['predictor']].append(row)
    summaries=[]
    for (scope,region,predictor),rows in groups.items():
        en=sum(x['target_energy'] for x in rows);err=sum(x['response_error'] for x in rows);me=sum(x['margin_target_energy'] for x in rows);mr=sum(x['margin_error'] for x in rows);gg=[geometry[x['pair_id'],x['endpoint'],x['region'],x['predictor']] for x in rows];ge=sum(x['target_delta_energy'] for x in gg);gr=sum(x['delta_error'] for x in gg)
        family=[]
        for fam in sorted({x['family'] for x in rows}):
            rr=[x for x in rows if x['family']==fam];den=sum(x['target_energy'] for x in rr)
            if den:family.append(sum(x['response_error'] for x in rr)/den)
        summaries.append(dict(scope=scope,region=region,predictor=predictor,n=len(rows),target_energy=en,error_energy=err,pooled_response_error=err/en if en else None,equal_family_response_error=float(np.mean(family)) if family else None,margin_target_energy=me,margin_error_energy=mr,relative_margin_error=mr/me if me else None,local_target_energy=ge,local_error_energy=gr,local_relative_error=gr/ge if ge else None))
    out=dict(status='completed',source_sha256=sha(__file__),results_sha256=sha(rp),audit_sha256=sha(V/'native_input_operator_audit.json'),local_sha256=sha(V/'native_input_changes.json'),summaries=summaries,scope='All three fixed predictors, all seven historically evaluated training-excluded families and14formats; qualified summary supplements full cohort. Shared anchor tests the training mean Jacobian, not the optimal global affine fit. Midpoint is an algebraic exact control, not compression or discovery.')
    path=V/'native_input_operator_comparison.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([s for s in summaries if s['scope'] in ['all','contrast_qualified'] and s['region']!='identity'],indent=2))
if __name__=='__main__':main()
