"""M75 competence and effect coverage, without outcome-based filtering."""
import json
from collections import Counter
import numpy as np
import torch
from audit_whole_task_native import margin
from fit_pca_encoder import sha,V

def main():
    q=json.loads((V/'whole_task_native_protocol.json').read_text());r=json.loads((V/'whole_task_native.json').read_text());audit=json.loads((V/'whole_task_native_audit.json').read_text())
    assert audit['status']=='passed' and audit['result_sha256']==sha(V/'whole_task_native.json')
    assert sha(r['samples_path'])==r['samples_sha256'];s=torch.load(r['samples_path'],weights_only=True);pair_rows=[]
    for p in q['pairs']:
        metrics=[]
        for ep in (0,1):
            b=s['baselines'][p['id']+f'|{ep}'];z=b['logits'].numpy();m=margin(z,b['groups'],b['variant'])
            metrics.append(dict(endpoint=ep,margin=m,binary_correct=bool(m>0),top1_correct=bool(int(z.argmax()) in b['groups'][b['variant']]),top1_token=int(z.argmax())))
        pair_rows.append(dict(id=p['id'],family=p['family'],format=p['format'],both_correct=all(x['binary_correct'] for x in metrics),endpoints=metrics,source_eligible=p['source_region_eligible']))
    summary=[]
    for family in q['training_exclusions']:
        pairs=[x for x in pair_rows if x['family']==family];assert len(pairs)==8
        rows=[x for x in r['records'] if x['family']==family]
        summary.append(dict(family=family,declared_pairs=8,both_correct=sum(x['both_correct'] for x in pairs),competent=sum(x['both_correct'] for x in pairs)>=6,binary_correct_endpoints=sum(y['binary_correct'] for x in pairs for y in x['endpoints']),top1_correct_endpoints=sum(y['top1_correct'] for x in pairs for y in x['endpoints']),source_eligible_pairs=sum(x['source_eligible'] for x in pairs),formats=[dict(format=fmt,both_correct=sum(x['both_correct'] for x in pairs if x['format']==fmt),pairs=sum(x['format']==fmt for x in pairs)) for fmt in sorted({x['format'] for x in pairs})],effects=[dict(region=region,evaluated=sum(x['status']=='evaluated' for x in rows if x['region']==region),response_energy=sum(x['response_energy'] for x in rows if x['region']==region and x['status']=='evaluated'),mean_KL=float(np.mean([x['output_KL'] for x in rows if x['region']==region and x['status']=='evaluated']))) for region in ['changed_input_positions','final_position']]))
    out=dict(status='completed',source_sha256=sha(__file__),native_sha256=sha(V/'whole_task_native.json'),audit_sha256=sha(V/'whole_task_native_audit.json'),pair_rows=pair_rows,summaries=summary,scope='All56 historically evaluated prompt pairs. Competence is the predeclared6/8 pairwise answer-group criterion, not exhaustive open-ended correctness. No filtering, tuning, new task discovery or training.')
    path=V/'whole_task_native_comparison.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
