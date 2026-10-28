"""M37 paired objective effects and transparent architecture comparison; no selection."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
np=P/'joint_shared_native_fidelity.json';op=P/'joint_shared_outputs.json';n=json.loads(np.read_text());o=json.loads(op.read_text());assert json.loads((P/'joint_shared_native_audit.json').read_text())['result_sha256']==sha(np);assert json.loads((P/'joint_shared_outputs_audit.json').read_text())['result_sha256']==sha(op)
native=[];outputs=[]
for s in n['summaries']:
 if s['arm']!='intact_KL_plus_native_response':continue
 keys=['kind','seed','region','generator','quadrant','family'];c=next(x for x in n['summaries'] if x['arm']=='intact_KL' and all(x[k]==s[k] for k in keys));initial=next(x for x in n['summaries'] if x['arm']=='initial' and all(x[k]==s[k] for k in keys));row={k:s[k] for k in keys}
 for metric in ['relative_logprob_response_error','relative_margin_error']:
  a,b=s[metric],c[metric];row[metric]=dict(initial=initial[metric],control=b,joint=a,joint_control_ratio=a/b if a is not None and b else None,joint_better=a<b if a is not None and b is not None else None)
 native.append(row)
for s in o['summaries']:
 if s['arm']!='intact_KL_plus_native_response':continue
 keys=['kind','seed','domain','split','template_split','gate','family'];c=next(x for x in o['summaries'] if x['arm']=='intact_KL' and all(x[k]==s[k] for k in keys));initial=next(x for x in o['summaries'] if x['arm']=='initial' and all(x[k]==s[k] for k in keys));row={k:s[k] for k in keys}
 for metric in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_error','active_mean']:
  a,b=s[metric],c[metric];row[metric]=dict(initial=initial[metric],control=b,joint=a,joint_minus_control=a-b)
 outputs.append(row)
headline=[x for x in native if x['generator']=='all' and x['quadrant']=='all' and x['family'] is None]
counts={k:sum(x[k]['joint_better'] for x in headline) for k in ['relative_logprob_response_error','relative_margin_error']}
result=dict(native_sha256=sha(np),outputs_sha256=sha(op),paired_native=native,paired_outputs=outputs,headline_counts=counts,headline_comparisons=len(headline),scope='Paired objective comparisons on reused development evaluation. All architectures, seeds, regions, source generators and strata retained. No best-model selection, confidence interval, semantic-unit or new-task claim.')
(P/'joint_shared_transfer_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print('COUNTS',counts)
for x in headline:print(x['kind'],x['seed'],x['region'],'logprob',x['relative_logprob_response_error'],'margin',x['relative_margin_error'])
print('OUTPUT HEADLINES')
for x in outputs:
 if x['family'] is None and x['gate']=='all' and (x['domain']=='natural' or (x['domain']=='expanded' and x['split']==x['template_split']=='evaluation')):print(x['kind'],x['seed'],x['domain'],'meanKL',x['mean_KL'],'top1',x['top1_agreement'])
