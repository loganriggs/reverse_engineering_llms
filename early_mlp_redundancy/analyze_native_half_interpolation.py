"""M43 fixed two-scale diagnosis; no best-amplitude selection."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'native_half_interpolation.json';r=json.loads(rp.read_text());assert json.loads((P/'native_half_interpolation_audit.json').read_text())['result_sha256']==sha(rp);q=json.loads((P/'native_half_interpolation_protocol.json').read_text());p0=P/'native_half_compensation.json';p1=P/'native_half_substitution_transfer.json';assert sha(p0)==q['M41_sha256'] and sha(p1)==q['M42_sha256'];r0=json.loads(p0.read_text());r1=json.loads(p1.read_text())
key=lambda x:(x['key'],x['layer'],x['removed'])
z={key(x):x for x in r0['rows'] if x['factor']==1};full={key(x):x for x in r1['rows'] if x['arm']=='rank32'};points={}
for x in r['rows']:points.setdefault(key(x),{})[x['amplitude']]=x
rows=[]
for k,ps in points.items():
 base=z[k];one=full[k];a=(ps[.01]['mean_KL']-ps[-.01]['mean_KL'])/.02;b=(ps[.001]['mean_KL']-ps[-.001]['mean_KL'])/.002;agreement=abs(a-b)/max(abs(a),abs(b),1e-8);la=(ps[.01]['local_relative_error']-ps[-.01]['local_relative_error'])/.02;lb=(ps[.001]['local_relative_error']-ps[-.001]['local_relative_error'])/.002
 rows.append(dict(key=k[0],layer=k[1],removed=k[2],domain=base['domain'],split=base['split'],family=base['family'],KL_0=base['mean_KL'],KL_p001=ps[.001]['mean_KL'],KL_p01=ps[.01]['mean_KL'],KL_p1=ps[.1]['mean_KL'],KL_1=one['mean_KL'],slope_01=a,slope_001=b,relative_scale_disagreement=agreement,scale_agrees=agreement<=.1,local_slope_01=la,local_slope_001=lb,local_error_0=base['local_relative_error'],local_error_1=one['local_relative_error']))
summaries=[]
for domain,part in [('task','train'),('task','evaluation'),('natural','evaluation')]:
 for layer in [0,1,2]:
  for half in [None,'half_01','half_23']:
   rr=[x for x in rows if x['domain']==domain and x['split']==part and x['layer']==layer and (half is None or x['removed']==half)];agree=[x for x in rr if x['scale_agrees']];avg=lambda f:sum(x[f] for x in rr)/len(rr)
   summary=dict(domain=domain,split=part,layer=layer,removed=half,n=len(rr),scale_agreement_count=len(agree),stable_positive_count=sum(x['slope_001']>0 for x in agree),stable_negative_count=sum(x['slope_001']<0 for x in agree),positive_local_slope_count=sum(x['local_slope_001']>0 for x in rr),full_harm_count=sum(x['KL_1']>x['KL_0'] for x in rr),stable_initial_help_full_harm_count=sum(x['slope_001']<0 and x['KL_1']>x['KL_0'] for x in agree),**{f:avg(f) for f in ['KL_0','KL_p001','KL_p01','KL_p1','KL_1','slope_01','slope_001','local_slope_001']});summaries.append(summary)
   if half is None:print(summary)
out=dict(result_sha256=sha(rp),source_sha256=sha(Path(__file__)),records=rows,summaries=summaries,scope='Fixed finite slopes around raw deletion, reported with two-scale agreement; no best amplitude or new map fit. Aggregate sign need not hold for every input or mask.')
(P/'native_half_interpolation_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
