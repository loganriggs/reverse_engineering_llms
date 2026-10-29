"""Independent NumPy norms, saved matched vectors, and all M65 response metrics."""
from pathlib import Path
import json,torch,numpy as np
from audit_support_change_diagnostic import sha,lp,margin
P=Path(__file__).resolve().parent;V=P/'validation'
def main():
 torch.set_num_threads(1);rp=V/'support_norm_responses.json';r=json.loads(rp.read_text());geo=json.loads((V/'support_change_geometry.json').read_text());native=json.loads((V/'source_holdout_native.json').read_text());assert r['status']=='completed' and r['source_sha256']==sha(P/'probe_support_norm_control.py') and r['geometry_sha256']==sha(V/'support_change_geometry.json');sp=Path(native['samples_path']);assert sha(sp)==native['samples_sha256'];saved=torch.load(sp,weights_only=True);n=0;normchecks=0;ineligible=0;maxerr=0.;maxnorm=0.
 for file in r['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];b=torch.load(p,weights_only=True);gf=next(x for x in geo['files'] if x['pair_id']==file['pair_id']);gp=Path(gf['path']);assert sha(gp)==gf['sha256'];gb=torch.load(gp,weights_only=True)
  for entry in gb['entries']:
   ident=entry['identity'];ms=':'.join(str(ident[k]) for k in ['kind','seed','arm']);ep=ident['key'].rsplit('|',1)[1];key=ms+'|'+ep+'|'+ident['region'];full=entry['deltas']['full'].double().numpy();within=entry['deltas']['within'].double().numpy();fn=np.linalg.norm(full.ravel());wn=np.linalg.norm(within.ravel());eligible=fn>0 or wn==0;ratio=wn/fn if fn>0 else 0.;expected=(full*ratio).astype(np.float32);actual=b['controls'][key].numpy();assert np.array_equal(actual,expected);cn=np.linalg.norm(actual.astype(float).ravel());maxnorm=max(maxnorm,abs(cn-wn)/max(wn,1.));normchecks+=1
   rr=[x for x in r['rows'] if all(x[k]==v for k,v in ident.items())];assert len(rr)==2
   for row in rr:
    if row['status']!='evaluated':assert not eligible and row['term']=='matched';ineligible+=1;continue
    assert row['term']=='full' or eligible
    for k,val in [('full_norm',fn),('within_norm',wn),('scale_ratio',ratio),('matched_norm',cn)]:assert np.isclose(row[k],val,rtol=1e-12,atol=1e-10)
    base=saved['baselines'][row['key']];bl=lp(base['logits']);teacher=lp(saved['changed'][row['key']+'|1|'+row['region']])-bl;sl=lp(b['changed'][key+'|'+row['term']]);er=float((np.exp(bl)*(sl-bl-teacher)**2).sum());en=float((np.exp(bl)*teacher**2).sum());g=base['groups'];v=base['variant'];tm=margin(bl+teacher,g,v)-margin(bl,g,v);me=(margin(sl,g,v)-margin(bl,g,v)-tm)**2
    for k,val in [('response_error',er),('target_energy',en),('margin_error',me),('margin_target_energy',tm**2)]:maxerr=max(maxerr,abs(row[k]-val));assert np.isclose(row[k],val,rtol=2e-9,atol=2e-10)
    if row['region']=='identity':assert er==en==0
    n+=1
 assert normchecks==1512 and n+ineligible==3024 and maxnorm<1e-6
 op=V/'support_norm_audit.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),helper_sha256=sha(P/'audit_support_change_diagnostic.py'),result_sha256=sha(rp),response_vectors=n,ineligible_rows=ineligible,norm_checks=normchecks,max_norm_relative_error=maxnorm,max_metric_abs_error=maxerr,scope='All saved matched directions exactly recomputed in NumPy; all eligible saved final-response metrics independently checked. Full-direction native controls replay M64 in producer.'),indent=2)+'\n');print('PASS',n,normchecks,ineligible,maxnorm,maxerr)
if __name__=='__main__':main()
