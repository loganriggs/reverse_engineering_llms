"""M64 independent NumPy audit of every saved native response and local energy."""
from pathlib import Path
import json,hashlib,torch,numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(x):x=x.double().numpy();m=x.max();return x-m-np.log(np.exp(x-m).sum())
def lse(x):m=x.max();return m+np.log(np.exp(x-m).sum())
def margin(x,g,v):return lse(x[g[v]])-lse(x[g[1-v]])
def main():
 torch.set_num_threads(1);rp=V/'support_change_responses.json';r=json.loads(rp.read_text());gp=V/'support_change_geometry.json';geo=json.loads(gp.read_text());native=json.loads((V/'source_holdout_native.json').read_text());old=json.loads((V/'state_direction_factorial.json').read_text());assert r['status']=='completed' and r['geometry_sha256']==sha(gp) and r['source_sha256']==sha(P/'probe_support_change_diagnostic.py');assert geo['source_sha256']==sha(P/'prepare_support_change_diagnostic.py');sp=Path(native['samples_path']);assert sha(sp)==native['samples_sha256'];saved=torch.load(sp,weights_only=True);maxerr=0.;n=0;ng=0
 for file in r['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];b=torch.load(p,weights_only=True)
  for row in [x for x in r['rows'] if x['key'].rsplit('|',1)[0]==file['pair_id']]:
   base=saved['baselines'][row['key']];bl=lp(base['logits']);teacher=lp(saved['changed'][row['key']+'|1|'+row['region']])-bl;ep=int(row['key'].rsplit('|',1)[1]);ms=':'.join(str(row[k]) for k in ['kind','seed','arm']);z=b['changed'][ms+f'|{ep}|'+row['region']+'|'+row['term']];sl=lp(z);er=float((np.exp(bl)*(sl-bl-teacher)**2).sum());en=float((np.exp(bl)*teacher**2).sum());g=base['groups'];v=base['variant'];targetmargin=margin(bl+teacher,g,v)-margin(bl,g,v);me=(margin(sl,g,v)-margin(bl,g,v)-targetmargin)**2
   for key,val in [('response_error',er),('target_energy',en),('margin_error',me),('margin_target_energy',targetmargin**2)]:maxerr=max(maxerr,abs(row[key]-val));assert np.isclose(row[key],val,rtol=2e-9,atol=2e-10),(key,row[key],val)
   if row['region']=='identity':assert er==en==0
   n+=1
 for file in geo['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];b=torch.load(p,weights_only=True);of=next(x for x in old['files'] if x['pair_id']==file['pair_id']);orig=torch.load(of['path'],weights_only=True);assert sha(Path(of['path']))==of['sha256']
  for entry in b['entries']:
   ident=entry['identity'];row=next(x for x in geo['rows'] if all(x[k]==v for k,v in ident.items()));ep=int(ident['key'].rsplit('|',1)[1]);target=next(x for x in native['records'] if x['sample_key']==ident['key']+'|1|'+ident['region']);nd=orig['writes'][1-ep][:,target['donor_positions']].double().numpy()-orig['writes'][ep][:,target['recipient_positions']].double().numpy();assert np.isclose((nd**2).sum(),row['native_energy'],rtol=1e-12)
   for term,d in entry['deltas'].items():
    dn=d.double().numpy()
    for key,val in [(term+'_energy',(dn**2).sum()),(term+'_native_error',((dn-nd)**2).sum())]:assert np.isclose(row[key],val,rtol=1e-12,atol=1e-8)
   for pre in ['', 'donor_']:assert np.isclose(row[pre+'cross'],(entry['deltas'][pre+'within'].double().numpy()*entry['deltas'][pre+'switch'].double().numpy()).sum(),rtol=1e-12,atol=1e-8)
   ng+=1
 assert n==4536 and ng==1512
 op=V/'support_change_audit.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),geometry_sha256=sha(gp),response_vectors=n,geometry_rows=ng,max_metric_abs_error=maxerr,scope='All saved final full-vocabulary responses and local energies independently recomputed in NumPy. Independent gate/encoder/decomposition checks are in geometry preflight. No independent transformer implementation.'),indent=2)+'\n');print('PASS',n,ng,maxerr)
if __name__=='__main__':main()
