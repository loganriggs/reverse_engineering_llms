"""M71 all native responses and independent QR-based projection audit."""
from pathlib import Path
import json,torch,numpy as np
from torch.nn import functional as F
from audit_support_change_diagnostic import sha,lp,margin
P=Path(__file__).resolve().parent;V=P/'validation'
def en(x):return float(np.square(x).sum())
def main():
 torch.set_num_threads(1);rp=V/'encoded_span_responses.json';r=json.loads(rp.read_text());gp=V/'encoded_span_geometry.json';geo=json.loads(gp.read_text());assert r['status']=='completed' and r['geometry_sha256']==sha(gp) and r['source_sha256']==sha(P/'probe_encoded_span_diagnostic.py');assert geo['source_sha256']==sha(P/'prepare_encoded_span_diagnostic.py');old=json.loads((V/'encoded_factorial.json').read_text());models=json.loads((V/'encoded_change_evaluation_models.json').read_text())['models'];D={}
 for e in models:
  p=Path(e['path']);assert sha(p)==e['sha256'];s=torch.load(p,weights_only=True);D[tuple(e[k] for k in ['kind','seed','arm'])]=F.normalize(s['state']['dictionary'],dim=0).double().numpy()
 native=json.loads((V/'source_holdout_native.json').read_text());p=Path(native['samples_path']);assert sha(p)==native['samples_sha256'];saved=torch.load(p,weights_only=True);bases={};n=0;ng=0;maxerr=0.;maxprojection=0.
 for file in r['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];b=torch.load(p,weights_only=True);gf=next(x for x in geo['files'] if x['pair_id']==file['pair_id']);gp_=Path(gf['path']);assert sha(gp_)==gf['sha256'];gb=torch.load(gp_,weights_only=True);of=next(x for x in old['files'] if x['pair_id']==file['pair_id']);op=Path(of['path']);assert sha(op)==of['sha256'];orig=torch.load(op,weights_only=True)
  for entry in gb['entries']:
   ident=entry['identity'];mk=tuple(ident[k] for k in ['kind','seed','arm']);ms=':'.join(map(str,mk));ep=int(ident['key'].rsplit('|',1)[1]);reg=ident['region'];row=next(x for x in geo['rows'] if all(x[k]==v for k,v in ident.items()));pos=entry['positions'];dp=entry['donor_positions'];nd=(orig['writes'][1-ep].numpy()[:,dp]-orig['writes'][ep].numpy()[:,pos]).astype(float);ne=en(nd);actual=orig['reconstructions'][ms+f'|{1-ep}']['write'].numpy()[:,dp]-orig['reconstructions'][ms+f'|{ep}']['write'].numpy()[:,pos];assert np.array_equal(actual,entry['deltas']['actual'].numpy());assert np.isclose(row['native_energy'],ne,rtol=1e-12)
   for term in ['recipient','union','dictionary']:
    pred=np.zeros_like(nd)
    for j in range(len(pos)):
     inds=entry['selected'][term][j] if term!='dictionary' else list(range(128));key=(mk,tuple(inds))
     if key not in bases:
      if inds:
       q,tri=np.linalg.qr(D[mk][:,inds],mode='reduced');u,sv,_=np.linalg.svd(tri,full_matrices=False);bases[key]=q@u[:,sv>1e-10*sv[0]]
      else:bases[key]=np.zeros((1152,0))
     u=bases[key];assert u.shape[1]==entry['ranks'][term][j];pred[0,j]=u@(u.T@nd[0,j]);assert set(entry['selected']['recipient'][j])<=set(entry['selected']['union'][j])
    applied=entry['deltas'][term].double().numpy();rel=en(applied-pred)/max(en(pred),1.);maxprojection=max(maxprojection,rel);assert rel<1e-12
   for term,delta in entry['deltas'].items():
    d=delta.double().numpy();assert np.isclose(row[term+'_error'],en(d-nd),rtol=1e-12,atol=1e-7) and np.isclose(row[term+'_energy'],en(d),rtol=1e-12,atol=1e-7)
   tol=1e-6*max(ne,1.);assert row['dictionary_error']<=row['union_error']+tol and row['union_error']<=row['recipient_error']+tol and row['union_error']<=row['actual_error']+tol;ng+=1
  for row in [x for x in r['rows'] if x['key'].rsplit('|',1)[0]==file['pair_id']]:
   base=saved['baselines'][row['key']];bl=lp(base['logits']);teacher=lp(saved['changed'][row['key']+'|1|'+row['region']])-bl;ep=int(row['key'].rsplit('|',1)[1]);ms=':'.join(str(row[k]) for k in ['kind','seed','arm']);sl=lp(b['changed'][ms+f'|{ep}|'+row['region']+'|'+row['term']]);er=float((np.exp(bl)*(sl-bl-teacher)**2).sum());te=float((np.exp(bl)*teacher**2).sum());g=base['groups'];v=base['variant'];tm=margin(bl+teacher,g,v)-margin(bl,g,v);me=(margin(sl,g,v)-margin(bl,g,v)-tm)**2
   for k,val in [('response_error',er),('target_energy',te),('margin_error',me),('margin_target_energy',tm**2)]:maxerr=max(maxerr,abs(row[k]-val));assert np.isclose(row[k],val,rtol=2e-9,atol=2e-10)
   if row['region']=='identity':assert er==te==0
   n+=1
  print('AUDIT_PAIRS',len({x['pair_id'] for x in r['files'][:r['files'].index(file)+1]}),flush=True)
 assert n==6048 and ng==1512 and r['full_corner_replays']==1512
 op=V/'encoded_span_audit.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),helper_sha256=sha(P/'audit_support_change_diagnostic.py'),result_sha256=sha(rp),geometry_sha256=sha(V/'encoded_span_geometry.json'),response_vectors=n,geometry_rows=ng,max_projection_relative_squared_error=maxprojection,max_response_metric_absolute_error=maxerr,scope='All applied projections independently rebuilt using QR then small-matrix SVD, ranks/local errors/nesting checked; all final response vectors scored in NumPy. Gate checks are in local preflight. All1512 actual encoded controls replay M70 exactly in producer.'),indent=2)+'\n');print('PASS',n,ng,maxprojection,maxerr,flush=True)
if __name__=='__main__':main()
