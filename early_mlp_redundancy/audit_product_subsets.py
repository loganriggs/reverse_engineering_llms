"""Independent Gram/SVD audit of every M90 subset; does not use producer metrics."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 torch.set_num_threads(1);result=json.loads((V/'product_subset_results.json').read_text());p=json.loads((V/'product_subset_protocol.json').read_text())
 assert result['protocol_sha256']==sha(V/'product_subset_protocol.json');assert result['source_sha256']==sha(P/'probe_product_subsets.py')
 assert p['manifest_sha256']==sha(V/'bilinear_input_evaluation_models.json');assert p['reference_receipt_sha256']==sha(V/'native_input_reference.json')
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256']
 r=torch.load(ref['path'],weights_only=True);x=r['z'][r['indices']].double();h=torch.nn.functional.linear(x,r['L'].double())*torch.nn.functional.linear(x,r['R'].double());G=(h.T@h).numpy();W=r['D'].double().numpy();n=len(x)
 entries=[None]+[e for e in json.loads((V/'bilinear_input_evaluation_models.json').read_text())['models'] if e['variant']=='input'];count=0;maxabs=0.;inventories=0
 def check(a,b):
  nonlocal maxabs
  maxabs=max(maxabs,abs(a-b));assert np.isclose(a,b,rtol=1e-7,atol=1e-8),(a,b)
 for e in entries:
  if e is None:name='native';C=W;offset2=0.
  else:
   assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True)['state'];D=s['features.dictionary'].double().numpy();U,sv,_=np.linalg.svd(D,full_matrices=False);assert min(sv)>max(sv)*1e-10
   C=U.T@W;c=s['output_mean'].double().numpy()+float(s['output_scale'])*s['features.bias'].double().numpy();b=c-U@(U.T@c);offset2=float(b@b)
   name=f"{e['kind']}_{e['seed']}_{e['base_arm']}"
  M=G*(C.T@C);tot=float(M.sum());diag=np.diag(M);score=diag/n;rowmass=M.sum(axis=1)
  inv=next(i for i in result['inventories'] if i['space']==name)
  for k,v in {'dynamic_energy':tot,'individual_energy':float(diag.sum()),'net_cross_energy':float(tot-diag.sum()),'affine_offset_energy':n*offset2,'diagonal_over_total':float(diag.sum()/tot)}.items():check(inv[k],v)
  inventories+=1
  for method,order in result['rankings'][name].items():
   ixall=np.array(order);expected=np.argsort(-score,kind='stable') if method=='energy' else np.random.default_rng(int(method[6:])).permutation(len(score));assert np.array_equal(ixall,expected)
   for size in p['subsets']:
    ix=ixall[:size];sel=float(M[np.ix_(ix,ix)].sum());dot=float(rowmass[ix].sum());err=tot+sel-2*dot;alpha=dot/sel;comp=tot-dot*dot/sel
    vals={'raw_relative_squared':err/tot,'raw_affine_relative_squared':err/(tot+n*offset2),'retained_individual_fraction':float(diag[ix].sum()/diag.sum()),'cosine':dot/np.sqrt(sel*tot),'scalar':alpha,'compensated_relative_squared':comp/tot,'compensated_affine_relative_squared':comp/(tot+n*offset2),'selected_energy':sel,'selected_omitted_dot':float(M[np.ix_(ix,np.setdiff1d(np.arange(len(score)),ix))].sum())}
    row=next(a for a in result['rows'] if a['space']==name and a['method']==method and a['size']==size)
    for k,v in vals.items():check(row[k],v)
    count+=1
  print('audited',name,flush=True)
 assert count==255 and inventories==17
 out={'status':'passed','source_sha256':sha(__file__),'result_sha256':sha(V/'product_subset_results.json'),'protocol_sha256':sha(V/'product_subset_protocol.json'),'rows_checked':count,'inventories_checked':inventories,'rankings_checked':51,'max_absolute_discrepancy_all_metrics_including_unnormalized_energies':maxabs,'scope':'Independent torch product reconstruction and Gram metrics; SVD output basis rather than producer QR. All rankings, subset metrics and signed energy inventories checked. Not an independent native transformer implementation or causal validation.'}
 (V/'product_subset_audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
