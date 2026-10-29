"""M91 independent per-document Gram/SVD audit; no evaluation fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run():
 torch.set_num_threads(1);result=json.loads((V/'product_subset_transfer_results.json').read_text());pr=json.loads((V/'product_subset_transfer_protocol.json').read_text());train=json.loads((V/'product_subset_results.json').read_text())
 for name,key in [('product_subset_results.json','training_result_sha256'),('product_subset_audit.json','training_audit_sha256'),('bilinear_input_natural.json','evaluation_source_sha256')]:assert sha(V/name)==pr[key]
 assert result['protocol_sha256']==sha(V/'product_subset_transfer_protocol.json');assert result['source_sha256']==sha(P/'probe_product_subset_transfer.py');assert result['space_helper_sha256']==sha(P/'probe_product_subsets.py')
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True);W=r['D'].double().numpy()
 spaces={'native':(W.T@W,0.)}
 for e in json.loads((V/'bilinear_input_evaluation_models.json').read_text())['models']:
  if e['variant']!='input':continue
  assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True)['state'];D=s['features.dictionary'].double().numpy();U,sv,_=np.linalg.svd(D,full_matrices=False);assert min(sv)>max(sv)*1e-10;C=U.T@W;c=s['output_mean'].double().numpy()+float(s['output_scale'])*s['features.bias'].double().numpy();b=c-U@(U.T@c)
  spaces[f"{e['kind']}_{e['seed']}_{e['base_arm']}"]=(C.T@C,float(b@b))
 lookup={(a['space'],a['method'],a['size'],a['document']):a for a in result['rows']+result['pooled']};totals={};maxabs=0.;maxratio=0.;count=0
 def check(a,b):
  nonlocal maxabs,maxratio
  delta=abs(a-b);maxabs=max(maxabs,delta);maxratio=max(maxratio,delta/(1e-8+1e-7*abs(b)));assert np.isclose(a,b,rtol=1e-7,atol=1e-8),(a,b)
 def compare(key,vals):
  nonlocal count
  E,sel,raw,comp,cross,offset=vals;row=lookup[key]
  for k,v in {'dynamic_energy':E,'raw_relative_squared':raw/E,'raw_affine_relative_squared':raw/(E+offset),'compensated_relative_squared':comp/E,'compensated_affine_relative_squared':comp/(E+offset),'cosine':(sel+cross)/np.sqrt(sel*E),'selected_omitted_dot':cross}.items():check(row[k],v)
  count+=1
 source=json.loads((V/'bilinear_input_natural.json').read_text());assert result['files']==source['files'];assert sorted(f['document'] for f in source['files'])==list(range(32,64));replays=[]
 for f in source['files']:
  assert sha(f['path'])==f['sha256'];s=torch.load(f['path'],weights_only=True,mmap=True);assert s['document']==f['document'];x=s['input'][0].double();h=torch.nn.functional.linear(x,r['L'].double())*torch.nn.functional.linear(x,r['R'].double());native=h@r['D'].double().T;y=s['write'][0].double();rel=float(torch.linalg.vector_norm(native-y)/torch.linalg.vector_norm(y));assert rel<1e-5;replays.append(rel);G=(h.T@h).numpy()
  for name,(CW,offset2) in spaces.items():
   M=G*CW;E=float(M.sum())
   for tr in [t for t in train['rows'] if t['space']==name]:
    method,size,alpha=tr['method'],tr['size'],tr['scalar'];order=np.array(train['rankings'][name][method]);ix=order[:size];omit=order[size:]
    sel=float(M[np.ix_(ix,ix)].sum());raw=float(M[np.ix_(omit,omit)].sum());cross=float(M[np.ix_(ix,omit)].sum());comp=raw+(1-alpha)**2*sel+2*(1-alpha)*cross
    key=(name,method,size,f['document']);assert lookup[key]['scalar']==alpha
    vals=np.array([E,sel,raw,comp,cross,len(x)*offset2]);compare(key,vals);base=key[:3];totals[base]=totals.get(base,np.zeros(6))+vals
  print('audited document',f['document'],flush=True)
 for key,vals in totals.items():compare((*key,'pooled'),vals)
 assert count==8415
 out={'status':'passed','source_sha256':sha(__file__),'result_sha256':sha(V/'product_subset_transfer_results.json'),'protocol_sha256':sha(V/'product_subset_transfer_protocol.json'),'per_document_cases':8160,'pooled_cases':255,'verified_source_files':32,'max_native_replay_relative':max(replays),'max_absolute_metric_discrepancy_including_unnormalized_energies':maxabs,'max_fraction_of_allowed_metric_tolerance':maxratio,'scope':'All inputs reloaded from independently hash-verified M88 samples. Torch product reconstruction, SVD output basis, explicit Gram subset/complement blocks. All frozen scalars and metrics checked; no downstream forwards or evaluation fitting.'}
 (V/'product_subset_transfer_audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
