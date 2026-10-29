"""M91: frozen product subsets/scalars on all heldout natural documents."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from probe_product_subsets import spaces
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run():
 torch.set_num_threads(1);protocol=json.loads((V/'product_subset_transfer_protocol.json').read_text())
 for name,key in [('product_subset_results.json','training_result_sha256'),('product_subset_audit.json','training_audit_sha256'),('bilinear_input_natural.json','evaluation_source_sha256')]:assert sha(V/name)==protocol[key]
 train=json.loads((V/'product_subset_results.json').read_text());source=json.loads((V/'bilinear_input_natural.json').read_text());files=sorted(source['files'],key=lambda f:f['document']);assert [f['document'] for f in files]==list(range(32,64))
 xs=[];ys=[]
 for f in files:
  assert sha(f['path'])==f['sha256'];s=torch.load(f['path'],weights_only=True,mmap=True);assert s['document']==f['document'];xs.append(s['input'][0].double().numpy().copy());ys.append(s['write'][0].double().numpy().copy());print('verified document',f['document'],flush=True)
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];r=torch.load(ref['path'],weights_only=True)
 x=np.concatenate(xs);W=r['D'].double().numpy();H=(x@r['L'].double().numpy().T)*(x@r['R'].double().numpy().T);native=H@W.T
 replay=[]
 for j,y in enumerate(ys):
  err=native[j*128:(j+1)*128]-y;rel=float(np.linalg.norm(err)/np.linalg.norm(y));replay.append({'document':32+j,'relative_error':rel,'max_absolute_error':float(abs(err).max())});assert rel<=1e-5
 rows=[];pooled=[]
 def energy(v):return np.sum(v.reshape(32,128,-1)**2,axis=(1,2))
 for name,C,offset2 in spaces(W):
  Y=H@C.T;E=energy(Y)
  for tr in [t for t in train['rows'] if t['space']==name]:
   method,size,alpha=tr['method'],tr['size'],tr['scalar'];ix=np.array(train['rankings'][name][method][:size]);S=Y.copy() if size==4608 else H[:,ix]@C[:,ix].T;D=Y-S
   se=energy(S);err=energy(D);comp=energy(Y-alpha*S);dot=np.sum((S*Y).reshape(32,128,-1),axis=(1,2));cross=np.sum((S*D).reshape(32,128,-1),axis=(1,2))
   for j in range(33):
    E0=float(E[j]) if j<32 else float(E.sum());sel=float(se[j]) if j<32 else float(se.sum());raw=float(err[j]) if j<32 else float(err.sum());co=float(comp[j]) if j<32 else float(comp.sum());dp=float(dot[j]) if j<32 else float(dot.sum());cr=float(cross[j]) if j<32 else float(cross.sum());offset=(128 if j<32 else 4096)*offset2
    row={'space':name,'method':method,'size':size,'document':32+j if j<32 else 'pooled','scalar':alpha,'dynamic_energy':E0,'raw_relative_squared':raw/E0,'raw_affine_relative_squared':raw/(E0+offset),'compensated_relative_squared':co/E0,'compensated_affine_relative_squared':co/(E0+offset),'cosine':dp/np.sqrt(sel*E0),'selected_omitted_dot':cr}
    if size==4608:assert row['raw_relative_squared']==0 and row['compensated_relative_squared']<1e-20
    (rows if j<32 else pooled).append(row)
  print('completed',name,flush=True)
 out={'status':'completed_pending_audit','source_sha256':sha(__file__),'space_helper_sha256':sha(P/'probe_product_subsets.py'),'protocol_sha256':sha(V/'product_subset_transfer_protocol.json'),'files':files,'native_replay':replay,'rows':rows,'pooled':pooled,'forward_count':0,'evaluation_positions':4096};assert len(rows)==8160 and len(pooled)==255
 (V/'product_subset_transfer_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
