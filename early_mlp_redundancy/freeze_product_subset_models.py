import json
from pathlib import Path
from fit_pca_encoder import sha,V
P=Path(__file__).resolve().parent
p=json.loads((V/'product_subset_causal_protocol.json').read_text())
for f,k in [('product_subset_results.json','training_result_sha256'),('product_subset_audit.json','training_audit_sha256'),('product_subset_transfer_results.json','transfer_result_sha256')]:assert sha(V/f)==p[k]
a=json.loads((V/'product_subset_transfer_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(V/'product_subset_transfer_results.json')
t=json.loads((V/'product_subset_results.json').read_text());ref=json.loads((V/'native_input_reference.json').read_text());old=json.loads((V/'bilinear_input_evaluation_models.json').read_text());spaces=[dict(space='native',kind='native',seed=0,base_arm='native',state_path=None,state_sha256=None)]
for e in old['models']:
 if e['variant']=='input':spaces.append(dict(space=f"{e['kind']}_{e['seed']}_{e['base_arm']}",kind=e['kind'],seed=e['seed'],base_arm=e['base_arm'],state_path=e['path'],state_sha256=e['sha256']))
models=[]
for s in spaces:
 for method in ['full','energy','random9001','random9002']:
  for correction in (['raw'] if method=='full' else ['raw','scaled']):
   size=4608 if method=='full' else 2304;indices=list(range(4608)) if method=='full' else t['rankings'][s['space']][method][:size]
   alpha=1. if correction=='raw' else next(r['scalar'] for r in t['rows'] if r['space']==s['space'] and r['method']==method and r['size']==size)
   models.append(s|dict(arm=s['base_arm']+'__'+method+'__'+correction,variant='input',method=method,correction=correction,size=size,indices=indices,scalar=alpha,reference_path=ref['path'],reference_sha256=ref['sha256']))
assert len(models)==119 and len({(e['kind'],e['seed'],e['arm']) for e in models})==119
out=dict(status='frozen_before_preflight',protocol_sha256=sha(V/'product_subset_causal_protocol.json'),source_sha256=sha(__file__),encoder_source_sha256=sha(P/'product_subset_predictor.py'),transfer_audit_sha256=sha(V/'product_subset_transfer_audit.json'),models=models)
f=V/'product_subset_models.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print('FROZEN',len(models))
