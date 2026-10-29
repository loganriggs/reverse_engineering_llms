"""Independent NumPy scoring of every global calibration branch."""
import json
import numpy as np
import torch
from fit_pca_encoder import sha,V,A

def lp(t):
 x=t.double().numpy();x=x-x.max();return x-np.log(np.exp(x).sum())

def main():
 torch.set_num_threads(1);rp=V/'bilinear_input_joint_calibration.json';r=json.loads(rp.read_text());pp=V/'bilinear_input_joint_calibration_protocol.json';assert r['protocol_sha256']==sha(pp);p=json.loads(pp.read_text());tp=json.loads((V/'balanced_response_training_protocol.json').read_text());scales={(x['family'],x['region']):x['denominator'] for x in tp['scales']};cachepath=A/'emlp_m60_training_cache.pt';receipt=json.loads((V/'encoded_change_training_receipt.json').read_text());assert sha(cachepath)==receipt['cache_sha256'];cache=torch.load(cachepath,weights_only=True)['inputs'];seen=set();maximum=0;branches_checked=0;identities=0;Kvalues=[];Bvalues=[]
 def check(x,y):
  nonlocal maximum
  assert np.isclose(x,y,rtol=1e-7,atol=1e-9),(x,y);maximum=max(maximum,abs(x-y))
 for file in r['files']:
  assert sha(file['path'])==file['sha256'];samples=torch.load(file['path'],weights_only=True);assert len(samples)==132
  for sample in samples:
   index=sample['row'];assert index not in seen;seen.add(index);row=r['rows'][index];assert (row['kind'],row['seed'])==(file['kind'],file['seed']);assignment=tp['schedules'][str(row['seed'])]['task_updates'][row['assignment']];assert row['key'].rsplit('|',1)[0]==assignment['pair_id'] and row['family']==assignment['family'];c=cache[row['key']];n=lp(c['native']);prob=np.exp(n);K=float((prob*(n-lp(sample['intact']))).sum());check(K,row['K']);Bs=[];expected={(t['region'],corner) for t in c['targets'] for corner in ['native_reconstructed','reconstructed_native','reconstructed_reconstructed']};actual=set()
   for branch in sample['branches']:
    ident=(branch['region'],branch['corner']);assert ident not in actual;actual.add(ident);teacher=next(t for t in c['targets'] if t['region']==branch['region']);response=lp(branch['changed'])-lp(branch['base']);target=lp(teacher['native_changed'])-n;R=float((prob*(response-target)**2).sum());check(R,branch['R']);branches_checked+=1
    expected_base=c['native'] if branch['corner']=='native_reconstructed' else sample['intact'];assert torch.equal(branch['base'],expected_base)
    if branch['region']=='identity':assert R==0;identities+=1
    else:Bs.append(R/scales[c['family'],branch['region']])
   assert actual==expected;B=float(np.mean(Bs));check(B,row['B']);Kvalues.append(K);Bvalues.append(B)
  print('AUDIT',file['kind'],file['seed'],flush=True)
 assert seen==set(range(1056));assert {(f['kind'],f['seed']) for f in r['files']}=={(m['kind'],m['seed']) for m in p['models']}
 for m in p['models']:
  rows=[x for x in r['rows'] if (x['kind'],x['seed'])==(m['kind'],m['seed'])];assert {(x['assignment'],int(x['key'].rsplit('|',1)[1])) for x in rows}=={(i,e) for i in range(66) for e in [0,1]}
 check(float(np.mean(Kvalues)),r['mean_K']);check(float(np.mean(Bvalues)),r['mean_B']);check(float(np.mean(Kvalues)/np.mean(Bvalues)),r['joint_weight'])
 result=dict(status='passed',source_sha256=sha(__file__),calibration_sha256=sha(rp),rows=1056,response_vectors=branches_checked,identity_vectors=identities,max_metric_absolute_error=maximum,joint_weight=r['joint_weight'],scope='Independent all-vector NumPy scores and own-background centering, exactschedule/modelinventory, teacher references/scales/identityzeros/globalweight. No independent transformer implementation.')
 out=V/'bilinear_input_joint_calibration_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
