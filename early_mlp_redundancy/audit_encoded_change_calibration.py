"""M68 independent training-address, forward-loss and calibration audit."""
from pathlib import Path
import json,torch,numpy as np
from audit_support_change_diagnostic import sha,lp
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def main():
 torch.set_num_threads(1);rp=V/'encoded_change_calibration.json';r=json.loads(rp.read_text());assert r['status']=='completed' and r['loss_source_sha256']==sha(P/'encoded_change_loss.py');pr=json.loads((V/'balanced_response_training_protocol.json').read_text());cachepath=A/'emlp_m60_training_cache.pt';assert sha(cachepath)==r['cache_sha256'];cache=torch.load(cachepath,weights_only=True)['inputs'];nr=json.loads((V/'training_causal_coverage.json').read_text());addresses={(x['pair_id']+f"|{x['endpoint']}",x['region']):(x['recipient_positions'],x['donor_positions']) for x in nr['records'] if x['layer']==1 and x['status']=='evaluated'};den={(x['family'],x['region']):x['denominator'] for x in pr['scales']};n=0;maxerr=0.;computed={};vectors=0
 for file in r['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];blob=torch.load(p,weights_only=True)
  for key,b in blob.items():
   c=cache[key];native=lp(c['native']);intact=lp(b['intact']);K=float((np.exp(native)*(native-intact)).sum());raw=[];bal=[];vectors+=1
   for branch in b['branches']:
    pos,dp=addresses[key,branch['region']];expected=np.zeros_like(c['write'].numpy());expected[:,pos]=b['qd'].numpy()[:,dp]-b['qr'].numpy()[:,pos];assert np.array_equal(expected,branch['delta'].numpy());t=next(t for t in c['targets'] if t['region']==branch['region']);response=lp(t['native_changed'])-native;R=float((np.exp(native)*(lp(branch['changed'])-native-response)**2).sum());vectors+=1
    if branch['region']=='identity':assert R==0 and np.count_nonzero(expected)==0
    else:raw.append(R);bal.append(R/den[c['family'],branch['region']])
   computed[file['kind'],file['seed'],key]=dict(K=K,R_raw=np.mean(raw),R_balanced=np.mean(bal))
 for row in r['rows']:
  schedule=pr['schedules'][str(row['seed'])]['task_updates'];pairid,ep=row['key'].rsplit('|',1);assert schedule[row['task_update']]['pair_id']==pairid and schedule[row['task_update']]['family']==row['family'];vals=computed[row['kind'],row['seed'],row['key']]
  for k,val in vals.items():maxerr=max(maxerr,abs(row[k]-val));assert np.isclose(row[k],val,rtol=2e-9,atol=2e-10)
  n+=1
 assert n==792;K=sum(x['K'] for x in r['rows'])/792;B=sum(x['R_balanced'] for x in r['rows'])/792;assert K==r['mean_K'] and B==r['mean_R_balanced'] and K/B==r['encoded_weight'];old=json.loads((V/'balanced_response_calibration.json').read_text());assert all(row['K']==next(x['K'] for x in old['rows'] if all(x[k]==row[k] for k in ['kind','seed','task_update','key'])) for row in r['rows']);receipt=json.loads((V/'balanced_response_training_receipt.json').read_text());assert sha(P/'balanced_training_runtime.py')==receipt['runtime_sha256'];assert sha(P/'balanced_response_loss.py')==receipt['task_loss_sha256']
 op=V/'encoded_change_calibration_audit.json';assert not op.exists();op.write_text(json.dumps(dict(status='passed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),scheduled_rows=n,unique_recipients=len(computed),forward_vectors_checked=vectors,max_metric_abs_error=maxerr,unchanged_natural_runtime_sha256=receipt['runtime_sha256'],scope='Every saved encoded delta/address and final loss in NumPy, all scheduled arithmetic and exact old taskK values; natural retention code hash matches original M61 receipt. No optimizer updates.'),indent=2)+'\n');print('PASS',n,len(computed),vectors,maxerr)
if __name__=='__main__':main()
