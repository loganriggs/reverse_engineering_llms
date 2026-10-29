"""M63 all final-vector and delta metrics, saved decoder outputs, and aggregates."""
from pathlib import Path
import json,hashlib,numpy as np,torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lp(z):
 z=np.asarray(z,dtype=np.float64);z=z-z.max();return z-np.log(np.exp(z).sum())
def margin(z,g,v):
 z=np.asarray(z,dtype=np.float64)
 def lse(x):return x.max()+np.log(np.exp(x-x.max()).sum())
 return float(lse(z[g[v]])-lse(z[g[1-v]]))
def main():
 torch.set_num_threads(1);rp=V/'pca_factorial.json';r=json.loads(rp.read_text());assert r['protocol_sha256']==sha(V/'pca_factorial_execution.json');assert r['source_sha256']==sha(P/'probe_pca_factorial.py');mp=V/'pca_evaluation_models.json';assert r['models_sha256']==sha(mp);models=json.loads(mp.read_text())['models'];n=json.loads((V/'source_holdout_native.json').read_text());p=Path(n['samples_path']);assert sha(p)==n['samples_sha256'];native=torch.load(p,weights_only=True);nr={x['sample_key']:x for x in n['records']};states={};maxerr=0;maxdecode=0;checked=0;decoderchecks=0;calls=28+56
 fit=json.loads((V/'pca_encoder_fit.json').read_text());assert sha(Path(fit['state_path']))==fit['state_sha256'];state=np.load(fit['state_path'])
 def check(a,b):
  nonlocal maxerr
  assert np.isclose(a,b,rtol=1e-8,atol=1e-9),(a,b);maxerr=max(maxerr,abs(a-b))
 for file in r['files']:
  p=Path(file['path']);assert sha(p)==file['sha256'];s=torch.load(p,weights_only=True);pid=s['pair_id'];rr=[x for x in r['rows'] if x['key'].rsplit('|',1)[0]==pid]
  for e in models:
   ms=':'.join(str(e[k]) for k in ['kind','seed','arm']);u=state['basis'][:,:e['rank']];recs=[]
   for ep in [0,1]:
    rec=s['reconstructions'][ms+f'|{ep}'];q=rec['write'].numpy();ref=state['mean']+(s['writes'][ep].double().numpy()-state['mean'])@(u@u.T);assert np.allclose((s['writes'][ep].double().numpy()-state['mean'])@u,rec['code'].numpy(),rtol=1e-10,atol=1e-9);assert bool((rec['support']==1).all());error=np.linalg.norm(ref-q)/max(np.linalg.norm(q),1);assert error<1e-5;maxdecode=max(maxdecode,float(error));decoderchecks+=1;recs.append(q)
   for ep in [0,1]:
    key=pid+f'|{ep}';b=native['baselines'][key];z=b['logits'].numpy();prob=np.exp(lp(z));g=b['groups'];variant=b['variant'];base=s['baselines'][ms+f'|{ep}'].numpy()
    for region in ['changed_input_positions','final_position','identity']:
     sk=key+'|1|'+region;target=nr[sk];nd=np.zeros_like(s['writes'][ep].numpy());rd=np.zeros_like(nd);pos=target['recipient_positions'];dp=target['donor_positions'];nd[:,pos]=s['writes'][1-ep].numpy()[:,dp]-s['writes'][ep].numpy()[:,pos];rd[:,pos]=recs[1-ep][:,dp]-recs[ep][:,pos];assert np.array_equal(nd[:,pos],native['deltas'][sk]['delta'].numpy());geo=next(x for x in r['geometry'] if x['key']==key and x['region']==region and all(x[k]==e[k] for k in ['kind','seed','arm']));check(float(np.sum(nd.astype(np.float64)**2)),geo['target_delta_energy']);check(float(np.sum((rd.astype(np.float64)-nd.astype(np.float64))**2)),geo['delta_error']);check(float(np.sum(rd.astype(np.float64)**2)),geo['predicted_delta_energy']);native_response=lp(native['changed'][sk])-lp(z)
     for corner,delta,bg in [('reconstructed_native',nd,base),('native_reconstructed',rd,z),('reconstructed_reconstructed',rd,base)]:
      out=s['changed'][ms+f'|{ep}|'+region+'|'+corner].numpy();row=next(x for x in rr if x['key']==key and x['region']==region and x['corner']==corner and all(x[k]==e[k] for k in ['kind','seed','arm']));check(float(np.sum(prob*((lp(out)-lp(bg))-native_response)**2)),row['response_error']);effect=margin(out,g,variant)-margin(bg,g,variant);check((effect-target['margin_change'])**2,row['margin_error']);assert row['target_energy']==target['response_energy'];assert row['margin_target_energy']==target['margin_change']**2;calls+=bool(np.count_nonzero(delta));checked+=1
      if region=='identity':assert row['response_error']==0 and row['margin_error']==0 and geo['delta_error']==0
 for summary in r['summaries']:
  rows=[x for x in r['rows'] if all(x[k]==summary[k] for k in ['kind','seed','arm','region','corner']) and (summary['family'] is None or x['family']==summary['family'])];assert len(rows)==summary['n']
  for field,src in [('target_energy','target_energy'),('error_energy','response_error'),('margin_target_energy','margin_target_energy'),('margin_error_energy','margin_error')]:check(sum(x[src] for x in rows),summary[field])
  if summary['target_energy']:check(summary['error_energy']/summary['target_energy'],summary['relative_response_error'])
  else:assert summary['relative_response_error'] is None
 assert checked==504 and decoderchecks==56 and r['old_corner_replays']==0 and r['forward_count']==calls
 out=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(Path(__file__)),response_vectors_checked=checked,geometry_rows_checked=168,decoder_checks=decoderchecks,max_decoder_relative_error=maxdecode,max_metric_absolute_error=maxerr,forward_count=calls,scope='All full-vocabulary saved final responses, saved donor/recipient deltas, all geometry and summaries. Independent explicit PCA projector applied to every saved native write, with code and support checks. Native baselines replayed by producer; fixed M72 scores remain references, not rerun controls. No additive attribution or transformer reimplementation.')
 op=V/'pca_factorial_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
