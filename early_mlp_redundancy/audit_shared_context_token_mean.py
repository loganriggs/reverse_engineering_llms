"""Independent saved-table identities, coverage and metric replay for M52."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);rp=V/'shared_context_token_mean.json';r=json.loads(rp.read_text());sp=Path(r['mean_state_path']);assert sha(sp)==r['mean_state_sha256'];state=torch.load(sp,weights_only=True);old=json.loads((V/'shared_context_mean.json').read_text());oldp=Path(old['mean_state_path']);assert sha(oldp)==old['mean_state_sha256'];prior=torch.load(oldp,weights_only=True);assert torch.equal(state['permutation'],prior['permutation'])
 inv=json.loads((V/'shared_context_inventory.json').read_text());meta=json.loads((V/'natural_token_context.json').read_text());p=Path(meta['cache']);assert sha(p)==meta['cache_sha256'];data=torch.load(p,weights_only=True);ids,counts=np.unique(data['tokens'][:4096].numpy(),return_counts=True);assert np.array_equal(ids,state['tokens'].numpy());assert np.array_equal(counts,state['training_counts'].numpy());lookup=dict(zip(ids,counts));coverage=[]
 for c in r['coverage']:
  us=[u for u,e in zip(inv['units'],inv['eligible']) if e and (u['domain'],u['split'])==(c['domain'],c['split'])];cc=np.array([lookup.get(u['token'],0) for u in us]);assert (len(us),int((cc==0).sum()),int((cc==1).sum()),int((cc>=2).sum()))==(c['eligible'],c['unseen'],c['singleton'],c['repeated']);coverage.append(c)
 manifest=json.loads((V/'coactivation_groups.json').read_text());maxerr=0;tables=0
 for m in state['models']:
  e=next(e for e in manifest['models'] if e['seed']==m['seed'] and e['arm']==m['arm']);p=Path(e['path']);assert sha(p)==e['sha256'];st=torch.load(p,weights_only=True);dd=st['state']['dictionary'].numpy();D=dd/np.linalg.norm(dd,axis=0);prev=next(z for z in prior['models'] if z['seed']==m['seed'] and z['arm']==m['arm'])
  ex=m['token_tables']['exclusive']['coefficients'].numpy();sh=m['token_tables']['shared']['coefficients'].numpy();un=m['token_tables']['union']['coefficients'].numpy();assert np.allclose(ex+sh,un,rtol=1e-12,atol=1e-12)
  for comp,t in m['token_tables'].items():
   coeff=t['coefficients'].numpy();ii=t['atom_indices'].numpy();assert coeff.shape==(32,len(ids),8)
   assert np.array_equal(ii, m['token_tables']['union']['atom_indices'].numpy())
   for group in range(32):
    a=np.sum(coeff[group]*counts[:,None],axis=0)/counts.sum();decoded=a@D[:,ii[group]].astype(np.float64).T*float(st['scale']);expected=prev['means'][comp][group].numpy();err=np.linalg.norm(decoded-expected)/max(np.linalg.norm(expected),1);assert err<1e-6;maxerr=max(maxerr,float(err));tables+=1
 key=['seed','arm','component','baseline','domain','split'];replayed=0
 for row in r['rows']:
  if row['subset']=='all':
   parts=[z for z in r['rows'] if z['subset']!='all' and all(z[k]==row[k] for k in key)]
   assert len(parts)==3
   for metric in ['target_energy','error_energy','eligible_units','missing_prediction_nonzero_target']:assert np.isclose(sum(z[metric] for z in parts),row[metric],rtol=1e-12,atol=1e-7)
   if row['baseline']!='token_mean':
    z=next(z for z in old['rows'] if all(z[k]==row[k] for k in key))
    for metric in ['target_energy','error_energy']:assert np.isclose(z[metric],row[metric],rtol=1e-12,atol=1e-7)
    replayed+=1
  if row['subset']=='unseen' and row['baseline']=='token_mean':
   z=next(z for z in r['rows'] if z['subset']=='unseen' and z['baseline']=='group_mean' and all(z[k]==row[k] for k in key if k!='baseline'));assert np.isclose(z['error_energy'],row['error_energy'],rtol=1e-12,atol=1e-7)
 out=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(Path(__file__)),weighted_token_to_group_mean_checks=tables,max_numpy_decoder_relative_error=maxerr,prior_metric_replays=replayed,coverage=coverage,checks='Token counts independently reconstructed; coefficient additivity; weighted token means recover M51 group means; cohort partitions; unseen fallback; fixed permutation. Training coefficient aggregation independently NumPy checked in producer. No native response claims.')
 op=V/'shared_context_token_mean_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
