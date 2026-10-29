"""M46 independent prefix inventory and explicit all-pairs error audit."""
from pathlib import Path
import collections,json,hashlib
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert np.isclose(a,b,rtol=1e-8,atol=1e-5),(a,b)
def main():
 torch.set_num_threads(1);rp=P/'native_half_pairing.json';r=json.loads(rp.read_text());pp=P/'native_half_pairing_protocol.json';q=json.loads(pp.read_text());assert sha(pp)==r['protocol_sha256'];assert sha(P.parent/'probe_native_half_pairing.py')==q['source_sha256'];mp=P/'native_half_token_controls.json';assert sha(mp)==r['models_sha256']==q['models_sha256'];cp=P/'natural_token_context_cohort.json';assert sha(cp)==q['cohort_sha256'];cache=Path(r['cache_path']);assert sha(cache)==r['cache_sha256'];d=torch.load(cache,weights_only=True);assert d['protocol_sha256']==sha(pp);assert r['forward_count']==192
 panel=json.loads((P/'training_source_context_audit.json').read_text());c=json.loads(cp.read_text());train=set(json.loads((P/'diverse_supervision_split.json').read_text())['train_pair_ids']);inputs=[]
 for pair in panel['rows']:
  for ep in [0,1]:inputs.append((f"task:{pair['id']}:{ep}",'task','train' if pair['id'] in train else 'evaluation',pair['token_ids'][ep]))
 for doc in range(32,64):inputs.append((f'natural:{doc}','natural','evaluation',c['token_ids'][doc]))
 seen=set();expected=[];occ=collections.Counter()
 for source,domain,part,tokens in inputs:
  for pos,tok in enumerate(tokens):
   occ[domain+':'+part]+=1;k=(domain,part,tuple(tokens[:pos+1]))
   if k in seen:continue
   seen.add(k);expected.append(dict(domain=domain,split=part,token=tok,prefix=tokens[:pos+1],source_key=source,position=pos))
 assert expected==d['metadata'] and dict(occ)==d['occurrences'];groups=collections.defaultdict(list)
 for i,x in enumerate(expected):groups[x['domain'],x['split'],x['token']].append(i)
 rows={(x['domain'],x['split'],x['token'],x['layer'],x['removed']):x for x in r['rows']};assert len(rows)==len(r['rows']);checked=set();maxerr=0
 for entry in json.loads(mp.read_text())['models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];s=torch.load(path,weights_only=True);lookup={int(t):i for i,t in enumerate(s['token_ids'])};X=d['halves'][f"{entry['layer']}:{entry['retained']}"].numpy().astype(float);Y=d['halves'][f"{entry['layer']}:{entry['removed']}"].numpy().astype(float)
  for (domain,part,tok),idx in groups.items():
   if tok not in lookup or len(idx)<2:continue
   key=domain,part,tok,entry['layer'],entry['removed'];x=rows[key];checked.add(key);j=lookup[tok];pred=(X[idx]-s['token_input_mean'][j].numpy())@s['U'].numpy()@s['B'].numpy();y=Y[idx]-s['token_target_mean'][j].numpy();n=len(idx)
   # Explicit recipient/donor distance matrix, independent of total-vector formula.
   dist=np.sum(y*y,axis=1)[:,None]+np.sum(pred*pred,axis=1)[None,:]-2*y@pred.T;paired=float(np.trace(dist));wrong=float((dist.sum()-np.trace(dist))/(n-1));close(paired,x['paired_error']);close(wrong,x['wrong_context_expected_error']);close(float(np.sum(y*y)),x['token_mean_error']);close(wrong-paired,x['pairing_gain']);close(2*n/(n-1)*np.sum((pred-pred.mean(0))*(y-y.mean(0))),x['pairing_gain']);assert x['contexts']==n;maxerr=max(maxerr,abs(wrong-x['wrong_context_expected_error'])/max(abs(wrong),1))
 assert checked==set(rows)
 for cov in r['coverage']:
  gg={tok:idx for (dom,part,tok),idx in groups.items() if (dom,part)==(cov['domain'],cov['split'])};eligible={t:ix for t,ix in gg.items() if t in lookup and len(ix)>=2};assert cov['eligible_prefixes']==sum(map(len,eligible.values())) and cov['eligible_token_types']==len(eligible);assert cov['unique_prefixes']==sum(map(len,gg.values()));assert cov['unknown_token_prefixes']==sum(len(ix) for t,ix in gg.items() if t not in lookup);assert cov['known_single_context_prefixes']==sum(len(ix) for t,ix in gg.items() if t in lookup and len(ix)==1);assert cov['original_occurrences']==occ[cov['domain']+':'+cov['split']]
 for summary in r['summaries']:
  rr=[x for x in r['rows'] if all(x[k]==summary[k] for k in ['domain','split','layer'])];assert len(rr)==summary['token_mask_groups'];assert sum(x['contexts'] for x in rr)==summary['context_mask_cases']
  for f in ['paired_error','wrong_context_expected_error','token_mean_error','pairing_gain']:close(sum(x[f] for x in rr),summary[f])
  close(summary['paired_error']/summary['wrong_context_expected_error'],summary['paired_over_wrong']);close(summary['paired_error']/summary['token_mean_error'],summary['paired_over_token_mean'])
 # Existing independently audited native samples bind new cache to previous captures.
 prior=json.loads((P/'native_half_token_transfer.json').read_text());assert prior['checkpoint_sha256']==r['checkpoint_sha256'];sp=Path(prior['samples_path']);assert sha(sp)==prior['samples_sha256'];samples=torch.load(sp,weights_only=True);index={(x['domain'],x['split'],tuple(x['prefix'])):i for i,x in enumerate(expected)};replays=0
 for sample in samples:
  row=prior['rows'][sample['row']]
  if row['arm']!='rank32':continue
  ids=sample['ids'][0].tolist();ii=[index[row['domain'],row['split'],tuple(ids[:pos+1])] for pos in range(len(ids))];other='half_23' if row['removed']=='half_01' else 'half_01';assert torch.equal(d['halves'][f"{row['layer']}:{other}"][ii],sample['retained'][0]);replays+=1
 out=dict(status='passed',result_sha256=sha(rp),unique_prefixes=len(expected),token_mask_groups_checked=len(checked),explicit_all_pairs_relative_error_max=maxerr,prior_native_captures_exact=replays,scope='Independent prefix reconstruction, all eligibility/exclusion counts, explicit NumPy all-pairs squared errors and summaries; six previous native capture replays. Statistical pairing test, not causal output validation.')
 (P/'native_half_pairing_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
