"""Check lexical exclusions, independently reconstruct directions, and replay sampled outputs."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from audit_preamble_screen import kl,margin
P=Path(__file__).resolve().parent;path=P/'validation/lexical_direction_transfer.json'
r=json.loads(path.read_text());saved=torch.load('/tmp/emlp_p10_samples.pt',weights_only=True)
assert r['forward_count']==832 and len(r['rows'])==560 and len(r['folds'])==16
bank=[x.double().numpy().reshape(-1) for x in saved['bank']];assert len(bank)==len(r['bank'])==40
folds={}
for f in r['folds']:
 groups={}
 for i in f['training_bank_indices']:
  meta=r['bank'][i];assert meta['family']==f['family'] and meta['receiver']==f['receiver']
  lex=tuple(meta['lexical_pair']);assert list(lex)!=f['held_pair']
  groups.setdefault(lex,[]).append(bank[i]/np.linalg.norm(bank[i]))
 assert len(groups)==3
 expected={i for i,m in enumerate(r['bank']) if m['family']==f['family'] and m['receiver']==f['receiver'] and m['lexical_pair']!=f['held_pair']}
 assert expected==set(f['training_bank_indices']) and len(expected)==len(f['training_bank_indices'])
 v=np.mean([np.mean(g,axis=0) for g in groups.values()],axis=0);v/=np.linalg.norm(v)
 folds[(f['family'],tuple(f['held_pair']),f['receiver'])]=v
p8=json.loads((P/'validation/preamble_transfer.json').read_text());old={(x['id'],x['preamble_index'],x['receiver'],x['kind']):x for x in p8['rows'] if x['kind'] in ('context_specific','original_norm_matched')}
replay=[];direrrors=[];metricerrors=[]
for target in saved['targets']:
 rr=[x for x in r['rows'] if (x['id'],x['preamble_index'],x['receiver'])==(target['id'],target['preamble_index'],target['receiver'])]
 assert len(rr)==7 and sorted(x['kind'] for x in rr)==sorted(['native_response','same_lexical_direction','heldout_mean_direction']+['random']*4)
 row=rr[0];v=folds[(row['family'],tuple(row['lexical_pair']),row['receiver'])]
 direrrors.append(float(np.max(np.abs(v-target['shared'].double().numpy().reshape(-1)))))
 for x in rr:
  assert abs(x['edit_norm']/x['native_norm']-1)<1e-6
  if x['kind'] in ('native_response','same_lexical_direction'):
   kind={'native_response':'context_specific','same_lexical_direction':'original_norm_matched'}[x['kind']]
   other=old[(x['id'],x['preamble_index'],x['receiver'],kind)]
   replay.extend(abs(x[f]-other[f]) for f in ('native_donor_KL','remaining_donor_KL','donor_margin_gap','margin_gain'))
for x in saved['samples']:
 row=r['rows'][x['row']];a=x['teacher'].double().numpy();b=x['baseline'].double().numpy();c=x['candidate'].double().numpy()
 metricerrors.extend([abs(kl(a,c)-row['remaining_donor_KL']),abs(kl(a,b)-row['native_donor_KL']),abs(margin(c,x['groups'],x['variant'])-margin(b,x['groups'],x['variant'])-row['margin_gain'])])
for summary in r['summaries']:
 for kind,v in summary['arms'].items():
  rr=[x for x in r['rows'] if x['family']==summary['family'] and x['preamble_index']==summary['preamble_index'] and x['kind']==kind and (summary['lexical_pair'] is None or x['lexical_pair']==summary['lexical_pair'])]
  assert len(rr)==v['n']
  assert abs(v['KL_recovery']-(1-sum(x['remaining_donor_KL'] for x in rr)/sum(x['native_donor_KL'] for x in rr)))<1e-12
  assert abs(v['margin_recovery']-sum(x['margin_gain'] for x in rr)/sum(x['donor_margin_gap'] for x in rr))<1e-12
assert max(direrrors)<1e-6 and max(metricerrors)<1e-10
# Unit-normalize then scale vs multiply by a norm ratio can differ in float32.
assert max(replay)<1e-4
out=dict(status='passed',folds=16,unique_training_source_responses=40,test_directions=80,sampled_outputs=len(saved['samples']),max_direction_error=max(direrrors),max_metric_error=max(metricerrors),max_P8_replay_metric_error=max(replay),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(P/'validation/lexical_direction_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
