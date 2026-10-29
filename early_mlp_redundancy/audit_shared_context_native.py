"""Audit M53 complete population, arithmetic, no-ops and independent response checks."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=V/'shared_context_native.json';d=json.loads(rp.read_text());q=json.loads((V/'shared_context_native_protocol.json').read_text());assert d['protocol_sha256']==sha(V/'shared_context_native_protocol.json');assert d['source_sha256']==sha(P/'probe_shared_context_native.py');assert sha(V/'shared_context_inventory.json')==q['inventory_sha256'];rows=d['rows'];assert len(rows)==23040;keys=['key','seed','arm','component','predictor'];assert len({tuple(r[k] for k in keys) for r in rows})==len(rows)
 inv=json.loads((V/'shared_context_inventory.json').read_text());occ={}
 for o in inv['occurrences']:occ.setdefault(o['key'],[]).append(o)
 assert set(occ)=={r['key'] for r in rows} and len(occ)==192
 targets={}
 for r in rows:
  assert r['positions']==len(occ[r['key']]);assert r['eligible_positions']==sum(inv['eligible'][o['unit']] for o in occ[r['key']]);tk=tuple(r[k] for k in ['key','seed','arm','component']);t=targets.setdefault(tk,r)
  for k in ['target_noop','target_energy','target_delta_energy']:assert r[k]==t[k]
  assert np.isfinite(r['error_energy']) and np.isfinite(r['target_energy']);assert min(r['error_energy'],r['target_energy'])>=0
  if r['target_noop']:assert r['target_energy']==0 and r['target_delta_energy']==0 and r['predictor_noop'] and r['error_energy']==0
  if r['predictor']=='zero_response' or r['predictor_noop']:assert r['error_energy']==r['target_energy']
  if r['target_energy']:assert np.isclose(r['relative_error'],r['error_energy']/r['target_energy'],rtol=1e-12)
  else:assert r['relative_error'] is None
 assert len(targets)==4608;expected_calls=192+sum(not r['target_noop'] for r in targets.values())+sum(not r['predictor_noop'] for r in rows);assert d['forward_count']==expected_calls
 sk=['seed','arm','component','predictor','domain','split'];assert len(d['summaries'])==360
 for s in d['summaries']:
  rr=[r for r in rows if all(r[k]==s[k] for k in sk)];assert len(rr)==s['inputs'];en=sum(r['target_energy'] for r in rr);er=sum(r['error_energy'] for r in rr)
  assert en==s['target_energy'] and er==s['error_energy'];assert s['zero_energy_inputs']==sum(r['target_energy']==0 for r in rr);assert s['wins_vs_zero']==sum(r['error_energy']<r['target_energy'] for r in rr)
  assert s['relative_error']==(er/en if en else None)
 assert len(d['numpy_checks'])==80
 out=dict(status='passed',result_sha256=sha(rp),source_sha256=sha(Path(__file__)),rows=len(rows),targets=len(targets),summaries=360,forward_count=expected_calls,zero_target_interventions=sum(r['target_noop'] for r in targets.values()),numpy_full_output_checks=80,max_numpy_absolute_error=max(r['max_absolute_error'] for r in d['numpy_checks']),scope='Full inventory, source/protocol hashes, no-op/zero identities, forward accounting, all summary aggregation. Producer independently computes full-output logsoftmax response metrics in NumPy for fixed 80 shared-component examples; does not independently rerun model forwards.')
 op=V/'shared_context_native_audit.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
