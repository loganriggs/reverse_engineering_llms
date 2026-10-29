"""Structural provenance and coverage audit for M92; numerical checks are in preflight."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 m=json.loads((V/'product_subset_models.json').read_text());a=json.loads((V/'product_subset_preflight.json').read_text());t=json.loads((V/'product_subset_results.json').read_text());p=json.loads((V/'product_subset_causal_protocol.json').read_text());old=json.loads((V/'bilinear_input_evaluation_models.json').read_text())
 assert a['status']=='passed';assert a['source_sha256']==sha(P/'preflight_product_subset_models.py');assert a['models_sha256']==sha(V/'product_subset_models.json');assert a['encoder_sha256']==m['encoder_source_sha256']==sha(P/'product_subset_predictor.py');assert a['protocol_sha256']==m['protocol_sha256']==sha(V/'product_subset_causal_protocol.json');assert p['training_result_sha256']==sha(V/'product_subset_results.json')
 expected={'native':None}
 for e in old['models']:
  if e['variant']=='input':expected[f"{e['kind']}_{e['seed']}_{e['base_arm']}"]=e
 assert len(m['models'])==119;keys=set();refs=json.loads((V/'native_input_reference.json').read_text())
 for e in m['models']:
  key=e['space'],e['method'],e['correction'];assert key not in keys;keys.add(key);assert e['space'] in expected;assert e['reference_path']==refs['path'] and e['reference_sha256']==refs['sha256'];oe=expected[e['space']]
  if oe is None:assert e['state_path'] is None and e['kind']=='native' and e['seed']==0
  else:
   assert e['state_path']==oe['path'] and e['state_sha256']==oe['sha256'];assert all(e[k]==oe[k] for k in ['kind','seed','base_arm'])
  if e['method']=='full':assert e['indices']==list(range(4608)) and e['size']==4608 and e['scalar']==1 and e['correction']=='raw'
  else:
   assert e['size']==2304 and e['indices']==t['rankings'][e['space']][e['method']][:2304]
   alpha=1 if e['correction']=='raw' else next(x['scalar'] for x in t['rows'] if x['space']==e['space'] and x['method']==e['method'] and x['size']==2304)
   assert e['scalar']==alpha
  checks=[x for x in a['prediction_checks'] if (x['space'],x['method'],x['correction'])==key];assert len(checks)==1;check=checks[0];assert check['relative_write_error']<=1e-5 and check['relative_product_error']<=1e-5
  if e['method']=='full':assert check['full_reference_relative_error']<=1e-5
 assert keys=={(s,method,correction) for s in expected for method in ['full','energy','random9001','random9002'] for correction in (['raw'] if method=='full' else ['raw','scaled'])}
 assert a['input_access_guards_passed']==119 and a['intervention_constructions_checked']==2142 and a['native_full_KL']==0 and a['forward_count']==84
 assert len(a['native_smoke_conditions'])==72
 assert all(x['identity_exact'] for x in a['native_smoke_conditions'] if x['region']=='identity')
 out={'status':'passed','source_sha256':sha(__file__),'preflight_sha256':sha(V/'product_subset_preflight.json'),'manifest_sha256':sha(V/'product_subset_models.json'),'exact_manifest_candidates':119,'full_references':17,'native_smoke_conditions':72,'identity_smoke_conditions':24,'scope':'Independent manifest coverage/provenance/scalar/ranking checks and review of complete preflight coverage. Float64 prediction calculations, guarded native execution and intervention construction were executed by preflight; this receipt is not a second transformer implementation.'}
 (V/'product_subset_preflight_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('PASSED all119 manifest and preflight coverage checks')
if __name__=='__main__':run()
