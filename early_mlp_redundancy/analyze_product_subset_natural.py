"""Matched raw/scaled natural-retention comparison, all frozen candidates."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 r=json.loads((V/'product_subset_natural.json').read_text());a=json.loads((V/'product_subset_natural_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(V/'product_subset_natural.json')
 es=json.loads((V/'product_subset_models.json').read_text())['models'];lookup={tuple(e[k] for k in ['kind','seed','arm']):e for e in es};summaries=[]
 for s in r['summaries']:
  e=lookup[tuple(s[k] for k in ['kind','seed','arm'])];summaries.append(s|{k:e[k] for k in ['space','method','correction','size','scalar']})
 rows=[]
 for space in sorted({e['space'] for e in es}):
  full=next(s for s in summaries if s['space']==space and s['method']=='full')
  for method in ['energy','random9001','random9002']:
   raw=next(s for s in summaries if s['space']==space and s['method']==method and s['correction']=='raw');scaled=next(s for s in summaries if s['space']==space and s['method']==method and s['correction']=='scaled')
   rows.append({'space':space,'method':method,'full':full,'raw':raw,'scaled':scaled,'deltas_scaled_minus_raw':{k:scaled[k]-raw[k] for k in ['mean_KL','delta_CE','top1_agreement','local_relative_error']}})
 out={'status':'completed','source_sha256':sha(__file__),'result_sha256':sha(V/'product_subset_natural.json'),'audit_sha256':sha(V/'product_subset_natural_audit.json'),'comparisons':rows,'summaries':summaries,'scope':'All51 matched raw/scaled pairs with17 full controls. Local error here is relative to actual native write; M91 subset error used each space full reference. Natural KL/CE are not finite causal-response fidelity. No candidate selection.'}
 (V/'product_subset_natural_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
