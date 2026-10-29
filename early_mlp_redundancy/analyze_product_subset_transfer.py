"""Descriptive transfer summary with no candidate selection."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 d=json.loads((V/'product_subset_transfer_results.json').read_text());a=json.loads((V/'product_subset_transfer_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(V/'product_subset_transfer_results.json')
 tr=json.loads((V/'product_subset_results.json').read_text());lookup={(r['space'],r['method'],r['size']):r for r in tr['rows']};rows=[]
 for r in d['pooled']:
  old=lookup[r['space'],r['method'],r['size']];rows.append({k:r[k] for k in ['space','method','size']}|{k:{'training':old[k],'evaluation':r[k],'delta':r[k]-old[k]} for k in ['raw_relative_squared','compensated_relative_squared','raw_affine_relative_squared','compensated_affine_relative_squared']})
 non=[r for r in d['rows'] if r['size']!=4608];po=[r for r in d['pooled'] if r['size']!=4608]
 out={'status':'completed','source_sha256':sha(__file__),'result_sha256':sha(V/'product_subset_transfer_results.json'),'audit_sha256':sha(V/'product_subset_transfer_audit.json'),'comparisons':rows,'document_cases_nontrivial':len(non),'compensation_improves_document_cases':sum(r['compensated_relative_squared']<r['raw_relative_squared'] for r in non),'pooled_cases_nontrivial':len(po),'compensation_improves_pooled_cases':sum(r['compensated_relative_squared']<r['raw_relative_squared'] for r in po),'scope':'Descriptive training/evaluation comparison. Fixed training scalar, no selection. Correlated cases are not independent samples; these counts are not a statistical significance calculation.'}
 (V/'product_subset_transfer_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
