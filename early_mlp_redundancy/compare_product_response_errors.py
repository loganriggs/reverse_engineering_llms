"""All fixed raw/scaled comparisons of audited error decompositions."""
import json,sys
import numpy as np
from pathlib import Path
from fit_pca_encoder import sha,V
P=V.parent
FIELDS=['A2','B2','I2','E2','AB2','AI2','BI2','additive2']
def main():
 cohort=sys.argv[1];assert cohort in ['whole','source'];rp=V/f'product_error_interaction_{cohort}.json';d=json.loads(rp.read_text());a=json.loads((V/f'product_error_interaction_{cohort}_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp)
 ss=d['summaries'];lookup={(s['space'],s['method'],s['correction'],s['scope'],s['region']):s for s in ss};out=[]
 for r in ss:
  if r['method']=='full' or r['correction']!='raw':continue
  s=lookup[r['space'],r['method'],'scaled',r['scope'],r['region']];row={k:r[k] for k in ['space','kind','seed','arm','method','scope','region','n','null_targets']}
  row['terms']={}
  for agg in ['sum','pooled','family']:
   if r['E2_'+agg] is None:row['terms'][agg]=None;continue
   raw={f:r[f+'_'+agg] for f in FIELDS};sc={f:s[f+'_'+agg] for f in FIELDS};delta={f:sc[f]-raw[f] for f in FIELDS};parts={'isolated_norms':delta['A2']+delta['B2'],'error_alignment':delta['AB2'],'finite_interaction_net':delta['I2']+delta['AI2']+delta['BI2']}
   assert np.isclose(sum(parts.values()),delta['E2'],rtol=1e-7,atol=1e-10)
   row['terms'][agg]={'raw':raw,'scaled':sc,'delta':delta,'delta_E2_decomposition':parts,'both_isolated_improve':delta['A2']<0 and delta['B2']<0,'combined_worsens':delta['E2']>0,'isolated_sum_improves':parts['isolated_norms']<0}
  out.append(row)
 result={'status':'completed','source_sha256':sha(__file__),'input_sha256':sha(rp),'audit_sha256':sha(V/f'product_error_interaction_{cohort}_audit.json'),'cohort':cohort,'comparisons':out,'scope':'All fixed paired corrections. Changes in squared norms, alignment and net finite-interaction terms are an exact metric decomposition, not unique causal attribution or native channel cancellation.'}
 (V/f'product_error_interaction_{cohort}_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPARED',cohort,len(out))
if __name__=='__main__':main()
