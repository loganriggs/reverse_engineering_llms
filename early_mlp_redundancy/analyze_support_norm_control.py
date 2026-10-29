"""M65 compare fixed-support changes to matched-magnitude full changes."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=V/'support_norm_responses.json';r=json.loads(rp.read_text());audit=json.loads((V/'support_norm_audit.json').read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(rp);prior=json.loads((V/'support_change_responses.json').read_text());models=json.loads((V/'balanced_response_training_results.json').read_text())['models'];rows=[];families=[]
 for e in models:
  for region in ['changed_input_positions','final_position']:
   ident={k:e[k] for k in ['kind','seed','arm']}|dict(region=region);new=[x for x in r['rows'] if all(x[k]==v for k,v in ident.items())];old=[x for x in prior['rows'] if all(x[k]==v for k,v in ident.items()) and x['term']=='within'];eligible={x['key'] for x in new if x['term']=='matched' and x['status']=='evaluated'};rr=[x for x in new+old if x['key'] in eligible];row=ident|dict(eligible_endpoints=len(eligible),ineligible_endpoints=28-len(eligible));ratios=[x['scale_ratio'] for x in rr if x['term']=='matched'];row.update(scale_ratio_min=min(ratios),scale_ratio_max=max(ratios))
   for term in ['full','within','matched']:
    tt=[x for x in rr if x['term']==term];fr=[];mr=[]
    for family in sorted({x['family'] for x in tt}):
     ff=[x for x in tt if x['family']==family];en=sum(x['target_energy'] for x in ff);er=sum(x['response_error'] for x in ff);men=sum(x['margin_target_energy'] for x in ff);mer=sum(x['margin_error'] for x in ff);families.append(ident|dict(term=term,family=family,target_energy=en,response_error=er,relative_error=er/en,margin_target_energy=men,margin_error=mer));fr.append(er/en);mr.append(mer/men if men else None)
    row[term+'_family_mean']=sum(fr)/len(fr);row[term+'_pooled']=sum(x['response_error'] for x in tt)/sum(x['target_energy'] for x in tt);row[term+'_margin_pooled']=sum(x['margin_error'] for x in tt)/sum(x['margin_target_energy'] for x in tt)
   rows.append(row)
 summary=[]
 for region in ['changed_input_positions','final_position']:
  rs=[x for x in rows if x['region']==region];s=dict(region=region,n=len(rs))
  for suffix in ['family_mean','pooled','margin_pooled']:
   for lhs,rhs in [('within','matched'),('matched','full')]:s[lhs+'_beats_'+rhs+'_'+suffix]=sum(x[lhs+'_'+suffix]<x[rhs+'_'+suffix] for x in rs)
  s['matched_below_zero_family']=sum(x['matched_family_mean']<1 for x in rs);summary.append(s)
 op=V/'support_norm_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),prior_sha256=sha(V/'support_change_responses.json'),rows=rows,families=families,summary=summary),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
