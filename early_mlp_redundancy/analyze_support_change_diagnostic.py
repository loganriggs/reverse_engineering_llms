"""M64 retain paired full/within/switch errors for all candidates and families."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=V/'support_change_responses.json';r=json.loads(rp.read_text());audit=json.loads((V/'support_change_audit.json').read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(rp);geo=json.loads((V/'support_change_geometry.json').read_text());models=json.loads((V/'balanced_response_training_results.json').read_text())['models'];rows=[];families=[]
 for e in models:
  for region in ['changed_input_positions','final_position']:
   ident={k:e[k] for k in ['kind','seed','arm']}|dict(region=region);rr=[x for x in r['rows'] if all(x[k]==v for k,v in ident.items())];gg=[x for x in geo['rows'] if all(x[k]==v for k,v in ident.items())];row=ident.copy();en=sum(x['native_energy'] for x in gg)
   for term in ['within','switch','full']:
    tt=[x for x in rr if x['term']==term];fr=[]
    for family in sorted({x['family'] for x in tt}):
     ff=[x for x in tt if x['family']==family];fe=sum(x['target_energy'] for x in ff);er=sum(x['response_error'] for x in ff);frow=ident|dict(term=term,family=family,response_error=er,target_energy=fe,relative_error=er/fe);families.append(frow);fr.append(er/fe)
    row[term+'_family_mean']=sum(fr)/len(fr);row[term+'_pooled']=sum(x['response_error'] for x in tt)/sum(x['target_energy'] for x in tt);row[term+'_local_error']=sum(x[term+'_native_error'] for x in gg)/en;row[term+'_energy_ratio']=sum(x[term+'_energy'] for x in gg)/en
   row['twice_cross_native_ratio']=2*sum(x['cross'] for x in gg)/en;row['donor_within_local_error']=sum(x['donor_within_native_error'] for x in gg)/en;row['donor_switch_local_error']=sum(x['donor_switch_native_error'] for x in gg)/en;row['support_change_fraction']=sum(x['changed_atoms'] for x in gg)/sum(x['union_atoms'] for x in gg);rows.append(row)
 summary=[]
 for region in ['changed_input_positions','final_position']:
  rs=[x for x in rows if x['region']==region];summary.append(dict(region=region,within_beats_full_family=sum(x['within_family_mean']<x['full_family_mean'] for x in rs),switch_beats_full_family=sum(x['switch_family_mean']<x['full_family_mean'] for x in rs),within_beats_zero_family=sum(x['within_family_mean']<1 for x in rs),switch_beats_zero_family=sum(x['switch_family_mean']<1 for x in rs),n=len(rs)))
 op=V/'support_change_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),rows=rows,families=families,summary=summary),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
