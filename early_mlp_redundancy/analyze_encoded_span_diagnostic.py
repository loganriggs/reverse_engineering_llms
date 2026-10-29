"""M71 all candidate/site/term summaries, preserving family and energy weighting."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rp=V/'encoded_span_responses.json';r=json.loads(rp.read_text());a=json.loads((V/'encoded_span_audit.json').read_text());assert a['status']=='passed' and a['result_sha256']==sha(rp);geo=json.loads((V/'encoded_span_geometry.json').read_text());models=json.loads((V/'encoded_change_evaluation_models.json').read_text())['models'];rows=[];families=[]
 for e in models:
  for region in ['changed_input_positions','final_position']:
   ident={k:e[k] for k in ['kind','seed','arm']}|dict(region=region);rr=[x for x in r['rows'] if all(x[k]==v for k,v in ident.items())];gg=[x for x in geo['rows'] if all(x[k]==v for k,v in ident.items())];ne=sum(x['native_energy'] for x in gg);out=ident.copy()
   for term in ['recipient','union','dictionary','actual']:
    tt=[x for x in rr if x['term']==term];fr=[]
    for family in sorted({x['family'] for x in tt}):
     ff=[x for x in tt if x['family']==family];te=sum(x['target_energy'] for x in ff);er=sum(x['response_error'] for x in ff);families.append(ident|dict(term=term,family=family,target_energy=te,response_error=er,relative_response_error=er/te));fr.append(er/te)
    out[term+'_family']=sum(fr)/len(fr);out[term+'_pooled']=sum(x['response_error'] for x in tt)/sum(x['target_energy'] for x in tt);out[term+'_margin']=sum(x['margin_error'] for x in tt)/sum(x['margin_target_energy'] for x in tt);out[term+'_local']=sum(x[term+'_error'] for x in gg)/ne;out[term+'_energy_ratio']=sum(x[term+'_energy'] for x in gg)/ne
    if term!='actual':rank=[z for x in gg for z in x['ranks'][term]];out[term+'_rank_mean']=sum(rank)/len(rank)
   rows.append(out)
 summaries=[]
 for region in ['changed_input_positions','final_position']:
  for arm in [None,'intact_KL','KL_balanced_response','KL_encoded_balanced_response']:
   rr=[x for x in rows if x['region']==region and (arm is None or x['arm']==arm)];out=dict(region=region,arm=arm,n=len(rr))
   for suffix in ['family','pooled','margin']:
    for lhs,rhs in [('recipient','actual'),('union','actual'),('dictionary','union'),('dictionary','actual')]:out[lhs+'_beats_'+rhs+'_'+suffix]=sum(x[lhs+'_'+suffix]<x[rhs+'_'+suffix] for x in rr)
   for term in ['recipient','union','dictionary','actual']:out[term+'_below_zero_family']=sum(x[term+'_family']<1 for x in rr)
   summaries.append(out)
 op=V/'encoded_span_comparison.json';assert not op.exists();op.write_text(json.dumps(dict(status='completed',source_sha256=sha(Path(__file__)),result_sha256=sha(rp),geometry_sha256=sha(V/'encoded_span_geometry.json'),rows=rows,families=families,summaries=summaries,scope='Teacher-informed local least-squares projections, not achievable learned accuracy or downstream-optimal span bounds. All candidates retained.'),indent=2)+'\n');print(json.dumps([x for x in summaries if x['arm'] is None],indent=2))
if __name__=='__main__':main()
