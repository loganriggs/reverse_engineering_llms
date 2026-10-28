"""M32 descriptive scale/orthogonal and family concentration accounting."""
from pathlib import Path
import hashlib,json,math
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'native_response_geometry.json';r=json.loads(rp.read_text());assert json.loads((P/'native_response_geometry_audit.json').read_text())['result_sha256']==sha(rp);rows=[];maxerror=0
for x in r['comparisons']:
 er=x['reference_centered_energy'];ec=x['context_centered_energy'];cos=x['centered_cosine']
 if er>0 and ec>0:
  dot=cos*math.sqrt(er*ec);gain=dot/er;scale=(gain-1)**2*er;orth=ec-dot**2/er;assert orth>=-1e-12;orth=max(0.,orth)
 else:
  assert er==ec==x['centered_difference']==0;gain=None;scale=orth=0.
 error=abs(scale+orth-x['centered_difference']);assert error<1e-10;maxerror=max(maxerror,error);rows.append(dict(**x,parallel_gain=gain,parallel_change_energy=scale,orthogonal_change_energy=orth))
summaries=[]
for s in r['summaries']:
 if s['family'] is not None:continue
 rs=[x for x in rows if all(x[k]==s[k] for k in ['kind','seed','context'])];total=sum(x['centered_difference'] for x in rs);scale=sum(x['parallel_change_energy'] for x in rs);orth=sum(x['orthogonal_change_energy'] for x in rs);families={f:sum(x['centered_difference'] for x in rs if x['family']==f) for f in sorted({x['family'] for x in rs})};loo=[]
 for f in families:
  keep=[x for x in rs if x['family']!=f];den=sum(x['total_difference'] for x in keep);loo.append(dict(omitted=f,centered_fraction=sum(x['centered_difference'] for x in keep)/den if den else None))
 largest=max(families,key=families.get);row=dict(kind=s['kind'],seed=s['seed'],context=s['context'],centered_fraction=s['centered_fraction'],orthogonal_fraction_of_centered_change=orth/total if total else None,parallel_fraction_of_centered_change=scale/total if total else None,negative_gain_count=sum(x['parallel_gain'] is not None and x['parallel_gain']<0 for x in rs),largest_centered_change_family=largest,largest_family_share=families[largest]/total if total else None,family_centered_change=families,leave_one_family_out=loo);summaries.append(row);print(row['kind'],row['seed'],row['context'],'centered',round(row['centered_fraction'],4),'orthogonal',round(row['orthogonal_fraction_of_centered_change'],4),'largest',largest,round(row['largest_family_share'],4),'LOO centered range',min(x['centered_fraction'] for x in loo if x['centered_fraction'] is not None),max(x['centered_fraction'] for x in loo if x['centered_fraction'] is not None))
out=dict(result_sha256=sha(rp),rows=rows,summaries=summaries,max_additive_residual=maxerror,scope='Post-result descriptive projection: context centered response onto reference centered response under common reference weights. Parallel and orthogonal energies add exactly; not a fitted predictive model or internal causal attribution. Paired family omissions are correlated robustness accounting, not validation/model selection.');(P/'native_geometry_concentration.json').write_text(json.dumps(out,indent=2)+'\n')
