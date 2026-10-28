"""M28 paired objective comparison on seen versus new discrete groups."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'unseen_direction_results.json';r=json.loads(rp.read_text());assert json.loads((P/'unseen_direction_audit.json').read_text())['result_sha256']==sha(rp);oldp=P/'aggregate_kl_comparison.json';old=json.loads(oldp.read_text());results=[];region_rows=[];family_rows=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  for mult in [1.,.1]:
   arms={'initial':'initial','joint':f'aggregate_gradient:{mult}:live','control':f'aggregate_KL:{mult}:live'}
   def cells(region,family):
    return {name:next(x for x in r['summaries'] if (x['kind'],x['seed'],x['arm'],x['region'],x['family'])==(kind,seed,arm,region,family)) for name,arm in arms.items()}
   c=cells(None,None);seen=next(x for x in old['training'] if (x['kind'],x['seed'],x['multiplier'])==(kind,seed,mult));base=dict(kind=kind,seed=seed,multiplier=mult,seen_R_control=seen['control_R'],seen_R_joint=seen['joint_R'],unseen_cells=c,unseen_joint_relative_to_control=c['joint']['R']/c['control']['R'],seen_joint_relative_to_control=seen['joint_R']/seen['control_R']);results.append(base);print(kind,seed,mult,'SEEN ratio',round(base['seen_joint_relative_to_control'],4),'UNSEEN ratio',round(base['unseen_joint_relative_to_control'],4),'UNSEEN norm initial/control/joint',*[round(c[a]['relative_response_error'],4) for a in ['initial','control','joint']])
   for region in ['prefix','task']:region_rows.append(dict(kind=kind,seed=seed,multiplier=mult,region=region,cells=cells(region,None)))
   for family in sorted({x['family'] for x in r['records']}):family_rows.append(dict(kind=kind,seed=seed,multiplier=mult,family=family,cells=cells(None,family)))
counts=[]
for mult in [1.,.1]:
 rs=[x for x in results if x['multiplier']==mult];regions=[x for x in region_rows if x['multiplier']==mult];families=[x for x in family_rows if x['multiplier']==mult];counts.append(dict(multiplier=mult,model_response_joint_better=sum(x['unseen_joint_relative_to_control']<1 for x in rs),region_response_joint_better=sum(x['cells']['joint']['R']<x['cells']['control']['R'] for x in regions),region_margin_joint_better=sum(x['cells']['joint']['margin_error_sum']<x['cells']['control']['margin_error_sum'] for x in regions),family_response_joint_better=sum(x['cells']['joint']['R']<x['cells']['control']['R'] for x in families),families_with_nonzero_effect=sum(x['cells']['initial']['native_response_energy_sum']>0 for x in families)))
print(json.dumps(counts,indent=2));out=dict(results_sha256=sha(rp),seen_comparison_sha256=sha(oldp),models=results,regions=region_rows,families=family_rows,counts=counts,scope='Seen versus unseen comparison uses within-panel joint/control ratios; raw error across panels is not directly comparable because native effects differ. Counts correlated descriptive development results; exclude no inactive or failed cases.');(P/'unseen_direction_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
