"""M30 paired context contrasts, all models/scales/families retained."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rp=P/'wrapper_crossover_results.json';r=json.loads(rp.read_text());assert json.loads((P/'wrapper_crossover_audit.json').read_text())['result_sha256']==sha(rp);contexts=['reference','passage','sample'];models=[];families=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  for mult in [1.,.1]:
   arms={'initial':'initial','joint':f'aggregate_gradient:{mult}:live','control':f'aggregate_KL:{mult}:live'};cells={}
   for context in contexts:
    cells[context]={name:next(x for x in r['summaries'] if (x['kind'],x['seed'],x['arm'],x['context'],x['family'])==(kind,seed,arm,context,None)) for name,arm in arms.items()}
   ratios={c:cells[c]['joint']['R']/cells[c]['control']['R'] for c in contexts};models.append(dict(kind=kind,seed=seed,multiplier=mult,cells=cells,joint_control_response_ratio=ratios));print(kind,seed,mult,'ratios', {k:round(v,4) for k,v in ratios.items()},'native binary/top1',{c:(cells[c]['initial']['native_binary_rate'],cells[c]['initial']['native_top1_rate']) for c in contexts})
   for fam in sorted({x['family'] for x in r['records']}):
    fc={c:{name:next(x for x in r['summaries'] if (x['kind'],x['seed'],x['arm'],x['context'],x['family'])==(kind,seed,arm,c,fam)) for name,arm in arms.items()} for c in contexts};families.append(dict(kind=kind,seed=seed,multiplier=mult,family=fam,cells=fc))
counts=[]
for mult in [1.,.1]:
 rs=[x for x in models if x['multiplier']==mult];fs=[x for x in families if x['multiplier']==mult];counts.append(dict(multiplier=mult,contexts={c:dict(model_response_wins=sum(x['joint_control_response_ratio'][c]<1 for x in rs),model_margin_wins=sum(x['cells'][c]['joint']['margin_error_sum']<x['cells'][c]['control']['margin_error_sum'] for x in rs),family_response_wins=sum(x['cells'][c]['joint']['R']<x['cells'][c]['control']['R'] for x in fs),ratio_worse_than_reference=sum(x['joint_control_response_ratio'][c]>x['joint_control_response_ratio']['reference'] for x in rs)) for c in contexts}))
print(json.dumps(counts,indent=2));out=dict(result_sha256=sha(rp),models=models,families=families,counts=counts,scope='Within-context objective ratios for identical task-span deltas/source/candidates. Context changes native response and competence, and may shift absolute task positions. Ratios of aggregate errors, not unweighted unstable per-direction ratios. Descriptive familiar-development evidence; no prefix interventions.');(P/'wrapper_crossover_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
