"""M29 compare joint/control fidelity within each fixed intervention generator."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
np=P/'basis_transfer_native_fidelity.json';new=json.loads(np.read_text());assert json.loads((P/'basis_transfer_native_fidelity_audit.json').read_text())['result_sha256']==sha(np);jp=P/'finite_candidate_native_fidelity.json';cp=P/'aggregate_kl_native_fidelity.json';j=json.loads(jp.read_text());c=json.loads(cp.read_text());old=dict(summaries=j['summaries']+[x for x in c['summaries'] if x['arm']!='initial'],records=j['records']+[x for x in c['records'] if x['arm']!='initial']);rows=[];energy=[]
for kind in ['scalar','block']:
 for seed in [71,72]:
  for region in ['prefix','task']:
   for mult in [1.,.1]:
    cells={}
    for basis,data in [('M11',old),('M18',new)]:
     cells[basis]={}
     for which,arm in [('initial','initial'),('joint',f'aggregate_gradient:{mult}:live'),('control',f'aggregate_KL:{mult}:live')]:
      base=[x for x in data['summaries'] if (x['kind'],x['seed'],x['region'],x['arm'])==(kind,seed,region,arm)];allrow=next(x for x in base if x['quadrant']=='all' and x['family'] is None);family=[x for x in base if x['quadrant']=='all' and x['family'] is not None];cell={k:allrow[k] for k in ['relative_margin_error','relative_logprob_response_error','native_margin_energy_sum','native_logprob_response_energy_sum','margin_error_sum','logprob_response_error_sum','nonzero']};valid=[x['relative_margin_error'] for x in family if x['relative_margin_error'] is not None];cell['equal_family_margin']=sum(valid)/len(valid);cell['quadrants']={q:{k:next(x for x in base if x['quadrant']==q and x['family'] is None)[k] for k in ['relative_margin_error','relative_logprob_response_error']} for q in ['evaluation:train','train:evaluation','evaluation:evaluation']};cells[basis][which]=cell
    rows.append(dict(kind=kind,seed=seed,region=region,multiplier=mult,cells=cells))
keys=['pair_id','endpoint','kind','seed','region','group'];prior={tuple(x[k] for k in keys):x for x in old['records'] if x['arm']=='initial'};current={tuple(x[k] for k in keys):x for x in new['records'] if x['arm']=='initial'};assert set(prior)==set(current) and len(prior)==704
for kind in ['scalar','block']:
 for seed in [71,72]:
  ks=[k for k,x in current.items() if (x['kind'],x['seed'])==(kind,seed)];energy.append(dict(kind=kind,seed=seed,directions=len(ks),old_noops=sum(prior[k]['noop'] for k in ks),new_noops=sum(current[k]['noop'] for k in ks),noop_status_changes=sum(prior[k]['noop']!=current[k]['noop'] for k in ks),old_delta_energy=sum(prior[k]['delta_energy'] for k in ks),new_delta_energy=sum(current[k]['delta_energy'] for k in ks),old_native_response_energy=sum(prior[k]['native_logprob_response_energy'] for k in ks),new_native_response_energy=sum(current[k]['native_logprob_response_energy'] for k in ks)))
counts=[]
for mult in [1.,.1]:
 rr=[x for x in rows if x['multiplier']==mult];metrics=['relative_margin_error','relative_logprob_response_error','equal_family_margin'];count=dict(multiplier=mult,by_basis={b:{m:sum(x['cells'][b]['joint'][m]<x['cells'][b]['control'][m] for x in rr) for m in metrics} for b in ['M11','M18']},winner_sign_changes={m:sum((x['cells']['M11']['joint'][m]<x['cells']['M11']['control'][m])!=(x['cells']['M18']['joint'][m]<x['cells']['M18']['control'][m]) for x in rr) for m in metrics},quadrants={q:{b:{m:sum(x['cells'][b]['joint']['quadrants'][q][m]<x['cells'][b]['control']['quadrants'][q][m] for x in rr) for m in metrics[:2]} for b in ['M11','M18']} for q in ['evaluation:train','train:evaluation','evaluation:evaluation']});counts.append(count)
out=dict(new_sha256=sha(np),old_joint_sha256=sha(jp),old_control_sha256=sha(cp),paired_cases=rows,native_effect_changes=energy,counts=counts,scope='Compare objectives within each intervention generator; raw cross-generator errors and effect denominators differ. Same prompts/candidates/group indices/regions; M18 generator changes encoding, selection and decoding, not a pure gauge rotation. Counts descriptive correlated development cases.');(P/'basis_transfer_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(counts=counts,native_effect_changes=energy),indent=2))
