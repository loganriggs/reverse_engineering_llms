"""M60 freeze matched objectives and training-only schedules/scales/calibration."""
from pathlib import Path
import json,hashlib,random,statistics
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 np=V/'training_causal_coverage_protocol.json';nativep=V/'training_causal_coverage.json';oldp=V/'training_coverage_featurizers.json';mp=V/'joint_shared_models.json';q=json.loads(np.read_text());native=json.loads(nativep.read_text());old=json.loads(oldp.read_text());models=[m for m in json.loads(mp.read_text())['models'] if m['arm']=='intact_KL'];assert len(models)==6;families=sorted({p['family'] for p in q['pairs']});schedules={}
 for seed in [71,72]:
  gen=random.Random(60000+seed);task=[]
  for cycle in range(6):
   order=families.copy();gen.shuffle(order)
   for fam in order:
    candidates=sorted(p['id'] for p in q['pairs'] if p['family']==fam);task.append(dict(family=fam,pair_id=gen.choice(candidates)))
  schedules[str(seed)]=dict(task_updates=task,natural_documents=[gen.randrange(32) for _ in range(66)])
 scales=[]
 for fam in families:
  for region in ['changed_input_positions','final_position']:
   rs=[r for r in native['records'] if r['family']==fam and r['layer']==1 and r['region']==region and r['status']=='evaluated'];scales.append(dict(family=fam,region=region,native_mean_energy=sum(r['response_energy'] for r in rs)/len(rs),endpoints=len(rs)))
 floor=.1*statistics.median(s['native_mean_energy'] for s in scales if s['native_mean_energy']>0)
 for s in scales:s['denominator']=max(s['native_mean_energy'],floor);s['floored']=s['native_mean_energy']<floor
 protocol=dict(stage='frozen before any fit',models=models,models_sha256=sha(mp),native_protocol_sha256=sha(np),native_results_sha256=sha(nativep),initial_results_sha256=sha(oldp),arms=['intact_KL','KL_raw_response','KL_balanced_response'],steps=132,natural_updates=66,task_updates=66,optimizer='Adam',learning_rate=.0001,schedules=schedules,task_K='Final-position native-vs-reconstructed KL. Explicitly differs from M36 all-position task KL; identical in all three arms. Selected to align task retention site and available exact M58 calibration; no comparisons treating this as the sole difference from historical M36.',natural_K='All-position mean native-vs-reconstructed KL on original natural documents0–31. One document per even update; task pair on odd update, average its two endpoints.',response='Mean over eligible changed-input and final-position native donor targets per endpoint. Identity retained as mandatory zero-loss/gradient control, excluded from mean since it is identically zero. Unaligned changed region omitted explicitly; final region retained.',scaling='Raw arm uses response squared error. Balanced arm divides each term by max(training family/region mean native response energy,0.1*median positive cell mean). Cell energies from entire M57 training bank; no holdout scores, losses or gradients. Floor coefficient fixed without search.',energy_floor=floor,scales=scales,calibration='For each raw/balanced arm, lambda=pooled initial scheduled task K / pooled initial scheduled task R across all6starts, using exact fixed schedules. Same lambda across architectures/seeds; calibration on training only. Control lambda0.',constraints='Identical starts/data/update schedules/lr/optimizer across arms. Dictionary unit-column normalization after updates; means/scales fixed. No weight/schedule/seed search, no per-family site selection, no filtering errors/weak targets.',evaluation='All final states, raw errors and native energies, pooled and equal-family source/final fidelity on M57 training and M59 familiar source-group holdouts; natural docs32–63 retention; support cost. Existing whole-task exclusions remain unused. No pristine transfer/manifold/discovery claim.',stop='One18fit experiment after calibration+loss interface/gradient audit pass;132updates each, no outcome-based checkpoint selection or retries changing hyperparameters. If preflight fails, diagnose correctness before fitting.')
 pp=V/'balanced_response_training_protocol.json';assert not pp.exists();pp.write_text(json.dumps(protocol,indent=2)+'\n')
 # Exact initial losses from already independently audited M58 raw final vectors.
 den={(s['family'],s['region']):s['denominator'] for s in scales};rows=[]
 for m in models:
  for step,t in enumerate(schedules[str(m['seed'])]['task_updates']):
   for ep in [0,1]:
    key=t['pair_id']+f'|{ep}';b=next(b for b in old['baselines'] if b['key']==key and all(b[k]==m[k] for k in ['kind','seed','arm']));rs=[r for r in old['rows'] if r['key']==key and all(r[k]==m[k] for k in ['kind','seed','arm']) and r['region']!='identity' and r['status']=='evaluated'];assert len(rs) in [1,2];rows.append(dict(kind=m['kind'],seed=m['seed'],task_update=step,key=key,family=t['family'],K=b['final_KL'],R_raw=sum(r['response_error'] for r in rs)/len(rs),R_balanced=sum(r['response_error']/den[t['family'],r['region']] for r in rs)/len(rs),regions=[r['region'] for r in rs]))
 K=statistics.mean(r['K'] for r in rows);raw=statistics.mean(r['R_raw'] for r in rows);balanced=statistics.mean(r['R_balanced'] for r in rows);cal=dict(status='calibrated_before_fit',protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),initial_results_sha256=sha(oldp),rows=rows,pooled_K=K,pooled_raw_R=raw,pooled_balanced_R=balanced,raw_weight=K/raw,balanced_weight=K/balanced,scope='No model fitting or new outcome selection. Exact scheduled initial losses reuse audited M58 training responses; future live loss checks must replay them.')
 cp=V/'balanced_response_calibration.json';assert not cp.exists();cp.write_text(json.dumps(cal,indent=2)+'\n');print('FROZEN',len(rows),'calibration rows; floor',floor,'weights',cal['raw_weight'],cal['balanced_weight'])
if __name__=='__main__':main()
