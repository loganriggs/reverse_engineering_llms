"""M18 freeze paired context training and native response validation; no model outcomes."""
from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(tag,x):return hashlib.sha256((tag+'|'+x).encode()).hexdigest()
def main():
 ep=P/'context_expansion_protocol.json';cp=P/'context_expansion_competence.json';sp=P/'diverse_supervision_split.json';mp=P/'diverse_supervision_results.json';e=json.loads(ep.read_text());c=json.loads(cp.read_text());split=json.loads(sp.read_text());m=json.loads(mp.read_text());assert c['protocol_sha256']==sha(ep);audit=json.loads((P/'context_expansion_audit.json').read_text());assert audit['competence_sha256']==sha(cp);passed={(x['template'],x['lexical_split'],x['family']) for x in c['cohorts'] if x['pass_gate']};eligible=[x for x in e['rows'] if x['training_eligible'] and (x['template'],x['lexical_split'],x['family']) in passed];families=sorted({x['family'] for x in eligible});assert len(families)==10;schedules={}
 for seed in [71,72]:
  rng=np.random.default_rng(2026102800+seed);order=[f for _ in range(6) for f in rng.permutation(families)];counts={f:0 for f in families};task=[];natural=rng.integers(0,32,size=120).tolist()
  for j,f in enumerate(order):
   ids=sorted({x['base_id'] for x in eligible if x['family']==f},key=lambda x:rank(f'source:{seed}:{f}',x));n=counts[f];pid=ids[n%len(ids)];options=sorted([x for x in eligible if x['base_id']==pid],key=lambda x:x['template']);selected=options[n%len(options)];task.append(dict(task_update=j,family=f,source_pair_id=pid,expanded_pair_id=selected['id']));counts[f]+=1
  assert set(counts.values())=={6};schedules[str(seed)]=dict(natural_documents=natural,task_updates=task)
 # Select one whole pair per family/nontraining quadrant using hashes, never featurizer outcomes.
 causal=[]
 for ls,ts in [('evaluation','train'),('train','evaluation'),('evaluation','evaluation')]:
  candidates=[x for x in e['rows'] if x['lexical_split']==ls and x['template_split']==ts and (x['template'],ls,x['family']) in passed]
  for family in sorted({x['family'] for x in candidates}):
   row=min([x for x in candidates if x['family']==family],key=lambda x:rank('causal-pair-v1',x['id']));causal.append(dict(pair_id=row['id'],family=family,lexical_split=ls,template_split=ts))
 initial=[x for x in m['models'] if x['arm']=='mixed_natural_task_KL'];assert len(initial)==4
 for x in initial:assert sha(Path(x['path']))==x['sha256']
 interventions=[]
 for pair in causal:
  for epidx in [0,1]:
   for seed in [71,72]:
    groups=sorted(range(32),key=lambda g:rank(f'causal-group-v1:{pair["pair_id"]}:{epidx}:{seed}',str(g)))[:2]
    for region in ['prefix','task']:interventions.append(dict(pair_id=pair['pair_id'],endpoint=epidx,seed=seed,region=region,groups=groups))
 protocol=dict(status='frozen_before_training',source_sha256=sha(Path(__file__)),expansion_sha256=sha(ep),competence_sha256=sha(cp),split_sha256=sha(sp),M11_results_sha256=sha(mp),initial_models=initial,arms=['original_context','expanded_context'],steps=120,learning_rate=.0003,optimizer='Fresh Adam per fit; normalize decoder columns after each step; frozen mean/scale; native model frozen.',schedules=schedules,training='Even0-basedsteps: paired same naturaltrainingdoc0–31. Oddsteps: paired same sourcepair, original vs specified expandedcontext. Equalendpoint-weight allposition/fullvocabulary native-output KL. Sixupdates/family=60taskupdates and60naturalupdates perfit. 4starts x2arms=8fits. Sameinit/sources/updates, notmatchedtokens/FLOPs. Gateonlyexpandedtrain/traincohorts. No causal-response loss in this comparison; response fidelity is validation.',evaluation='Fixed final checkpoint plus each initial checkpoint. All480 expandedpairs, all80originalpairs, naturaldocs32–63. Report lexical/template quadrants and families separately, allrows and native-passingcohorts separately; train/train descriptive only. Naturalforgetting vs initial and original-context control. No evaluation-based checkpoint/hyperparameter selection.',causal_pairs=causal,causal_interventions=interventions,causal_design='For each initial checkpoint and selected endpoint, freeze native-write-derived initial featurizer codes/dictionary, group4 contiguousatoms, decoded contribution delta masked to region. Same seed-selectedgroups for scalar/block; no shuffled matching search. Compute native original-write minus delta versus each initial/final reconstructed-write minus identical delta. Preserve bias and recompute downstream. Compare each perturbed output to its own intactbaseline. Do not recompute delta from trainedfeaturizer or patch teacher activations downstream. Retain exactzero directions, report separately. Onepair/family/nontrainingquadrant, hashselection after nativegate but before featurizer outcomes. Everyplanneddirectionretained; no amplitudefitting.',causal_metrics='Primary: squared discrepancy between reconstructed and native final-answer-margin effects, normalized by summed native squared effects, plus unnormalized RMSE. Aggregate zero denominator -> null, no denominator clipping or per-example ratios. Secondary: final-logprob response discrepancy weighted by intact native probabilities, normalized by summed native response energy, with unnormalized numerator. Also intactKL and perturbedKL against corresponding native outputs. Region/quadrant/family separate, weightedaggregate and equal-family metrics labeled. Fixed-direction fidelity does not prove final feature identity, semantics, or manifold structure.',stopping_rule='Exactly8fits/fixed120updates. No gain/group/width/template search. Complete native fixed-perturbation validation before interpreting improved outputKL as improved causal structure. Preserve original task exclusions. New task discovery remains unachieved.')
 op=P/'context_training_protocol.json'
 if op.exists():assert json.loads(op.read_text())==protocol
 else:op.write_text(json.dumps(protocol,indent=2)+'\n')
 print('FROZEN',len(initial)*2,'fits;',len(causal),'causalpairs;',len(interventions)*2*2,'nativeperturbations across4initialmodels')
if __name__=='__main__':main()
