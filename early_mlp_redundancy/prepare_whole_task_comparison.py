"""M75 verify historical exposure and freeze all38 candidates for M76."""
import json
from pathlib import Path
from fit_pca_encoder import sha
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')

def main():
    q=json.loads((V/'whole_task_native_protocol.json').read_text())
    paths=[A/name for name in ['bilin18_expanded_task_bank.json','bilin18_broader_task_bank.json','bilin18_reserved_task_format_screen.json']]
    history={}
    for path in paths:
        d=json.loads(path.read_text())
        for r in d['records']:history[r['id']]=(path,r,'evaluated')
        for r in d.get('excluded',[])+d.get('invalid',[]):history.setdefault(r['id'],(path,r,'attempted_but_invalid'))
    exposure=[]
    for pair in q['pairs']:
        path,old,status=history[pair['historical_id']]
        if 'prompts' in old:assert pair['prompts']==old['prompts']
        exposure.append(dict(id=pair['id'],historical_id=pair['historical_id'],status=status,source=str(path)))
    out=dict(status='passed',source_sha256=sha(__file__),cohort_sha256=sha(V/'whole_task_native_protocol.json'),historical_results_hashes={str(p):sha(p) for p in paths},rows=exposure,scope='Every selected pair has prior evaluation or an attempted tokenization-invalid screen; historical task exposure is not inferred solely from family names.')
    (V/'whole_task_exposure_verification.json').write_text(json.dumps(out,indent=2)+'\n')
    old=json.loads((V/'least_squares_evaluation_models.json').read_text())['models']
    models=[e for e in old if e['variant'] in ['original','full_dictionary_LS']]+json.loads((V/'pca_evaluation_models.json').read_text())['models']
    assert len(models)==38 and len({(e['kind'],e['seed'],e['arm']) for e in models})==38
    for e in models:
        if 'path' in e:assert sha(e['path'])==e['sha256']
    native=json.loads((V/'whole_task_native.json').read_text());audit=json.loads((V/'whole_task_native_audit.json').read_text());assert audit['status']=='passed' and audit['result_sha256']==sha(V/'whole_task_native.json')
    manifest=dict(models=models,input_hashes={name:sha(V/name) for name in ['least_squares_evaluation_models.json','pca_evaluation_models.json','whole_task_native_protocol.json','whole_task_native.json','whole_task_native_audit.json','whole_task_native_comparison.json','whole_task_exposure_verification.json']})
    (V/'whole_task_evaluation_models.json').write_text(json.dumps(manifest,indent=2)+'\n')
    protocol=dict(stage='M76 fixed joint representation comparison on training-excluded tasks',models_sha256=sha(V/'whole_task_evaluation_models.json'),native_protocol_sha256=sha(V/'whole_task_native_protocol.json'),native_results_sha256=sha(V/'whole_task_native.json'),
      candidates='All18 original sparse encoders, all18 dense LS encoders from the same dictionaries, PCA32 and PCA128. No checkpoint/seed/rank choice, refitting or new labels.',
      evaluation='All56 pairs/112endpoints, seven families and both formats, including weak families. Native/native teacher plus reconstructed-background/native-change, native-background/reconstructed-change, and both-reconstructed corners. Source changed-token positions, final position, identity. Preserve all8 source assignment ineligibilities for each candidate; do not fabricate alignment.',
      scores='All raw and equal-family/pooled full-vocabulary response errors, answer-margin errors, local delta errors, intact output KL, binary and top1 accuracy. Report each format; separate full-cohort and predeclared contrast-gate-qualified summaries without dropping raw weak-family results. Zero-response baseline=1 is not an adequacy criterion. Reuse fixed M72/M73 natural retention measurements with hashes; do not rerun or tune them.',
      controls='Native baseline/teacher and delta replays; independent projection/decoder checks, all saved final-response vectors independently rescored; identity must be zero. For a newly encountered original encoder support use existing fixed forward implementation; do not learn gates on validation.',
      theory='If dense retained directions encode reusable computation, their gain over original sparse encoders should extend beyond familiar source-group tasks and both formats. If gains depend on narrow task/background geometry, retained local writes may transfer without causal response fidelity. Distinguish intact performance from intervention responses.',
      interpretation='All tasks were historically evaluated and have overlap with broader natural language phenomena. They are excluded explicit featurizer-training operations, not pristine novel tasks or zero semantic overlap. Pairwise contrast gate does not imply free-form competence. No manifold/discovery or literal neuron redundancy inference from transfer alone.',
      stop='One frozen comparison; no task-specific winner selection or additional rank/loss sweep. Decide whether any representation is ready for a targeted reusable-component/manifold intervention, or whether broad causal fidelity remains insufficient and requires a clearly specified new training design.')
    (V/'whole_task_comparison_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('EXPOSURE',len(exposure),sum(x['status']=='evaluated' for x in exposure),'evaluated; MODELS',len(models))
if __name__=='__main__':main()
