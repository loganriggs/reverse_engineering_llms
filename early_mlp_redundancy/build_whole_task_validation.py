"""M75 reuse frozen task cohorts and document training/exposure boundaries."""
import json
from pathlib import Path
from collections import Counter
import tiktoken
from fit_pca_encoder import sha
P=Path(__file__).resolve().parent;V=P/'validation';R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition'

def main():
    protocol=json.loads((V/'whole_task_validation_coverage_protocol.json').read_text())
    for name,digest in protocol['references'].items():assert sha(V/name)==digest
    paths=[A/'bilin18_expanded_task_bank_manifest.json',A/'bilin18_broader_task_bank_manifest.json',A/'reserved_task_format_screen/manifest.json']
    historical=[json.loads(p.read_text()) for p in paths]
    selected=[]
    for pi,rows in enumerate(historical):
        for row in rows:
            family=row.get('base_family',row['family'])
            if family not in protocol['exclusions']:continue
            if pi<2 and row['source']>=4:continue
            selected.append((paths[pi],row,family))
    assert Counter(f for _,_,f in selected)=={f:8 for f in protocol['exclusions']}
    train=json.loads((V/'training_causal_coverage_protocol.json').read_text())['pairs']
    old=json.loads((V/'training_source_context_audit.json').read_text())['rows']
    assert not set(protocol['exclusions'])&{p['family'] for p in train+old}
    tok=tiktoken.get_encoding('gpt2');seen=set();pairs=[];assignments=[]
    training_tokens={tuple(ids) for row in train+old for ids in row['token_ids']}
    for source,row,family in selected:
        texts=row['prompts'];ids=[tok.encode(t) for t in texts]
        assert tuple(texts) not in seen;seen.add(tuple(texts))
        assert all(tuple(x) not in training_tokens for x in ids)
        groups=[];invalid=[]
        for text in texts:
            gg=[]
            for answer in row['answers']:
                valid=[]
                for form in dict.fromkeys([answer,answer.capitalize(),answer.upper()]):
                    suffix=' '+form;full=tok.encode(text+suffix);prefix=tok.encode(text)
                    if len(full)==len(prefix)+1 and full[:-1]==prefix:valid.append(full[-1])
                gg.append(sorted(set(valid)))
            groups.append(gg)
            if any(not g for g in gg) or set(gg[0])&set(gg[1]):invalid.append('empty_or_overlapping_single_token_answer_groups')
        equal=len(ids[0])==len(ids[1]);changed=[i for i,(x,y) in enumerate(zip(*ids)) if x!=y] if equal else []
        pair=dict(id='m75-'+row['id'],historical_id=row['id'],family=family,format=row.get('format',row['family'].split('__')[-1]),source=str(source),prompts=texts,answers=row['answers'],token_ids=ids,answer_groups=groups,variants=[0,1],token_valid=not invalid,token_invalid_reasons=invalid,equal_length=equal,changed_positions=changed,source_region_eligible=equal and bool(changed) and not invalid,prior_evaluation_exposure=True)
        pairs.append(pair)
        for ep in (0,1):
            for region in ('changed_input_positions','final_position','identity'):
                eligible=not invalid and (region!='changed_input_positions' or pair['source_region_eligible'])
                positions=changed if region=='changed_input_positions' else [len(ids[ep])-1] if region=='final_position' else []
                donor=changed if region=='changed_input_positions' else [len(ids[1-ep])-1] if region=='final_position' else []
                assignments.append(dict(pair_id=pair['id'],endpoint=ep,layer=1,region=region,eligible=eligible,recipient_positions=positions,donor_positions=donor))
    definitions={
      'sequence_copy':'Repeated key/value sequence continuation; related computations can occur in natural training text.',
      'spatial_inverse':'Infer the inverse of left/right or above/below when the queried entities reverse.',
      'weekday_successor':'Next weekday in sequence or yesterday/today prose; historical H1041 formats were adaptive.',
      'role_binding':'Identify the agent under agent/patient swaps in active/passive clauses. Related to trained entity binding, but distinct operation/templates.',
      'tense':'Yesterday/Tomorrow cue with was/will answer contrast; differs from trained explicit past-tense lexical lookup. Open completion is non-unique.',
      'continent_recall':'Country-to-continent retrieval, related to trained country/capital/language facts. Historical America label is broad; not a unique-answer benchmark.',
      'object_function':'Object-to-use/purpose completion. Labels may be broad, nonunique or syntactically awkward; retain failures.'}
    audit=dict(status='passed_current_inventory_and_historical_exposure_audit',source_sha256=sha(__file__),protocol_sha256=sha(V/'whole_task_validation_coverage_protocol.json'),input_hashes={str(path):sha(path) for path in paths+[V/'training_causal_coverage_protocol.json',V/'training_source_context_audit.json',R/'scripts/run_bilin18_expanded_task_bank.py',R/'scripts/bilin18_broader_task_stimuli.py',R/'scripts/run_bilin18_reserved_task_format_screen.py']},current_training_pairs=len(train),historical_training_panel_pairs=len(old),training_families=sorted({p['family'] for p in train+old}),excluded_family_name_overlap=[],exact_training_token_sequence_overlap=0,definitions=definitions,selection='First four source slots of each A/B historical manifest; all four weekday pairs of each historical sequence/prose format. Selected without reading outcome scores.',scope='Verifies current explicit causal training inventory and historical80-pair panel, not absence of related computations from natural-text fitting or LM pretraining. All seven families have historical evaluation exposure; no pristine/new-task claim. No new native inference or training in this builder.')
    out=dict(stage='frozen_before_M75_native_scoring',source_sha256=sha(__file__),coverage_protocol_sha256=sha(V/'whole_task_validation_coverage_protocol.json'),input_hashes=audit['input_hashes'],pairs=pairs,assignments=assignments,layers=[1],training_exclusions=protocol['exclusions'],competence_rule='At least6/8 declared pairs have both endpoints positive intended-versus-alternative answer-group margin. Invalid pairs count as failed; all results retained.',answer_policy='Original/capitalized/uppercase single-token space-prefixed answers with exact prefix boundary. Alternative correct continuations are not exhaustive.',semantics='MLP1 native product-write donor changes at all changed token positions when equal-length, or explicitly aligned final positions. Preserve Down bias and recompute downstream. Address alignment does not establish semantic alignment; role swaps may change several tokens.',selection='No competence filtering, task removal or prompt rescue. Preserve invalid, zero-effect and weak cases; flag competence only for interpretation.')
    for name,data in [('whole_task_exclusion_audit.json',audit),('whole_task_native_protocol.json',out)]:
        path=V/name;assert not path.exists();path.write_text(json.dumps(data,indent=2)+'\n')
    print('FROZEN',len(pairs),'pairs',sum(p['token_valid'] for p in pairs),'token-valid',sum(p['source_region_eligible'] for p in pairs),'source-eligible',len(assignments),'assignments')
if __name__=='__main__':main()
