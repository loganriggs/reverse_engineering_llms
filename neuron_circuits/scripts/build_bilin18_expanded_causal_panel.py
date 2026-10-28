from pathlib import Path
import sys,json,hashlib
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import circuit_fast_screen_candidate_subject_number_rank1_fresh_confirmation as fresh
import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as old
enc=fresh.old_task14.ENCODING
vocab,_,oldtokens=fresh._prior_material();vocab.update(w for p in fresh.NOUN_PAIRS for w in p);vocab.update(w for p in old.PAIRS for w in p[:2]);vocab.update(w for p in fresh.old_task14.NOUN_PAIRS for w in p)
words='accountant actor administrator agent analyst architect artist auditor author banker barber builder captain carpenter chef clerk coach collector commander consultant contractor cook counselor dancer dealer dentist designer detective developer director doctor driver editor educator engineer examiner farmer firefighter fisherman founder gardener governor guard guide historian hunter inspector instructor investigator janitor journalist judge lawyer lecturer librarian manager mechanic merchant miner musician nurse officer operator painter passenger patient photographer physician pilot plumber poet politician porter producer professor programmer publisher reporter researcher sailor scholar scientist secretary senator singer soldier speaker specialist student surgeon teacher technician therapist tourist trainer translator traveler vendor visitor waiter warrior worker writer'.split()
pairs=[(w,w+'s') for w in words if w not in vocab and w+'s' not in vocab and len(enc.encode(' '+w))==len(enc.encode(' '+w+'s'))==1]
assert len(pairs)>=32,len(pairs)
trainpairs=pairs[:24];testpairs=pairs[24:32];rows=[]
for split,pool,templates in [('expanded_discovery',trainpairs,[('near','Near'),('behind','Behind')]),('new_lexical_test',testpairs,[('near','Near'),('behind','Behind')]),('new_joint_test',testpairs,[('under','Under'),('above','Above')])]:
 for i,pair in enumerate(pool):
  for n in [0,1]:
   for template,word in templates:
    for offset in ([3,7] if split=='expanded_discovery' else [3]):
     attractor=pool[(i+offset)%len(pool)][1-n];subject=pair[n];text=f'{word} the {attractor}, the {subject}';ids=enc.encode(text)
     assert len(ids)==6 and ids[-1]==enc.encode(' '+subject)[0] and tuple(ids) not in oldtokens
     answer=enc.encode(' is' if n==0 else ' are')[0];foil=enc.encode(' are' if n==0 else ' is')[0]
     assert enc.encode(text+(' is' if n==0 else ' are'))==ids+[answer]
     rows.append(dict(split=split,subject=subject,attractor=attractor,template=template,text=text,token_ids=ids,position=5,answer=answer,foil=foil))
assert len({tuple(r['token_ids']) for r in rows})==len(rows)
assert not set(w for p in trainpairs for w in p)&set(w for p in testpairs for w in p)
report={'training_pairs':trainpairs,'test_pairs':testpairs,'rows':rows,'excluded_prior_vocabulary_count':len(vocab),'scope':'Noun disjoint from inspected prior subject-number authority vocabulary, not global novelty across every research artifact. Reserved BSF data not accessed. Fixed candidate list order filtered only on lexical overlap and atomic tokenization, before model evaluation.'}
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');p=A/'bilin18_expanded_causal_panel.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(len(rows),trainpairs,testpairs,hashlib.sha256(p.read_bytes()).hexdigest())
