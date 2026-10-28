"""M34 reconcile existing shared implementations, replay pilot and audit budgets."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
import torch
from train_write_featurizers import Featurizer
P=Path(__file__).resolve().parent
R=Path('/home/loganriggs/Coding/neuron_circuits')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1)
src=R/'scripts/feature_structures/models.py';test=R/'tests/test_feature_structures.py'
proc=subprocess.run([sys.executable,str(test)],capture_output=True,text=True)
assert proc.returncode==0,proc.stdout+proc.stderr
sys.path.insert(0,str(R/'scripts'))
from feature_structures.models import SharedGroups,union_support
m7p=P/'validation/write_featurizers.json';m7=json.loads(m7p.read_text());m4p=P/'validation/natural_token_context.json';m4=json.loads(m4p.read_text());cache=Path(m4['cache']);assert sha(cache)==m4['cache_sha256'];data=torch.load(cache,weights_only=True);Y=data['Y'][1];records=[]
for entry in m7['models']:
 path=Path(entry['path']);assert sha(path)==entry['sha256'];state=torch.load(path,weights_only=True);kind,seed=entry['kind'],entry['seed'];m=Featurizer(kind,seed);m.load_state_dict(state['state']);m.eval()
 # Full saved evaluation cohort, not outcome-selected positions.
 x=(Y[4096:]-state['mean'])/state['scale']
 with torch.no_grad():pred,support,code=m(x)
 a=(x.double()@m.encoder.detach().double().T+m.encoder_bias.detach().double()).numpy();M=m.membership.numpy();q=np.zeros_like(a)
 if kind=='scalar':
  selected=np.argsort(-np.abs(a),axis=1)[:,:32]
  np.put_along_axis(q,selected,1,axis=1)
 else:
  assert np.all(M.sum(1)==(4 if kind=='block' else 8));assert np.all(M.sum(0)==(1 if kind=='block' else 2))
  scores=a*a@M.T;selected=np.argsort(-scores,axis=1)[:,:8 if kind=='block' else 4]
  for j,groups in enumerate(selected):
   # Explicit Python sets avoid duplicating the production union implementation.
   atoms=set()
   for g in groups:atoms.update(np.flatnonzero(M[g]).tolist())
   q[j,list(atoms)]=1
 assert np.array_equal(q,support.numpy()), 'Support mismatch, including potential near-ties'
 D=m.dictionary.detach().double();D=D/D.norm(dim=0)
 decoded=(a*q)@D.numpy().T+m.bias.detach().double().numpy()
 error=float(np.max(np.abs(decoded-pred.detach().numpy())))
 assert error<1e-4,error
 recon=pred*state['scale']+state['mean'];saved_error=float((recon-state['reconstructions']).abs().max());normalized_saved_error=float(((recon-state['reconstructions'])/state['scale']).abs().max());assert normalized_saved_error<1e-4,(saved_error,normalized_saved_error)
 active=q.sum(1)
 records.append(dict(kind=kind,seed=seed,model_sha256=sha(path),positions=len(x),parameters=sum(p.numel() for p in m.parameters()),membership_slots=int(M.sum()),covered_atoms=int((M.sum(0)>0).sum()),active_mean=float(active.mean()),active_min=int(active.min()),active_max=int(active.max()),numpy_forward_max_error=error,saved_reconstruction_max_error=saved_error,normalized_saved_reconstruction_max_error=normalized_saved_error))
# Explicit intervention semantic distinction on a controlled overlap.
M=torch.tensor([[1.,1.,0.],[0.,1.,1.]])
gates=torch.ones(1,2);support=union_support(gates,M)
removed_group=union_support(torch.tensor([[0.,1.]]),M)
removed_atoms=support*(1-M[0])
assert torch.equal(removed_group,torch.tensor([[0.,1.,1.]]))
assert torch.equal(removed_atoms,torch.tensor([[0.,0.,1.]]))
# Count the separately learned gates/membership architecture at pilot dimensions.
learned=SharedGroups(1152,128,32,4,8)
learned_counts=dict(total_parameters=sum(p.numel() for p in learned.parameters()),coordinate_encoder=sum(p.numel() for p in learned.coordinate_encoder.parameters()),group_encoder=sum(p.numel() for p in learned.group_encoder.parameters()),membership_parameters=learned.membership_logits.numel(),membership_slots=32*8,maximum_union=4*8)
# Validate current task schedule exclusions and train/evaluation separation.
cp=P/'validation/context_training_protocol.json';protocol=json.loads(cp.read_text());ep=P/'validation/context_expansion_protocol.json';expanded={x['id']:x for x in json.loads(ep.read_text())['rows']};sp=P/'validation/diverse_supervision_split.json';split=json.loads(sp.read_text());assert protocol['split_sha256']==sha(sp) and protocol['expansion_sha256']==sha(ep)
train=set(split['train_pair_ids']);evaluation=set(split['eval_pair_ids']) if 'eval_pair_ids' in split else set(split['evaluation_pair_ids']);assert not train&evaluation
excluded={'sequence_copy','spatial_inverse','weekday_successor','role_binding','tense'};used=[]
for seed,schedule in protocol['schedules'].items():
 for u in schedule['task_updates']:
  e=expanded[u['expanded_pair_id']];assert u['source_pair_id'] in train and e['training_eligible'];assert e['lexical_split']=='train' and e['template_split']=='train';assert e['family'] not in excluded;used.append(e['family'])
 assert all(0<=d<32 for d in schedule['natural_documents'])
out=dict(status='passed',source_sha256=sha(Path(__file__)),implementation_sha256=sha(src),structural_tests_sha256=sha(test),structural_test_output=(proc.stdout+proc.stderr).strip(),pilot_results_sha256=sha(m7p),native_pilot_source_sha256=sha(P/'train_write_featurizers.py'),pilot_replays=records,learned_shared_parameters=learned_counts,intervention_semantics=dict(remove_group_gate_support=removed_group.tolist(),remove_all_member_atoms_support=removed_atoms.tolist(),meaning='Removing a group gate preserves atoms still selected by another group. Atom deletion removes them from all consumers; these are different interventions.'),training_schedule_sha256=sha(cp),split_sha256=sha(sp),training_families=sorted(set(used)),schedule_task_updates_checked=len(used),whole_task_exclusions=sorted(excluded),scope='Implementation/coverage/budget readiness, not a new training result or semantic recovery. Structural synthetic oracle tests do not establish unsupervised recovery. Shared fixed was omitted after M7; learned SharedGroups is a distinct larger architecture using straight-through surrogate gradients.')
(P/'validation/shared_comparison_readiness.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
