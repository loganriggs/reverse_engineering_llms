"""H1030 discovery-only three-port causal factorial for a matched representation contract."""
from pathlib import Path
import sys,json,hashlib,math,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_multitask_boundary_contract';out=A/'multitask_boundary_contract';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
paths=[A/'bilin18_context_training_bank.json',A/'bilin18_context_training_bank_training_eligibility.json',A/'bilin18_context_training_bank_captures.pt'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};bank=json.loads(paths[0].read_text());eligible=set(json.loads(paths[1].read_text())['training_indices']);data=torch.load(paths[2],weights_only=False,map_location='cpu');families=['agreement','capital_recall','entity_binding','sentiment','negation','pronoun_gender'];pairs=[r for r in bank['records'] if r['family'] in families and all(i in eligible for i in r['indices'])];excluded=[r['id'] for r in pairs if not r['equal_length']];pairs=[r for r in pairs if r['equal_length']]
indices=sorted({i for r in pairs for i in r['indices']});assert all(r['split']=='discovery' and not r['whole_task_holdout'] for r in pairs)
for r in pairs:assert data['examples'][r['indices'][0]]['groups']==data['examples'][r['indices'][1]]['groups']
config={'experiment':'H1030','families':families,'pairs':len(pairs),'states':len(indices),'unequal_length_excluded':excluded,'ports':['pre-attention11 residual after embedding mixture','embedding recurrence x0 across all positions','layer0 cached value bus across all positions and heads'],'design':'All eight donor/recipient boundary corners, both directions, eligible discovery rows only. Full boundary transplant exact replay control. No per-output filtering.','contract':'Future candidates receive identical port values and token positions, normalized from training only. Last-token MLP12-only models have unequal information access and remain historical references, not matched baselines. Native suffix remains charged for behavior evaluation.','metrics':'Single-port aligned natural margin change and Shapley allocation; interactions reported; no claim of feature discovery.','limits':'Port interventions are off-manifold hybrids; attribution depends on boundary and grouped answer contrast. Full-sequence states, not individual semantic features.','source_hashes':hashes}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(pairs,indent=2)+'\n')
torch.set_num_threads(6);F=torch.nn.functional;model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
@torch.no_grad()
def capture(i):
 e=data['examples'][i];initial=F.rms_norm(model.transformer.wte(torch.tensor([e['ids']],device='cuda')),(1152,)).float();raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);return raw,x0,bus
@torch.no_grad()
def suffix(state,groups):
 x,x0,bus=state
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);return float(torch.logsumexp(y[0,groups[0]].double(),0)-torch.logsumexp(y[0,groups[1]].double(),0))
states={i:capture(i) for i in indices};records=[];maxnative=maxsum=0
for p in pairs:
 for i,d in [p['indices'],p['indices'][::-1]]:
  groups=data['examples'][i]['groups'];corners=[suffix(tuple(states[d][k] if mask&(1<<k) else states[i][k] for k in range(3)),groups) for mask in range(8)];base=float(torch.logsumexp(data['native'][i][groups[0]],0)-torch.logsumexp(data['native'][i][groups[1]],0));donor=float(torch.logsumexp(data['native'][d][groups[0]],0)-torch.logsumexp(data['native'][d][groups[1]],0));maxnative=max(maxnative,abs(corners[0]-base),abs(corners[7]-donor));phi=[]
  for bit in range(3):
   val=0
   for mask in range(8):
    if mask&(1<<bit):continue
    n=mask.bit_count();val+=math.factorial(n)*math.factorial(2-n)/6*(corners[mask|(1<<bit)]-corners[mask])
   phi.append(val)
  delta=corners[7]-corners[0];maxsum=max(maxsum,abs(sum(phi)-delta));records.append(dict(pair=p['id'],family=p['family'],recipient=i,donor=d,corners=corners,shapley=phi,natural_delta=delta))
summary=[]
for f in families:
 rr=[r for r in records if r['family']==f]
 if not rr:
  summary.append(dict(family=f,directions=0,status='not_measured_unequal_token_lengths'));continue
 c=np.array([r['corners'] for r in rr]);delta=c[:,7]-c[:,0];single=c[:,[1,2,4]]-c[:,0,None];phi=np.array([r['shapley'] for r in rr]);den=delta@delta;res=delta-single.sum(1);summary.append(dict(family=f,directions=len(rr),natural_effect_rms=float(np.sqrt(np.mean(delta**2))),single_port_aligned_share=(single.T@delta/den).tolist(),shapley_aligned_share=(phi.T@delta/den).tolist(),additive_relative_error=float(np.linalg.norm(res)/np.linalg.norm(delta))))
audit={'native_full_boundary_replay_max_abs':maxnative,'shapley_sum_max_abs':maxsum,'prefix_sequences':len(indices),'suffix_sequences':len(records)*8,'fit_rows':0,'whole_task_holdout_access':0,'finite':bool(np.isfinite([r['corners'] for r in records]).all()),'execution_attempts':3,'total_prefix_sequences_including_failed_summary_attempt':3*len(indices),'total_suffix_sequences_including_failed_summary_attempt':3*len(records)*8,'source_hashes_unchanged':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in paths)};torch.save(dict(states={i:tuple(v.cpu() for v in st) for i,st in states.items()},examples={i:data['examples'][i] for i in indices}),out/'states.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,audits=audit),indent=2))

# Historical outputs differ numerically; verify against independent current full forwards.
fresh={}
with torch.no_grad():
 for i in indices:
  e=data['examples'][i];x0=F.rms_norm(model.transformer.wte(torch.tensor([e['ids']],device='cuda')),(1152,)).float();x=x0;bus=None
  for b in model.transformer.h:x,bus=b(x,bus,x0)
  y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);g=e['groups'];fresh[i]=float(torch.logsumexp(y[0,g[0]].double(),0)-torch.logsumexp(y[0,g[1]].double(),0))
error=max(max(abs(r['corners'][0]-fresh[r['recipient']]),abs(r['corners'][7]-fresh[r['donor']])) for r in records)
audit['historical_margin_replay_max_abs']=audit.pop('native_full_boundary_replay_max_abs');audit['fresh_direct_full_model_replay_max_abs']=error;audit['additional_direct_full_sequences']=len(indices);audit['fresh_reference_passed']=error<1e-4
assert error<1e-4 and maxsum<1e-10 and audit['finite'] and audit['source_hashes_unchanged']
(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,audits=audit),indent=2)+'\n')
