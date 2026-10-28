"""H1040 fixed eight-corner causal boundary contract for added training cohorts."""
from pathlib import Path
import sys,json,hashlib,ast,math,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_expanded_training_port_contract';out=A/'expanded_training_port_contract';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
root=A/'diverse_task_expansion_screen';files=[root/'states.pt',root/'training_eligibility.json',A/'bilin18_diverse_task_expansion_screen.json'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};data=torch.load(files[0],weights_only=False);eligible=json.loads(files[1].read_text());screen=json.loads(files[2].read_text());pairs=[p for p in screen['records'] if p['id'] in eligible['pair_ids']];assert len(pairs)==20 and all(p['split']=='train' and p['equal_length'] for p in pairs)
for p in pairs:
 i,d=p['indices'];assert data['examples'][i]['groups']==data['examples'][d]['groups']
config=dict(experiment='H1040',source_hashes=hashes,ports=['residual pre-attention11','embedding recurrence input','cached first-layer values'],design='All eight donor/recipient corners in both directions; all20 fixed eligible training pairs, including native errors. No outcome-based filtering.',target='Native donor answer-group margin and full output distribution; binary answer correctness is distinct from native causal fidelity.',scope='Boundary port allocation, not identified semantic features. Off-manifold hybrid interventions. No reserved task outcomes used.',fit_rows=0)
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(pairs,indent=2)+'\n')
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
tree=ast.parse((R/'scripts/run_bilin18_projection_stream_factorial.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['suffix','KL','margin']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<audited-functions>','exec'),globals())
ix=eligible['indices'];states={i:tuple(v.cuda() for v in data['states'][i]) for i in ix};native={i:data['native'][i].cuda() for i in ix};records=[];samples=[];replay=0;shapley_error=0
for p in pairs:
 for i,d in [p['indices'],p['indices'][::-1]]:
  g=data['examples'][i]['groups'];ys=[suffix(tuple(states[d][k] if mask&(1<<k) else states[i][k] for k in range(3))) for mask in range(8)];corners=[margin(y,g) for y in ys];replay=max(replay,float((ys[0]-native[i]).abs().max()),float((ys[7]-native[d]).abs().max()));phi=[]
  for bit in range(3):
   value=0
   for mask in range(8):
    if mask&(1<<bit):continue
    n=mask.bit_count();value+=math.factorial(n)*math.factorial(2-n)/6*(corners[mask|(1<<bit)]-corners[mask])
   phi.append(value)
  delta=corners[7]-corners[0];shapley_error=max(shapley_error,abs(sum(phi)-delta));den=KL(native[d],native[i]);records.append(dict(pair=p['id'],family=p['family'],recipient=i,donor=d,native_pair_correct=p['both_correct'],corners=corners,shapley=phi,natural_delta=delta,donor_KL=[KL(native[d],y) for y in ys],donor_KL_denominator=den))
  if i==pairs[0]['indices'][0]:samples=[dict(mask=mask,logits=y.cpu(),donor=native[d].cpu(),groups=g) for mask,y in enumerate(ys)]
 print('EVALUATED',p['id'],flush=True)
summary=[]
for f in eligible['families']:
 rr=[r for r in records if r['family']==f];c=np.array([r['corners'] for r in rr]);delta=c[:,7]-c[:,0];ph=np.array([r['shapley'] for r in rr]);den=delta@delta;single=c[:,[1,2,4]]-c[:,0,None];summary.append(dict(family=f,directions=len(rr),natural_effect_rms=float(np.sqrt(np.mean(delta**2))),shapley_aligned_share=(ph.T@delta/den).tolist(),single_port_aligned_share=(single.T@delta/den).tolist(),additive_relative_error=float(np.linalg.norm(delta-single.sum(1))/np.linalg.norm(delta)),relative_donor_KL_by_mask=(np.array([r['donor_KL'] for r in rr]).sum(0)/sum(r['donor_KL_denominator'] for r in rr)).tolist()))
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),native_full_boundary_logit_replay_max=replay,shapley_efficiency_max=shapley_error,suffix_sequences=320,fit_rows=0,reserved_pairs_selected=0,finite=all(np.isfinite(r['corners']).all() and np.isfinite(r['donor_KL']).all() for r in records))
assert replay<1e-5 and shapley_error<1e-10 and audit['finite'] and audit['source_hashes_unchanged']
torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,audits=audit),indent=2))
