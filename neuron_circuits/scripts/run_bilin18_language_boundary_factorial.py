"""H1019 exhaustive five-port donor/recipient boundary factorial."""
from pathlib import Path
import sys,json,math,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_language_boundary_factorial';out=A/'language_boundary_factorial';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
meta=json.loads((A/'language_functional_subspace/metadata.json').read_text());ports=['source_residual','last_residual','other_residual','embedding_recurrence','first_layer_values'];config={'experiment':'H1019','ports':ports,'design':'All32donor/recipient port corners per32historical language examples. Source position is changed country token; last residual is final token; others exclude both. Entire x0/bus swapped separately. Native recipient/donor baselines and complete boundary replay audited.','analysis':'Exact finite Mobius decomposition and Shapley allocation; main-only, main+pair and every fixed mask reported. No fit or selected sparse graph. All historically opened evaluation remains diagnostic.','purpose':'Establish causal coverage of boundary before fitting third-task featurizers.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'corners.npz').exists();torch.set_num_threads(6)
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);F=torch.nn.functional
cache=[];native=[]
for e in meta:
 ids=torch.tensor([e['ids']],device='cuda');pos=torch.tensor([len(e['ids'])-1],device='cuda');ans=torch.tensor([[g[0] for g in e['groups']]],device='cuda')
 with torch.no_grad():initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float()
 raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);cache.append((raw,x0,bus));native.append(float(runner.component.native_margin(model,ids,pos,ans,torch,F)))
corners=np.zeros((32,32));zero_other=0;source_positions=[];donor_ids=[]
for i,e in enumerate(meta):
 donor=next(j for j,f in enumerate(meta) if f['pair']==e['pair'] and f['format']==e['format'] and f['order']!=e['order']);donor_ids.append(donor);changed=[k for k,(x,y) in enumerate(zip(e['ids'],meta[donor]['ids'])) if x!=y];assert len(changed)==1 and changed[0]!=len(e['ids'])-1;source=changed[0];source_positions.append(source);r,x,b=cache[i];rd,xd,bd=cache[donor];other=[k for k in range(r.shape[1]) if k not in [source,r.shape[1]-1]];zero_other=max(zero_other,float((rd[:,other]-r[:,other]).abs().max()))
 for mask in range(32):
  raw=r.clone()
  if mask&1:raw[:,source]=rd[:,source]
  if mask&2:raw[:,-1]=rd[:,-1]
  if mask&4:raw[:,other]=rd[:,other]
  xx=xd if mask&8 else x;bb=bd if mask&16 else b
  value=runner.component._suffix_margin(model,raw,xx,bb,torch.tensor([r.shape[1]-1],device='cuda'),torch.tensor([[g[0] for g in e['groups']]],device='cuda'),torch,F);corners[i,mask]=float(value)
 print('row',i,flush=True)
base=np.array(native);don=np.array([native[i] for i in donor_ids]);delta=corners-corners[:,:1];mobius=delta.copy()
for bit in range(5):
 for mask in range(32):
  if mask&(1<<bit):mobius[:,mask]-=mobius[:,mask^(1<<bit)]
shapley=np.zeros((32,5))
for mask in range(1,32):
 for bit in range(5):
  if mask&(1<<bit):shapley[:,bit]+=mobius[:,mask]/mask.bit_count()
np.savez(out/'corners.npz',corners=corners,mobius=mobius,shapley=shapley,native=base,donor=don,source_positions=source_positions,donor_ids=donor_ids)
torch.save([tuple(v.cpu() for v in c) for c in cache],out/'boundary_states.pt');(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
summary=[]
for cohort in sorted({e['cohort'] for e in meta}):
 ii=[i for i,e in enumerate(meta) if e['cohort']==cohort];target=don[ii]-base[ii];den=target@target
 metrics=[]
 for mask in range(32):
  y=delta[ii,mask];metrics.append({'mask':mask,'relative_l2':float(np.linalg.norm(y-target)/np.linalg.norm(target)),'aligned_recovery':float(y@target/den)})
 mains=mobius[ii][:,[1,2,4,8,16]].sum(1);pairs=mobius[ii][:,[m for m in range(1,32) if m.bit_count()<=2]].sum(1)
 summary.append({'cohort':cohort,'target_rms':float(np.sqrt(np.mean(target*target))),'mask_metrics':metrics,'main_only_relative_l2':float(np.linalg.norm(mains-target)/np.linalg.norm(target)),'main_pair_relative_l2':float(np.linalg.norm(pairs-target)/np.linalg.norm(target)),'shapley_aligned_share':{name:float(shapley[ii,k]@target/den) for k,name in enumerate(ports)},'order_rms':{str(order):float(np.sqrt(np.mean(mobius[ii][:,[m for m in range(1,32) if m.bit_count()==order]].sum(1)**2))) for order in range(1,6)}})
audit={'recipient_replay_max_abs':float(np.max(np.abs(corners[:,0]-base))),'donor_replay_max_abs':float(np.max(np.abs(corners[:,31]-don))),'mobius_closure_max_abs':float(np.max(np.abs(mobius.sum(1)-delta[:,31]))),'shapley_closure_max_abs':float(np.max(np.abs(shapley.sum(1)-delta[:,31]))),'other_residual_delta_max_abs':zero_other,'finite':bool(np.isfinite(corners).all()),'prefix_sequences':32,'native_sequences':32,'suffix_sequences':1024,'new_reserved_test_examples':0}
assert audit['recipient_replay_max_abs']<1e-4 and audit['donor_replay_max_abs']<1e-4 and audit['mobius_closure_max_abs']<1e-10 and audit['finite'];(A/f'{S}.json').write_text(json.dumps({'config':config,'summary':summary,'audits':audit},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audit},indent=2))
