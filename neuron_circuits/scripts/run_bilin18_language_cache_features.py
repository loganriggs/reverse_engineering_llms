"""H1020 cached-value causal-coordinate and architectural head-slot census."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_language_cache_features';out=A/'language_cache_features';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
meta=json.loads((A/'language_boundary_factorial/metadata.json').read_text());cache=torch.load(A/'language_boundary_factorial/boundary_states.pt',weights_only=True);old=np.load(A/'language_boundary_factorial/corners.npz');train=[i for i,e in enumerate(meta) if e['cohort']=='discovery'];assert len(train)==8
config={'experiment':'H1020','port':'First-layer cached attention values at source-country position;9headslots x128coordinates, repeatedly mixed into suffix attentions','training':'Eight discovery gradients; four independent opposite-country pairs; no evaluation gradients','bases':'gradient4/8; discovery delta PCA4 after numerical rank check; weight output singular vectors4/8/32 from c_v layer0; seeded random4/8; nine native cached head-slot groups128coords each','target':'Native cache-only port swap, sourceposition; residual and x0remain recipient. Headslot is reused across suffix layers, not a particular layer/head attribution.','scope':'Historical evaluation; task/output vocabulary transfer; no semantic feature adoption, manifold claim or head-versus-rank dimension-equivalence claim.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'responses.npz').exists();torch.set_num_threads(6)
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
F=torch.nn.functional

def suffix(raw,x0,bus,ans):
 x=raw
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,bus=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);return y[0,ans[0]]-y[0,ans[1]]
deltas=[];grads=[];finite_errors=[];zero_other=0
for i,e in enumerate(meta):
 raw,x0,bus=[v.cuda() for v in cache[i]];donor=int(old['donor_ids'][i]);source=int(old['source_positions'][i]);donbus=cache[donor][2].cuda();delta=(donbus-bus)[0,source].detach().cpu().double().numpy().reshape(-1);deltas.append(delta);other=[k for k in range(bus.shape[1]) if k!=source];zero_other=max(zero_other,float((donbus[:,other]-bus[:,other]).abs().max()));ans=[g[0] for g in e['groups']]
 if i in train:
  bb=bus.detach().requires_grad_(True);v=suffix(raw,x0,bb,ans);g=torch.autograd.grad(v,bb)[0][0,source].cpu().double().numpy().reshape(-1);grads.append(g)
  with torch.no_grad():vp=suffix(raw,x0,bus+.001*(donbus-bus),ans);vm=suffix(raw,x0,bus-.001*(donbus-bus),ans)
  finite_errors.append([float((vp-vm)/.002),float(g@delta)])
deltas=np.stack(deltas);grads=np.stack(grads);_,sg,vg=np.linalg.svd(grads,full_matrices=False);_,sd,vd=np.linalg.svd(deltas[train],full_matrices=False);rank=int(np.sum(sd>sd[0]*1e-8));assert rank==4
weight=model.transformer.h[0].attn.c_v.weight.detach().cpu().double().numpy();uw,_,_=np.linalg.svd(weight,full_matrices=False);rng=np.random.default_rng(1020);random,_=np.linalg.qr(rng.normal(size=(1152,8)))
bases={'gradient4':vg[:4].T,'gradient8':vg[:8].T,'source4':vd[:4].T,'weight4':uw[:,:4],'weight8':uw[:,:8],'weight32':uw[:,:32],'random4':random[:,:4],'random8':random};np.savez(out/'bases.npz',**bases,gradients=grads,discovery_delta_singular_values=sd);frozen=hashlib.sha256((out/'bases.npz').read_bytes()).hexdigest();records=[];maxbase=maxfull=0
for i,e in enumerate(meta):
 raw,x0,bus=[v.cuda() for v in cache[i]];source=int(old['source_positions'][i]);ans=[gg[0] for gg in e['groups']];d=deltas[i];edits={'full':d}|{name:d@b@b.T for name,b in bases.items()}
 for head in range(9):h=np.zeros(1152);h[head*128:(head+1)*128]=d[head*128:(head+1)*128];edits[f'headslot{head}']=h
 with torch.no_grad():base=float(suffix(raw,x0,bus,ans))
 maxbase=max(maxbase,abs(base-old['corners'][i,0]))
 for name,edit in edits.items():
  bb=bus.clone();bb[0,source]+=torch.tensor(edit,device='cuda',dtype=bb.dtype).reshape_as(bb[0,source])
  with torch.no_grad():value=float(suffix(raw,x0,bb,ans))
  if name=='full':maxfull=max(maxfull,abs(value-old['corners'][i,16]))
  records.append(dict(row=i,cohort=e['cohort'],arm=name,change=value-base,edit_norm=float(np.linalg.norm(edit)),full_edit_norm=float(np.linalg.norm(d))))
summary=[];heads=[]
for cohort in sorted({e['cohort'] for e in meta}):
 rr=[r for r in records if r['cohort']==cohort];full=np.array([r['change'] for r in rr if r['arm']=='full']);den=full@full
 for arm in edits:
  aa=[r for r in rr if r['arm']==arm];y=np.array([r['change'] for r in aa]);summary.append(dict(cohort=cohort,arm=arm,relative_l2=float(np.linalg.norm(y-full)/np.linalg.norm(full)),aligned_recovery=float(y@full/den),mean_energy_fraction=float(np.mean([(r['edit_norm']/r['full_edit_norm'])**2 for r in aa]))))
 yy=np.array([[r['change'] for r in rr if r['arm']==f'headslot{h}'] for h in range(9)]);heads.append(dict(cohort=cohort,headslot_main_sum_relative_l2=float(np.linalg.norm(yy.sum(0)-full)/np.linalg.norm(full)),aligned_shares=(yy@full/den).tolist()))
fd=np.array(finite_errors);audit={'cache_only_full_replay_max_abs':maxfull,'base_replay_max_abs':maxbase,'finite_difference_relative_error':float(np.linalg.norm(fd[:,0]-fd[:,1])/np.linalg.norm(fd[:,1])),'unchanged_non_source_cache_max_abs':zero_other,'discovery_delta_rank':rank,'basis_hash_unchanged':frozen==hashlib.sha256((out/'bases.npz').read_bytes()).hexdigest(),'evaluation_gradients':0,'suffix_sequences':632,'backward_sequences':8,'finite':all(np.isfinite(r['change']) for r in records),'new_reserved_test_examples':0}
assert maxbase<1e-4 and maxfull<1e-4 and audit['finite_difference_relative_error']<.02 and audit['finite'];(A/f'{S}.json').write_text(json.dumps({'config':config,'summary':summary,'headslot_summary':heads,'records':records,'audits':audit},indent=2)+'\n');np.savez(out/'responses.npz',deltas=deltas,finite_difference=fd,changes=np.array([r['change'] for r in records]));print(json.dumps({'summary':summary,'headslot_summary':heads,'audits':audit},indent=2))
