"""H1023 export and validate source-embedding-to-write program without QK inference."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch,importlib.util
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_head13_lexical_export';out=A/'head13_lexical_program'
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
spec=importlib.util.spec_from_file_location('lexical_program',out/'execute.py');program=importlib.util.module_from_spec(spec);spec.loader.exec_module(program)
config={'experiment':'H1023','inputs':'Raw1152-dimensional source-token recipient/donor embeddings, float32. Embedding table/token lookup remains external and is not a free compressed component.','weights':'Layer0head0value128x1152, layer13head0output1152x128, firstblock residual/embedding scalars, layer13cachemix, frozen2routingmeans','operation':'Two native RMSnormalizations per token, firstvalueprojection, difference, outputprojection and fixed scalar routing. No QK/attention or model prefix at write inference.','validation':'Cache values and writes against native saved32historical cases; install two-position write at native layer13attention boundary, downstream suffix retained.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'validation.npz').exists();torch.set_num_threads(6)
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);F=torch.nn.functional
means=np.load(A/'head13_cached_write_program/frozen_mean_routing.npz')['mean'];b0=model.transformer.h[0];a13=model.transformer.h[13].attn
weights={'value0_head0':b0.attn.c_v.weight[:128].detach().float().clone(),'output13_head0':a13.c_proj.weight[:,:128].detach().float().clone(),'residual0':b0.lambdas[0].detach().float().clone(),'embedding0':b0.lambdas[1].detach().float().clone(),'cache_mix13':a13.lamb.detach().float().clone(),'routing_mean':torch.tensor(means,device='cuda',dtype=torch.float32)}
torch.save({k:v.cpu() for k,v in weights.items()},out/'weights.pt');loaded={k:v.cuda() for k,v in torch.load(out/'weights.pt',weights_only=True).items()}
meta=json.loads((A/'language_boundary_factorial/metadata.json').read_text());cache=torch.load(A/'language_boundary_factorial/boundary_states.pt',weights_only=True);old=np.load(A/'language_boundary_factorial/corners.npz');reference=np.load(A/'head13_cached_write_program/outputs.npz');previous=json.loads((A/'bilin18_head13_cached_write_program.json').read_text());ids=[e['ids'][int(old['source_positions'][i])] for i,e in enumerate(meta)];donids=[ids[int(j)] for j in old['donor_ids']]
with torch.no_grad():emb=model.transformer.wte(torch.tensor(ids,device='cuda'));donemb=model.transformer.wte(torch.tensor(donids,device='cuda'));values=program.cache_value(loaded,emb);writes=program.delta_writes(loaded,emb,donemb);identity=program.delta_writes(loaded,emb,emb)
cache_ref=np.stack([cache[i][2][0,int(old['source_positions'][i]),0].numpy() for i in range(32)]);write_ref=reference['mean'][None,:,None]*reference['payloads'][:,None,:];cache_err=float(np.linalg.norm(values.cpu().numpy()-cache_ref)/np.linalg.norm(cache_ref));write_err=float(np.linalg.norm(writes.cpu().numpy()-write_ref)/np.linalg.norm(write_ref))
@torch.no_grad()
def suffix(raw,x0,bus,ans,write=None):
 x=raw
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus)
  if l==13 and write is not None:
   at=at.clone();at[:,-2:]+=write[None]
  x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);return float(y[0,ans[0]]-y[0,ans[1]])
changes=[];replay=0
for i,e in enumerate(meta):
 raw,x0,bus=[v.cuda() for v in cache[i]];assert int(old['source_positions'][i])==raw.shape[1]-2;ans=[g[0] for g in e['groups']];base=suffix(raw,x0,bus,ans);changed=suffix(raw,x0,bus,ans,writes[i])-base;changes.append(changed);ref=next(r['change'] for r in previous['records'] if r['row']==i and r['arm']=='discovery_mean_routing');replay=max(replay,abs(changed-ref))
np.savez(out/'validation.npz',recipient_embeddings=emb.detach().cpu().numpy(),donor_embeddings=donemb.detach().cpu().numpy(),cache_values=values.cpu().numpy(),writes=writes.cpu().numpy(),effect_changes=changes)
summary=[]
for cohort in sorted({e['cohort'] for e in meta}):
 ix=[i for i,e in enumerate(meta) if e['cohort']==cohort];target=np.array([next(r['change'] for r in previous['records'] if r['row']==i and r['arm']=='native_cache13') for i in ix]);y=np.array(changes)[ix];summary.append({'cohort':cohort,'isolated_path_relative_l2':float(np.linalg.norm(y-target)/np.linalg.norm(target))})
count=sum(v.numel() for v in weights.values());manifest={'input_dimension':1152,'cached_dimension':128,'output_offsets':[0,1],'weight_scalars':count,'tensor_storage_bytes':4*count,'file_bytes':(out/'weights.pt').stat().st_size,'per_pair_matrix_multiply_accumulates':2*128*1152+1152*128,'external_ports':['recipient raw source-token embedding','donor raw source-token embedding'],'does_not_compute':['embedding lookup','context/QK routing','baseline attention','downstream response'],'sha256':{f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ['weights.pt','execute.py']},'routing_fit_rows':8}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');audit={'native_cache_relative_error':cache_err,'native_fixed_routing_write_relative_error':write_err,'installed_effect_replay_max_abs':replay,'identity_write_max_abs':float(identity.abs().max()),'standalone_write_qk_calls':0,'standalone_write_prefix_calls':0,'suffix_validation_sequences':64,'finite':bool(torch.isfinite(writes).all()),'new_reserved_test_examples':0};assert cache_err<1e-5 and write_err<1e-5 and replay<1e-4 and audit['identity_write_max_abs']==0
(A/f'{S}.json').write_text(json.dumps({'config':config,'manifest':manifest,'summary':summary,'audits':audit},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audit,'manifest':manifest},indent=2))
