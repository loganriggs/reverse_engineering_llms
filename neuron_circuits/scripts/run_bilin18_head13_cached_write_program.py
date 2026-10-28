"""H1022 exact cached OV write and discovery-only routing simplification."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_head13_cached_write_program';out=A/'head13_cached_write_program';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
meta=json.loads((A/'language_boundary_factorial/metadata.json').read_text());cache=torch.load(A/'language_boundary_factorial/boundary_states.pt',weights_only=True);old=np.load(A/'language_boundary_factorial/corners.npz');previous=json.loads((A/'bilin18_cache_consumer_localization.json').read_text());train=[i for i,e in enumerate(meta) if e['cohort']=='discovery'];config={'experiment':'H1022','equation':'delta_write(t)=lambda13 * P13,head0(t,source) * W_O13,head0 * delta_cached_value0(source); P=(Q.K/128)*(Q2.K2/128), causal mask, no softmax','arms':['native_cache13','exact_write','discovery_mean_routing','last_only','source_only'],'routing_mean':'Fit2offset coefficients from8discovery rows only; same mean used for bothtemplates and alllexical examples','scope':'Conditional exact port-to-write program. Native prefix/QK and baseline attention/suffix remain charged; not whole-head or whole-model compression. Same native isolated cache intervention asH1021.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'outputs.npz').exists();torch.set_num_threads(6)
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);F=torch.nn.functional;att=model.transformer.h[13].attn;original=att.squared_attention;payloads=[];routing=[];native_deltas=[];programs=[];maxwrite=0
@torch.no_grad()
def pre13(raw,x0,bus):
 x=raw
 for l in [11,12]:
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 b=model.transformer.h[13];return F.rms_norm(b.lambdas[0]*x+b.lambdas[1]*x0,(1152,))
for i,e in enumerate(meta):
 raw,x0,bus=[v.cuda() for v in cache[i]];source=int(old['source_positions'][i]);don=cache[int(old['donor_ids'][i])][2].cuda();edited=bus.clone();edited[0,source,0]=don[0,source,0];stash={}
 def capture(q,k,v,q2,k2):
  a=(q[0,:,0].double()@k[0,source,0].double()/128)*(q2[0,:,0].double()@k2[0,source,0].double()/128);a[:source]=0;stash['coeff']=a.cpu().numpy();return original(q,k,v,q2,k2)
 att.squared_attention=capture
 try:
  with torch.no_grad():state=pre13(raw,x0,bus);base,_=att(state,bus);edited_out,_=att(state,edited)
 finally:att.squared_attention=original
 coeff=stash['coeff'];dv=(don[0,source,0]-bus[0,source,0]).cpu().double().numpy();W=att.c_proj.weight[:,:128].detach().cpu().double().numpy();payload=float(att.lamb.detach())*(W@dv);compiled=coeff[:,None]*payload[None,:];native=(edited_out-base)[0].cpu().double().numpy();maxwrite=max(maxwrite,float(np.linalg.norm(compiled-native)/np.linalg.norm(native)));assert len(coeff[source:])==2
 routing.append(coeff[source:]);payloads.append(payload);native_deltas.append(native);programs.append(compiled)
routing=np.stack(routing);payloads=np.stack(payloads);mean=routing[train].mean(0);np.savez(out/'frozen_mean_routing.npz',mean=mean,training_indices=train);mean_hash=hashlib.sha256((out/'frozen_mean_routing.npz').read_bytes()).hexdigest()
@torch.no_grad()
def suffix(raw,x0,bus,edited,program,arm,ans):
 x=raw
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  supplied=edited if l==13 and arm=='native_cache13' else bus
  at,_=b.attn(F.rms_norm(x,(1152,)),supplied)
  if l==13 and arm not in ['native_cache13','base']:at=at+torch.tensor(program,device='cuda',dtype=at.dtype)[None]
  x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);return float(y[0,ans[0]]-y[0,ans[1]])
records=[];maxreplay=0;exacterr=0
for i,e in enumerate(meta):
 raw,x0,bus=[v.cuda() for v in cache[i]];don=cache[int(old['donor_ids'][i])][2].cuda();source=int(old['source_positions'][i]);edited=bus.clone();edited[0,source,0]=don[0,source,0];ans=[g[0] for g in e['groups']];base=suffix(raw,x0,bus,edited,None,'base',ans);meanprog=np.zeros_like(programs[i]);meanprog[source:]=mean[:,None]*payloads[i];last=programs[i].copy();last[:-1]=0;src=programs[i].copy();src[-1]=0
 for arm,program in [('native_cache13',None),('exact_write',programs[i]),('discovery_mean_routing',meanprog),('last_only',last),('source_only',src)]:
  value=suffix(raw,x0,bus,edited,program,arm,ans);change=value-base;records.append(dict(row=i,cohort=e['cohort'],arm=arm,change=change))
  if arm=='native_cache13':
   reference=change;prev=next(r['change'] for r in previous['records'] if r['row']==i and r['payload']=='headslot0' and r['arm']=='only13');maxreplay=max(maxreplay,abs(reference-prev))
  if arm=='exact_write':exacterr=max(exacterr,abs(change-reference))
summary=[]
for co in sorted({e['cohort'] for e in meta}):
 rr=[r for r in records if r['cohort']==co];reference=np.array([r['change'] for r in rr if r['arm']=='native_cache13']);ii=[i for i,e in enumerate(meta) if e['cohort']==co]
 for arm in config['arms']:
  y=np.array([r['change'] for r in rr if r['arm']==arm]);summary.append(dict(cohort=co,arm=arm,relative_l2=float(np.linalg.norm(y-reference)/np.linalg.norm(reference)),aligned_recovery=float(y@reference/(reference@reference)),reference_rms=float(np.sqrt(np.mean(reference**2)))))
np.savez(out/'outputs.npz',routing=routing,payloads=payloads,mean=mean,changes=np.array([r['change'] for r in records]));torch.save({'programs':programs,'native_deltas':native_deltas},out/'write_replay.pt')
audit={'native_write_relative_error_max':maxwrite,'prior_effect_replay_max_abs':maxreplay,'compiled_effect_replay_max_abs':exacterr,'frozen_mean_hash_unchanged':mean_hash==hashlib.sha256((out/'frozen_mean_routing.npz').read_bytes()).hexdigest(),'finite':all(np.isfinite(r['change']) for r in records),'pre13_prefix_sequences':32,'head13_local_calls':64,'suffix_sequences':192,'evaluation_fit_rows':0,'new_reserved_test_examples':0}
assert maxwrite<1e-5 and maxreplay<1e-4 and exacterr<1e-4 and audit['finite'];(A/f'{S}.json').write_text(json.dumps({'config':config,'summary':summary,'records':records,'audits':audit},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audit},indent=2))
