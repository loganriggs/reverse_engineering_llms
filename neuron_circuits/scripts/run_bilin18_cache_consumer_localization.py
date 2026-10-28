"""H1021 isolate consumers of the cached-value payload without persisting swaps."""
from pathlib import Path
import sys,json,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_cache_consumer_localization';out=A/'cache_consumer_localization';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
meta=json.loads((A/'language_boundary_factorial/metadata.json').read_text());cache=torch.load(A/'language_boundary_factorial/boundary_states.pt',weights_only=True);prior=np.load(A/'language_boundary_factorial/corners.npz');slots=json.loads((A/'bilin18_language_cache_features.json').read_text());layers=list(range(11,18));arms={'all':layers}|{f'only{l}':[l] for l in layers}|{f'except{l}':[k for k in layers if k!=l] for l in layers}
config={'experiment':'H1021','payloads':['all_cache','headslot0','headslot4'],'arms':arms,'design':'At each suffix layer explicitly supply recipient or edited cache according to fixed arm. Discard returned bus rather than persist an edited bus into later layers. All other residual paths live. Single-consumer and leave-one-consumer-out for layers11–17.','scope':'Causal consumer attribution of three observed cache payloads, not root cause of whole model or feature/manifold recovery. Historical evaluation and discovery reported separately; no reader selected by evaluation.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'responses.npz').exists();torch.set_num_threads(6)
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);F=torch.nn.functional
@torch.no_grad()
def suffix(raw,x0,recipient,edited,selected,ans):
 x=raw
 for l in layers:
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  supplied=edited if l in selected else recipient
  at,returned=b.attn(F.rms_norm(x,(1152,)),supplied)
  assert returned.data_ptr()==supplied.data_ptr()
  x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);return float(y[0,ans[0]]-y[0,ans[1]])
records=[];maxbase=maxall=maxslot=maxidentity=0
for i,e in enumerate(meta):
 raw,x0,bus=[v.cuda() for v in cache[i]];don=cache[int(prior['donor_ids'][i])][2].cuda();ans=[g[0] for g in e['groups']];source=int(prior['source_positions'][i]);base=suffix(raw,x0,bus,bus,[],ans);maxbase=max(maxbase,abs(base-prior['corners'][i,0]));original=bus.clone()
 payloads={'all_cache':don};
 for head in [0,4]:
  bb=bus.clone();bb[0,source,head]=don[0,source,head];payloads[f'headslot{head}']=bb
 for name,bb in payloads.items():
  for arm,selected in arms.items():
   value=suffix(raw,x0,bus,bb,selected,ans);change=value-base
   if arm=='all':
    if name=='all_cache':maxall=max(maxall,abs(value-prior['corners'][i,16]))
    else:
     previous=next(r['change'] for r in slots['records'] if r['row']==i and r['arm']==name);maxslot=max(maxslot,abs(change-previous))
   records.append(dict(row=i,cohort=e['cohort'],pair=e['pair'],payload=name,arm=arm,change=change))
 assert torch.equal(bus,original)
 # Equal donor/recipient must be identity even when an isolated layer is selected.
 maxidentity=max(maxidentity,abs(suffix(raw,x0,bus,bus,[14],ans)-base))
 print('row',i,flush=True)
summary=[];chosen={}
for payload in payloads:
 for cohort in sorted({e['cohort'] for e in meta}):
  rr=[r for r in records if r['cohort']==cohort and r['payload']==payload];full=np.array([r['change'] for r in rr if r['arm']=='all']);den=full@full;metrics=[]
  for l in layers:
   single=np.array([r['change'] for r in rr if r['arm']==f'only{l}']);without=np.array([r['change'] for r in rr if r['arm']==f'except{l}']);metrics.append({'layer':l,'single_aligned_share':float(single@full/den),'single_relative_l2':float(np.linalg.norm(single-full)/np.linalg.norm(full)),'omission_damage_share':float((full-without)@full/den)})
  summed=sum(np.array([r['change'] for r in rr if r['arm']==f'only{l}']) for l in layers)
  summary.append(dict(payload=payload,cohort=cohort,effect_rms=float(np.sqrt(np.mean(full*full))),layers=metrics,single_sum_relative_l2=float(np.linalg.norm(summed-full)/np.linalg.norm(full))))
  if cohort=='discovery':chosen[payload]=max(metrics,key=lambda m:m['single_aligned_share'])['layer']
audit={'base_replay_max_abs':maxbase,'all_cache_replay_max_abs':maxall,'headslot_replay_max_abs':maxslot,'identity_swap_max_abs':maxidentity,'recipient_cache_mutated':False,'returned_bus_discarded_each_layer':True,'suffix_sequences':32*(1+3*15+1),'finite':all(np.isfinite(r['change']) for r in records),'new_reserved_test_examples':0}
assert max(maxbase,maxall,maxslot,maxidentity)<1e-4 and audit['finite'];result={'config':config,'summary':summary,'discovery_top_single_consumer':chosen,'records':records,'audits':audit};(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');np.savez(out/'responses.npz',changes=np.array([r['change'] for r in records]));print(json.dumps({k:result[k] for k in ['summary','discovery_top_single_consumer','audits']},indent=2))
