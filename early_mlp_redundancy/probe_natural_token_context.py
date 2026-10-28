"""M4 fixed reused natural-text cohort and token/context partition."""
import json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent; A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 source=A/'bilin18_early_bsf_data_manifest.json';m=json.loads(source.read_text());train=[r for r in m['records'] if r['split']=='train'];reserved={r['document_hash'] for r in m['records'] if r['split']!='train'}
 order=np.random.default_rng(2026102804).permutation(len(train))[:64];records=[];tokens=[];hashes={}
 for i in order:
  r=train[int(i)];assert r['document_hash'] not in reserved
  f=Path(r['source'])
  if str(f) not in hashes:hashes[str(f)]=sha(f);assert hashes[str(f)]==m['source_hashes'][str(f)]
  ids=np.asarray(np.load(f,mmap_mode='r')[r['source_row'],:128],dtype=np.int64);assert hashlib.sha256(ids.tobytes()).hexdigest()==r['window_sha256_int64']
  records.append(r);tokens.append(ids.tolist())
 out=dict(seed=2026102804,source_manifest=str(source),source_manifest_sha256=sha(source),source_hashes=hashes,records=records,token_ids=tokens,scope='64 reused BSF training documents, first128 tokens of existing windows; not necessarily document beginnings. No model-training or scientific untouched holdout claim. BSF validation/test document hashes excluded.',cache_decision='Inspected early BSF data producer: stored sites are value0 and post-block residual0/1/2 at16positions, not native MLP writes. Replay pinned native model.',protocol='Frozen before forwards. Native product writes excluding Down bias, layers0/1/2, all128positions. Occurrence and complete causal-prefix deduplicated sample ANOVA, all tokens and >=2distinct-prefix tokens. Additionally symmetric document-half token-mean prediction on covered token IDs; report coverage and same-half global-mean baseline. Each half32documents. These splits are within previously used training data, not untouched validation. No sweeps or whole-task holdout exposure.')
 (P/'validation/natural_token_context_cohort.json').write_text(json.dumps(out,indent=2)+'\n');print('FROZEN64',flush=True)
def run():
 cp=P/'validation/natural_token_context_cohort.json';cohort=json.loads(cp.read_text());torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);model,receipt=load_model();device=next(model.parameters()).device;ys={l:[] for l in range(3)};prefixmap={};unique=[];toks=[];docs=[]
 for d,ids in enumerate(cohort['token_ids']):
  _,c=trace(model,torch.tensor([ids],device=device))
  for l in range(3):ys[l].append(model.transformer.h[l].mlp.Down(c[l]['products'][0]).cpu())
  for pos,t in enumerate(ids):
   key=tuple(ids[:pos+1])
   if key not in prefixmap:prefixmap[key]=len(toks);unique.append(len(toks))
   toks.append(t);docs.append(d)
  if d%16==15:print('FORWARDS',d+1,flush=True)
 saved=dict(Y={l:torch.cat(ys[l]) for l in range(3)},tokens=torch.tensor(toks),documents=torch.tensor(docs),unique_indices=torch.tensor(unique));dest=A/'emlp_m4_natural_writes.pt';torch.save(saved,dest)
 toks=np.array(toks);docs=np.array(docs);unique=np.array(unique);types,counts=np.unique(toks[unique],return_counts=True);contexts=dict(zip(types,counts));rows=[];prediction=[]
 for l in range(3):
  y=saved['Y'][l].double().numpy()
  for sampling,ix in [('occurrences',np.arange(len(toks))),('unique_prefixes',unique)]:
   for subset in ['all_tokens','repeated_context_tokens']:
    idx=ix if subset=='all_tokens' else ix[np.array([contexts[t]>=2 for t in toks[ix]])];z=y[idx];tt=toks[idx];means=np.empty_like(z)
    for t in np.unique(tt):means[tt==t]=z[tt==t].mean(0)
    centered=z-z.mean(0);between=means-z.mean(0);within=z-means;total=float((centered**2).sum());be=float((between**2).sum());we=float((within**2).sum());assert abs(total-be-we)<1e-10*total
    rows.append(dict(layer=l,sampling=sampling,subset=subset,rows=len(idx),token_types=len(np.unique(tt)),centered_energy=total,between_energy=be,within_energy=we,within_fraction=we/total,within_over_raw=float(we/(z*z).sum())))
  for trainhalf in [0,1]:
   train=unique[(docs[unique]//32)==trainhalf];test=unique[(docs[unique]//32)!=trainhalf];means={int(t):y[train[toks[train]==t]].mean(0) for t in np.unique(toks[train])};covered=test[np.isin(toks[test],list(means))];pred=np.stack([means[int(t)] for t in toks[covered]]);err=float(((y[covered]-pred)**2).sum());base=float(((y[covered]-y[train].mean(0))**2).sum())
   prediction.append(dict(layer=l,train_half=trainhalf,test_rows=len(test),covered_rows=len(covered),coverage=len(covered)/len(test),covered_token_types=len(np.unique(toks[covered])),squared_error=err,global_baseline_error=base,relative_to_global_baseline=err/base,relative_to_raw=float(err/(y[covered]**2).sum())))
 out=dict(status='completed',cohort_sha256=sha(cp),source_sha256=sha(Path(__file__)),checkpoint=asdict(receipt),forward_count=64,cache=str(dest),cache_sha256=sha(dest),positions=len(toks),unique_prefixes=len(unique),token_types=len(types),single_prefix_token_types=int(sum(counts==1)),rows=rows,cross_document_prediction=prediction)
 (P/'validation/natural_token_context.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in rows:
  if r['sampling']=='unique_prefixes':print(r['layer'],r['subset'],r['within_fraction'],flush=True)
 print('PREDICTION',[(r['layer'],r['train_half'],round(r['coverage'],3),round(r['relative_to_global_baseline'],3)) for r in prediction],flush=True)
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='run':run()
 else:raise ValueError('prepare or run')
