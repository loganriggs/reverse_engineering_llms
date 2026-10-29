"""M46 exact within-token wrong-context expectation on unique causal prefixes."""
from pathlib import Path
import json,hashlib,collections
import torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
 pp=V/'native_half_pairing_protocol.json';q=json.loads(pp.read_text());assert sha(Path(__file__))==q['source_sha256'];mp=V/'native_half_token_controls.json';assert sha(mp)==q['models_sha256'];manifest=json.loads(mp.read_text());cp=V/'natural_token_context_cohort.json';assert sha(cp)==q['cohort_sha256'];cohort=json.loads(cp.read_text());panel=json.loads((V/'training_source_context_audit.json').read_text());split=json.loads((V/'diverse_supervision_split.json').read_text());train=set(split['train_pair_ids']);masks=json.loads((V/'mlp1_group_deletion.json').read_text())['removed_sets'];inputs=[]
 for pair in panel['rows']:
  for ep in [0,1]:inputs.append(dict(key=f"task:{pair['id']}:{ep}",ids=pair['token_ids'][ep],domain='task',split='train' if pair['id'] in train else 'evaluation'))
 for doc in range(32,64):inputs.append(dict(key=f'natural:{doc}',ids=cohort['token_ids'][doc],domain='natural',split='evaluation'))
 cache=A/'emlp_m46_unique_half_writes.pt'
 if cache.exists():
  data=torch.load(cache,weights_only=True);assert data['protocol_sha256']==sha(pp)
 else:
  model,receipt=load_model();device=next(model.parameters()).device;seen=set();metadata=[];halves={f'{l}:{h}':[] for l in [0,1,2] for h in ['half_01','half_23']};occurrences=collections.Counter()
  for i,c in enumerate(inputs):
   ids=torch.tensor([c['ids']],device=device);_,cap=trace(model,ids);selected=[]
   for pos,tok in enumerate(c['ids']):
    occurrences[c['domain']+':'+c['split']]+=1;prefix=tuple(c['ids'][:pos+1]);key=c['domain'],c['split'],prefix
    if key in seen:continue
    seen.add(key);selected.append(pos);metadata.append(dict(domain=c['domain'],split=c['split'],token=tok,prefix=list(prefix),source_key=c['key'],position=pos))
   for layer in [0,1,2]:
    mlp=model.transformer.h[layer].mlp;prod=cap[layer]['products']
    for half in ['half_01','half_23']:
     mask=torch.zeros(4608,device=device);mask[masks[half]]=1;halves[f'{layer}:{half}'].append(mlp.Down(prod*mask)[0,selected].cpu())
   if (i+1)%32==0:print('INPUTS',i+1,len(inputs),flush=True)
  halves={k:torch.cat(v) for k,v in halves.items()};data=dict(protocol_sha256=sha(pp),checkpoint_sha256=receipt.weights_sha256,metadata=metadata,halves=halves,occurrences=dict(occurrences),forward_count=len(inputs));torch.save(data,cache)
 rows=[];coverage=[];metadata=data['metadata'];groups=collections.defaultdict(list)
 for i,x in enumerate(metadata):groups[x['domain'],x['split'],x['token']].append(i)
 keys=None
 for entry in manifest['models']:
  path=Path(entry['path']);assert sha(path)==entry['sha256'];st=torch.load(path,weights_only=True);lookup={int(t):i for i,t in enumerate(st['token_ids'])};keys=set(lookup);X=data['halves'][f"{entry['layer']}:{entry['retained']}"].double();Y=data['halves'][f"{entry['layer']}:{entry['removed']}"].double()
  for (domain,part,tok),indices in groups.items():
   if tok not in lookup or len(indices)<2:continue
   j=lookup[tok];x=X[indices];y=Y[indices]-st['token_target_mean'][j];pred=(x-st['token_input_mean'][j])@st['U']@st['B'];n=len(indices);paired=float((pred-y).square().sum());p2=float(pred.square().sum());y2=float(y.square().sum());cross=float(pred.sum(0)@y.sum(0));wrong=(n*(p2+y2)-2*cross-paired)/(n-1);cov=float(((pred-pred.mean(0))*(y-y.mean(0))).sum());assert abs(wrong-paired-2*n/(n-1)*cov)<1e-6*max(abs(wrong),1)
   rows.append(dict(domain=domain,split=part,token=tok,layer=entry['layer'],removed=entry['removed'],contexts=n,paired_error=paired,wrong_context_expected_error=wrong,token_mean_error=y2,pairing_gain=wrong-paired,centered_cross_inner_product=cov))
 for domain,part in [('task','train'),('task','evaluation'),('natural','evaluation')]:
  gg={t:idx for (d,s,t),idx in groups.items() if (d,s)==(domain,part)};eligible={t:idx for t,idx in gg.items() if t in keys and len(idx)>=2};coverage.append(dict(domain=domain,split=part,original_occurrences=data['occurrences'][domain+':'+part],unique_prefixes=sum(map(len,gg.values())),unknown_token_prefixes=sum(len(idx) for t,idx in gg.items() if t not in keys),known_single_context_prefixes=sum(len(idx) for t,idx in gg.items() if t in keys and len(idx)<2),eligible_prefixes=sum(map(len,eligible.values())),eligible_token_types=len(eligible)))
 summaries=[]
 for c in coverage:
  for layer in [0,1,2]:
   rr=[x for x in rows if x['domain']==c['domain'] and x['split']==c['split'] and x['layer']==layer];s={k:c[k] for k in ['domain','split']};s.update(layer=layer,token_mask_groups=len(rr),context_mask_cases=sum(x['contexts'] for x in rr),**{k:sum(x[k] for x in rr) for k in ['paired_error','wrong_context_expected_error','token_mean_error','pairing_gain']});s['paired_over_wrong']=s['paired_error']/s['wrong_context_expected_error'];s['paired_over_token_mean']=s['paired_error']/s['token_mean_error'];summaries.append(s)
 out=dict(protocol_sha256=sha(pp),models_sha256=sha(mp),cache_path=str(cache),cache_sha256=sha(cache),forward_count=data['forward_count'],checkpoint_sha256=data['checkpoint_sha256'],coverage=coverage,rows=rows,summaries=summaries,scope='Unique causal prefixes within each domain/split; known tokens with2+contexts; exact expected other-context squared error, not output causal intervention or independent samples.')
 op=V/'native_half_pairing.json';assert not op.exists();op.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(coverage=coverage,summaries=summaries),indent=2))
if __name__=='__main__':main()
