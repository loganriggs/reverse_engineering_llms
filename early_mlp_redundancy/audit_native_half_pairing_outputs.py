"""M47 independent donor permutation, coverage, predictions and output metrics."""
from pathlib import Path
import json,hashlib,collections,itertools
import numpy as np
import torch
P=Path(__file__).resolve().parent/'validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert np.isclose(a,b,rtol=1e-7,atol=1e-7),(a,b)
def ls(a):
 a=a.astype(float);a-=a.max(-1,keepdims=True);return a-np.log(np.exp(a).sum(-1,keepdims=True))
def main():
 torch.set_num_threads(1);rp=P/'native_half_pairing_outputs.json';r=json.loads(rp.read_text());pp=P/'native_half_pairing_native_protocol.json';assert sha(pp)==r['protocol_sha256'];q=json.loads(pp.read_text());assert q['source_sha256']==r['source_sha256']==sha(P.parent/'probe_native_half_pairing_outputs.py');pr=P/'native_half_pairing.json';assert sha(pr)==r['pairing_result_sha256']==q['pairing_result_sha256'];old=json.loads(pr.read_text());cache=Path(old['cache_path']);assert sha(cache)==old['cache_sha256'];data=torch.load(cache,weights_only=True);meta=data['metadata'];mp=P/'native_half_token_controls.json';assert sha(mp)==r['models_sha256'];states={}
 for e in json.loads(mp.read_text())['models']:
  f=Path(e['path']);assert sha(f)==e['sha256'];states[e['layer'],e['removed']]=torch.load(f,weights_only=True)
 known=set(next(iter(states.values()))['token_ids'].tolist());groups=collections.defaultdict(list)
 for i,x in enumerate(meta):groups[x['domain'],x['split'],x['token']].append(i)
 donor=np.arange(len(meta));eligible=np.zeros(len(meta),dtype=bool)
 for (_,_,tok),ii in groups.items():
  if tok not in known or len(ii)<2:continue
  ordered=sorted(ii,key=lambda i:hashlib.sha256(json.dumps(meta[i]['prefix'],separators=(',',':')).encode()).hexdigest());donor[ordered]=ordered[1:]+ordered[:1];eligible[ordered]=True
 assert np.array_equal(np.sort(donor),np.arange(len(meta))) and np.all(donor[eligible]!=np.arange(len(meta))[eligible])
 for i,j in enumerate(donor):assert all(meta[i][k]==meta[j][k] for k in ['domain','split','token'])
 sp=Path(r['samples_path']);assert sha(sp)==r['samples_sha256'];saved=torch.load(sp,weights_only=True);assert np.array_equal(saved['donors'].numpy(),donor) and np.array_equal(saved['eligible'].numpy(),eligible)
 idx={(x['domain'],x['split'],tuple(x['prefix'])):i for i,x in enumerate(meta)};panel=json.loads((P/'training_source_context_audit.json').read_text());cohort=json.loads((P/'natural_token_context_cohort.json').read_text());train=set(json.loads((P/'diverse_supervision_split.json').read_text())['train_pair_ids']);inputs={}
 for p in panel['rows']:
  for ep in [0,1]:inputs[f"task:{p['id']}:{ep}"]=('task','train' if p['id'] in train else 'evaluation',p['token_ids'][ep])
 for doc in range(32,64):inputs[f'natural:{doc}']=('natural','evaluation',cohort['token_ids'][doc])
 seen=set()
 for x in r['rows']:
  key=x['key'],x['layer'],x['removed'],x['arm'];assert key not in seen;seen.add(key);dom,part,tokens=inputs[x['key']];ii=[idx[dom,part,tuple(tokens[:j+1])] for j in range(len(tokens))];assert x['domain']==dom and x['split']==part and x['positions']==len(tokens) and x['eligible_positions']==int(eligible[ii].sum())
  if not x['eligible_positions']:assert x['local_error_energy']==x['mean_KL']==x['last_KL']==0
 assert seen==set(itertools.product(inputs,[0,1,2],['half_01','half_23'],['paired','permuted']));assert r['forward_count']==192+sum(x['eligible_positions']>0 for x in r['rows'])
 decoder_errors=[];metric_errors=[]
 for s in saved['samples']:
  row=r['rows'][s['row']];ii=s['indices'].numpy();jj=ii if row['arm']=='paired' else donor[ii];assert np.array_equal(s['selected'].numpy(),jj);assert np.array_equal(s['eligible'].numpy()[0],eligible[ii]);st=states[row['layer'],row['removed']];other='half_23' if row['removed']=='half_01' else 'half_01';X=data['halves'][f"{row['layer']}:{other}"].numpy().astype(float)[jj];tok=np.array([meta[j]['token'] for j in jj]);ti=np.minimum(np.searchsorted(st['token_ids'].numpy(),tok),len(st['token_ids'])-1);pred=(X-st['token_input_mean'].numpy()[ti])@st['U'].numpy()@st['B'].numpy()+st['token_target_mean'].numpy()[ti];pred[~eligible[jj]]=0
  rec=s['retained'].numpy()+pred.astype(np.float32)[None]+s['bias'].numpy();nw=s['native_write'].numpy();write=np.where(eligible[ii][None,:,None],rec,nw);cached=s['write'].numpy();err=np.linalg.norm(write.astype(float)-cached)/np.linalg.norm(cached);decoder_errors.append(err);assert err<1e-6;assert np.array_equal(cached[:,~eligible[ii]],nw[:,~eligible[ii]]);close(np.sum((cached.astype(float)-nw)**2),row['local_error_energy']);close(np.sum(s['net'].numpy().astype(float)[0,eligible[ii]]**2),row['eligible_native_energy'])
  a=s['native'].numpy();b=s['output'].numpy();lp=ls(a);lq=ls(b);kl=(np.exp(lp)*(lp-lq)).sum(-1);tokens=s['ids'].numpy();vals=dict(mean_KL=kl.mean(),last_KL=kl[0,-1],observed_next_token_dCE=np.take_along_axis(lp[:,:-1]-lq[:,:-1],tokens[:,1:,None],axis=-1).mean(),top1_agreement=(a.argmax(-1)==b.argmax(-1)).mean())
  for k,v in vals.items():close(v,row[k]);metric_errors.append(abs(v-row[k]))
 for s in r['summaries']:
  rr=[x for x in r['rows'] if all(x[k]==s[k] for k in ['domain','split','layer','arm'])];assert len(rr)==s['n']
  for f in ['mean_KL','last_KL','observed_next_token_dCE','top1_agreement']:close(sum(x[f] for x in rr)/len(rr),s[f])
  for f in ['local_error_energy','eligible_native_energy']:close(sum(x[f] for x in rr),s[f])
 out=dict(status='passed',result_sha256=sha(rp),rows_checked=len(seen),eligible_unique_prefixes=int(eligible.sum()),donor_bijection_no_fixed_points=True,independent_numpy_samples=len(saved['samples']),max_write_relative_error=max(decoder_errors),max_output_metric_error=max(metric_errors),scope='Full deterministic donor replay, eligibility/inventory/summaries, independent sampled predictions/output metrics and unchanged ineligible writes. Unique-prefix marginals preserved; occurrence weighting and oracle donor information remain limitations.')
 (P/'native_half_pairing_outputs_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
