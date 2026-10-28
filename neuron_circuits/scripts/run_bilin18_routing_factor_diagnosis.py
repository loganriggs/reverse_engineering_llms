"""H1025 exact score-factor recombination; discovery countries only, no fitting."""
from pathlib import Path
import sys,json,math,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_routing_factor_diagnosis';out=A/'routing_factor_diagnosis';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
panel=json.loads((A/'lexical_routing_challenge/panel.json').read_text());pairs=sorted({e['pair'] for e in panel})[:4];examples=[e for e in panel if e['pair'] in pairs]
config={'experiment':'H1025','data':'First four original discovery country pairs, both endpoints, five H1024 contexts. Opened diagnostic contexts, not fresh validation.','factors':['normalized pre-rotary destination query pair','normalized pre-rotary source key pair','native source/destination rotary tables'],'design':'Eight recombination corners, baseline vs context for each factor. Shapley allocation of signed coefficient change. No fitted predictor, no behavioral intervention claim.','fit_rows':0}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(examples,indent=2)+'\n')
torch.set_num_threads(6);F=torch.nn.functional
model,receipt=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
states=[]
@torch.no_grad()
def capture(e):
 ids=torch.tensor([e['ids']],device='cuda');initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float();x,x0,bus,_,_=runner.component._capture(model,initial,torch,F)
 for l in [11,12]:
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 b=model.transformer.h[13];x=F.rms_norm(b.lambdas[0]*x+b.lambdas[1]*x0,(1152,));att=b.attn;T=x.shape[1];source=e['source'];d={}
 for name,pos in [('q',T-1),('q2',T-1),('k',source),('k2',source)]:
  v=getattr(att,'c_'+name)(x).view(1,T,9,128);d[name]=F.rms_norm(v,(128,))[0,pos,0].cpu()
 cos,sin=att.rotary(v)
 for name,pos in [('q',T-1),('k',source)]:
  d[name+'cos']=cos[0,pos,0].cpu();d[name+'sin']=sin[0,pos,0].cpu()
 original=att.squared_attention
 def hook(q,k,v,q2,k2):
  d['native']=float((q[0,-1,0].double()@k[0,source,0].double()/128)*(q2[0,-1,0].double()@k2[0,source,0].double()/128));return original(q,k,v,q2,k2)
 att.squared_attention=hook
 try:att(x,bus)
 finally:att.squared_attention=original
 return d
for e in examples:states.append(capture(e))
def rotate(v,c,s):
 a,b=v.chunk(2);return torch.cat([a*c+b*s,-a*s+b*c])
def score(q,k,r):
 dots=[]
 for name in ['', '2']:
  a=rotate(q['q'+name],r['qcos'],r['qsin']);b=rotate(k['k'+name],r['kcos'],r['ksin']);dots.append(float(a.double()@b.double()/128))
 return dots[0]*dots[1],dots
records=[];maxreplay=0;maxsum=0
for i,e in enumerate(examples):
 j=next(j for j,f in enumerate(examples) if f['format']=='baseline' and f['pair']==e['pair'] and f['order']==e['order']);base,ctx=states[j],states[i];corners=[]
 for mask in range(8):corners.append(score(ctx if mask&1 else base,ctx if mask&2 else base,ctx if mask&4 else base)[0])
 phi=[]
 for bit in range(3):
  value=0
  for mask in range(8):
   if mask&(1<<bit):continue
   n=mask.bit_count();value+=math.factorial(n)*math.factorial(2-n)/6*(corners[mask|(1<<bit)]-corners[mask])
  phi.append(value)
 maxreplay=max(maxreplay,abs(corners[0]-base['native']),abs(corners[7]-ctx['native']));maxsum=max(maxsum,abs(sum(phi)-(corners[7]-corners[0])))
 records.append(dict(row=i,pair=e['pair'],order=e['order'],format=e['format'],corners=corners,shapley=phi,baseline_scores=score(base,base,base)[1],context_scores=score(ctx,ctx,ctx)[1],delta=corners[7]-corners[0]))
summary=[]
for fmt in dict.fromkeys(e['format'] for e in examples):
 rr=[r for r in records if r['format']==fmt];d=np.array([r['delta'] for r in rr]);p=np.array([r['shapley'] for r in rr]);c=np.array([r['corners'] for r in rr]);summary.append(dict(format=fmt,mean_baseline=float(c[:,0].mean()),mean_context=float(c[:,7].mean()),mean_change=float(d.mean()),rms_change=float(np.sqrt(np.mean(d*d))),shapley_mean=p.mean(0).tolist(),shapley_rms=np.sqrt(np.mean(p*p,0)).tolist(),aligned_change_share=(p.T@d/(d@d)).tolist() if d@d>1e-20 else None))
audit={'native_coefficient_replay_max_abs':maxreplay,'shapley_sum_max_abs':maxsum,'finite':bool(np.isfinite([r['corners'] for r in records]).all()),'discovery_country_pairs':pairs,'prefix_sequences':40,'fit_rows':0,'reserved_test_examples':0,'behavioral_interventions':0}
assert maxreplay<1e-6 and maxsum<1e-12 and len(examples)==40 and audit['finite']
torch.save(states,out/'factors.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,audits=audit),indent=2))
