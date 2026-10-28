"""H1035: fixed stream projection factorial; no fitting or holdout tuning."""
from pathlib import Path
import sys,json,hashlib,itertools,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_projection_stream_factorial';out=A/'projection_stream_factorial';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
F=torch.nn.functional;torch.set_num_threads(6);torch.set_float32_matmul_precision('highest')
source=A/'selector_task_holdouts';data=torch.load(source/'states.pt',weights_only=False);old=json.loads((A/'bilin18_selector_task_holdouts.json').read_text());gates={(r['family'],r['split']) for r in old['competence'] if r['pass_gate']};pairs=[p for p in json.loads((source/'panel.json').read_text()) if (p['family'],p['split']) in gates];assert len(pairs)==16
pcafile=A/'multistream_span_diagnosis/pca.pt';files=[pcafile,source/'states.pt',source/'panel.json'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
config=dict(experiment='H1035',ports=['residual_pre_attention_11','initial_embedding','cached_first_layer_values'],projected_bits='1 projects port into discovery-trained PCA128; 0 preserves native port',selection='All16 pairs from H1034 prespecified competent cohorts. Historical whole-task training holdouts, now diagnostic only.',fit_rows=0,source_hashes=hashes,paired_target='Full donor output; recipient plus donor-minus-recipient hybrid representation',next_step_rule='Localize missing stream information before changing training geometry; no tuning on this panel.')
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
pca=torch.load(pcafile,weights_only=False);B=[b.float().cuda() for b in pca['bases']];mu=pca['center'].float().cuda();scale=pca['scale'].float().cuda();ix=sorted({i for p in pairs for i in p['indices']});states={i:tuple(x.cuda() for x in data['states'][i]) for i in ix};native={i:v.cuda() for i,v in data['native'].items() if i in ix}
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
@torch.no_grad()
def suffix(st):
 x,x0,bus=st
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 return (30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30))[0].double()
def KL(a,b):la=a.log_softmax(0);return float((la.exp()*(la-b.log_softmax(0))).sum())
def margin(y,g):return float(y[g[0]].logsumexp(0)-y[g[1]].logsumexp(0))
projected={};energy=[];records=[];samples=[];replay=0.;paired_replay=0.
with torch.no_grad():
 for i,st in states.items():
  parts=[]
  for k,x in enumerate(st):
   z=(x.reshape(-1,1152)-mu[k])/scale[k];pz=(z@B[k])@B[k].T;parts.append((pz*scale[k]+mu[k]).reshape_as(x))
   energy.append(dict(row=i,port=k,centered_retained_energy=float(pz.square().sum()/z.square().sum())))
  projected[i]=tuple(parts)
 for bits in itertools.product([0,1],repeat=3):
  name=''.join(map(str,bits));hybrid={i:tuple(projected[i][k] if bits[k] else states[i][k] for k in range(3)) for i in ix}
  for p in pairs:
   for i,d in [p['indices'],p['indices'][::-1]]:
    y=suffix(hybrid[i]);patch=tuple(states[i][k]+hybrid[d][k]-hybrid[i][k] for k in range(3));yp=suffix(patch);base=native[i];don=native[d];g=data['examples'][i]['groups'];end=p['indices'].index(i)
    if name=='000':replay=max(replay,float((y-base).abs().max()));paired_replay=max(paired_replay,float((yp-don).abs().max()))
    records.append(dict(arm=name,family=p['family'],split=p['split'],row=i,donor=d,whole_KL=KL(base,y),paired_KL=KL(don,yp),paired_denominator=KL(don,base),correct_top1=int(y.argmax()) in g[end],natural_delta=margin(don,g)-margin(base,g),predicted_delta=margin(yp,g)-margin(base,g)))
    if i==pairs[0]['indices'][0]:samples.append(dict(arm=name,row=i,base=base.cpu(),donor=don.cpu(),whole=y.cpu(),paired=yp.cpu()))
  print('EVALUATED',name,flush=True)
 for p in pairs:
  i,d=p['indices']
  for k in range(3):
   delta=states[d][k]-states[i][k];pd=projected[d][k]-projected[i][k];energy.append(dict(pair=p['id'],family=p['family'],split=p['split'],port=k,difference_retained_energy=float(pd.square().sum()/delta.square().sum()) if delta.square().sum()>0 else None))
summary=[]
for name in sorted({r['arm'] for r in records}):
 for f,sp in sorted(gates):
  rr=[r for r in records if (r['arm'],r['family'],r['split'])==(name,f,sp)];summary.append(dict(arm=name,family=f,split=sp,whole_KL=float(np.mean([r['whole_KL'] for r in rr])),paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),top1=float(np.mean([r['correct_top1'] for r in rr]))))
# Check projection agrees with previous execution on the identical frozen panel.
cross=max(abs(s[k]-next(r for r in old['summary'] if r['arm']=='PCA128' and r['family']==s['family'] and r['split']==s['split'])[k]) for s in summary if s['arm']=='111' for k in ['whole_KL','paired_relative_KL','top1'])
audit=dict(source_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),native_logit_replay_max=replay,native_donor_patch_logit_replay_max=paired_replay,previous_PCA128_summary_max_error=cross,suffix_sequences=len(records)*2,fit_rows=0,finite=all(np.isfinite(r[k]) for r in records for k in ['whole_KL','paired_KL','natural_delta','predicted_delta']))
assert audit['source_hashes_unchanged'] and audit['finite'] and replay<1e-5 and paired_replay<1e-3 and cross<1e-8
j=dict(config=config,summary=summary,records=records,energy=energy,audits=audit);torch.save(samples,out/'audit_logits.pt');(A/f'{S}.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(audit,indent=2))
