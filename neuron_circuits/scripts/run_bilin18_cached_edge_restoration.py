"""H1027 finite restoration of cached target edge with matched write controls."""
from pathlib import Path
import sys,json,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_cached_edge_restoration';out=A/'cached_edge_restoration';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
old=json.loads((A/'bilin18_routing_wording_controls.json').read_text());panel=json.loads((A/'routing_wording_controls/panel.json').read_text());examples=[e for e in panel if e['format']=='mention_before'];arms=['native','restore','reverse','random0','random1','wrong_country'];config={'experiment':'H1027','panel':'All128 H1026 neutral mention examples, four discovery pairs; no selection on outputs.','intervention':'Add (baseline coefficient - recipient coefficient)*lambda13*W_O13_head0*v1_head0(source) at final position after attention13. Only cached target edge changed. Baseline coefficient per target from H1026 is an oracle counterfactual, not learned predictor.','arms':arms,'controls':'reverse sign; two fixed Gaussian head-space vectors projected through same W_O and normalized to same output write norm; opposite-pair country cache payload, normalized to same write norm. Seeds1027,1028.','metrics':'Correct answer paired margin, full-vocabulary log probability/top1, recovery relative to no-prefix baseline. No removal of failures.','audit':'Replay native H1026 margins; compare post-projection write with pre-projection head-output installation for first8rows.','fit_rows':0}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(examples,indent=2)+'\n')
torch.set_num_threads(6);F=torch.nn.functional;model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);att=model.transformer.h[13].attn;W=att.c_proj.weight[:,:128];random=[]
for seed in [1027,1028]:
 g=torch.Generator(device='cuda').manual_seed(seed);v=W@torch.randn(128,generator=g,device='cuda');random.append(v/v.norm())
@torch.no_grad()
def capture(e):
 initial=F.rms_norm(model.transformer.wte(torch.tensor([e['ids']],device='cuda')),(1152,)).float();raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);return raw,x0,bus
@torch.no_grad()
def suffix(state,e,write=None,headwrite=None):
 x,x0,bus=state;original=att.squared_attention
 if headwrite is not None:
  def hook(q,k,v,q2,k2):
   z=original(q,k,v,q2,k2);z[0,0,-1]+=headwrite;return z
  att.squared_attention=hook
 try:
  for l in range(11,18):
   b=model.transformer.h[l]
   if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
   at,_=b.attn(F.rms_norm(x,(1152,)),bus)
   if l==13 and write is not None:at=at.clone();at[0,-1]+=write
   x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 finally:att.squared_attention=original
 y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);ans=[v[0] for v in e['groups']];target=ans[e['expected']];return dict(margin=float((y[0,ans[0]]-y[0,ans[1]])*(1 if e['expected']==0 else -1)),logp=float(F.log_softmax(y,dim=-1)[0,target]),top1=int(y.argmax(1)),correct_top1=bool(int(y.argmax(1))==target))
states=[capture(e) for e in examples];records=[];maxnative=maxreplay=maxnorm=0;norms=[]
with torch.no_grad():
 for i,e in enumerate(examples):
  o=next(r for r in old['records'] if panel[r['row']]['text']==e['text']);base=next(r for r in old['records'] if r['format']=='baseline' and r['pair']==e['pair'] and r['order']==e['order']);delta=base['coefficient']-o['coefficient'];bus=states[i][2];headwrite=delta*att.lamb*bus[0,e['source'],0];write=W@headwrite;norm=write.norm();norms.append(float(norm));mate=next(k for k,f in enumerate(examples) if f['pair']==e['pair'] and f['order']!=e['order'] and f['distractor']==e['distractor']);wrong=W@(delta*att.lamb*states[mate][2][0,e['source'],0]);wrong=wrong/wrong.norm()*norm
  writes={'native':None,'restore':write,'reverse':-write,'random0':random[0]*norm,'random1':random[1]*norm,'wrong_country':wrong};values={}
  for arm,w in writes.items():
   val=suffix(states[i],e,write=w);values[arm]=val;records.append(dict(row=i,cls=e['distractor_class'],pair=e['pair'],distractor=e['distractor'],arm=arm,baseline_margin=base['margin']*(1 if e['expected']==0 else -1),**val))
   if w is not None:maxnorm=max(maxnorm,float(abs(w.norm()-norm)/norm))
  maxnative=max(maxnative,abs(values['native']['margin']-o['margin']*(1 if e['expected']==0 else -1)))
  if i<8:
   replay=suffix(states[i],e,headwrite=headwrite);maxreplay=max(maxreplay,abs(replay['margin']-values['restore']['margin']),abs(replay['logp']-values['restore']['logp']));assert replay['top1']==values['restore']['top1']
summary=[]
for cls in ['country','city','person','abstract']:
 for arm in arms:
  rr=[r for r in records if r['cls']==cls and r['arm']==arm];native=[next(b for b in records if b['row']==r['row'] and b['arm']=='native') for r in rr];gain=np.array([r['margin']-b['margin'] for r,b in zip(rr,native)]);gap=np.array([r['baseline_margin']-b['margin'] for r,b in zip(rr,native)]);summary.append(dict(cls=cls,arm=arm,top1=float(np.mean([r['correct_top1'] for r in rr])),mean_margin=float(np.mean([r['margin'] for r in rr])),mean_margin_gain=float(gain.mean()),mean_logp_gain=float(np.mean([r['logp']-b['logp'] for r,b in zip(rr,native)])),fraction_margin_improved=float(np.mean(gain>0)),baseline_margin_gap=float(gap.mean()),aligned_baseline_gap_recovery=float(gain@gap/(gap@gap))))
audit={'native_margin_replay_max_abs':maxnative,'head_vs_output_installation_max_abs':maxreplay,'matched_write_norm_max_relative_error':maxnorm,'prefix_sequences':128,'suffix_sequences':776,'fit_rows':0,'reserved_test_examples':0,'finite':bool(np.isfinite([r['margin'] for r in records]).all())};assert maxnative<1e-4 and maxreplay<1e-4 and maxnorm<1e-6 and audit['finite'];(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,write_norms=norms,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,audits=audit),indent=2))
