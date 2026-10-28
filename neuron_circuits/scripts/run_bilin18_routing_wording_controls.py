"""H1026 fixed-position distractor-class and wording control, without fitting."""
from pathlib import Path
import sys,json,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_routing_wording_controls';out=A/'routing_wording_controls';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
enc=runner.original_authority.task14.ENCODING
old=json.loads((A/'routing_factor_diagnosis/panel.json').read_text());base=[e for e in old if e['format']=='baseline'];classes={'country':['Canada','Brazil','Mexico','India'],'city':['London','Paris','Boston','Berlin'],'person':['Alice','Peter','David','Mary'],'abstract':['music','science','poetry','history']}
templates={'contrast_before':'Unlike {d}, the main language spoken in {country} is','contrast_after':'The main language spoken in {country}, rather than {d}, is','mention_before':'The guide mentions {d}. The main language spoken in {country} is'}
examples=[]
for e in base:examples.append(dict(e,format='baseline',distractor_class='none',distractor='none'))
for fmt,template in templates.items():
 for cls,words in classes.items():
  for word in words:
   assert len(enc.encode(' '+word))==1
   for b in base:
    text=template.format(d=word,country=b['country']);ids=enc.encode(text);token=enc.encode(' '+b['country'])[0];assert ids.count(token)==1
    examples.append(dict(b,format=fmt,distractor_class=cls,distractor=word,text=text,ids=ids,source=ids.index(token)))
for fmt in templates:
 rows=[e for e in examples if e['format']==fmt];assert len({(len(e['ids']),e['source']) for e in rows})==1
assert len(examples)==392 and len({e['text'] for e in examples})==392
config={'experiment':'H1026','classes':classes,'templates':templates,'design':'4 original discovery pairs x2 endpoints x16 distractors x3wordings +8baseline. All distractors one GPT2token; source and destination positions fixed within wording.','hypotheses':['Country-specific competition: country distractors attenuate more than people/abstract nouns within wording.','Geographic association: cities resemble countries.','Wording gate: similar attenuation across distractor classes, strong wording effect.','General mention load: neutral mention prefix attenuates comparably to contrast.'],'primary':'Compare signed coefficient changes within matched wording and target; report per-lexeme ranges, no independent-prompt significance claims.','competence':'Report both-endpoint accuracy and fullvocabularytop1 for every class/wording; no exclusion or tuning.','limitations':'Classes differ in semantic plausibility and associations. Abstract nouns are lower-case; words are not frequency-matched. Fixed small discovery panel; no broad category claim.','fit_rows':0}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(examples,indent=2)+'\n')
torch.set_num_threads(6);F=torch.nn.functional;model,receipt=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
records=[]
@torch.no_grad()
def evaluate(e):
 initial=F.rms_norm(model.transformer.wte(torch.tensor([e['ids']],device='cuda')),(1152,)).float();x,x0,bus,_,_=runner.component._capture(model,initial,torch,F);att=model.transformer.h[13].attn;original=att.squared_attention;stash={}
 def hook(q,k,v,q2,k2):
  a=q[0,-1,0].double()@k[0,e['source'],0].double()/128;b=q2[0,-1,0].double()@k2[0,e['source'],0].double()/128;stash['scores']=[float(a),float(b)];stash['coefficient']=float(a*b);return original(q,k,v,q2,k2)
 att.squared_attention=hook
 try:
  for l in range(11,18):
   block=model.transformer.h[l]
   if l>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
   at,_=block.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+block.mlp(F.rms_norm(x,(1152,)))
 finally:att.squared_attention=original
 y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);answers=[g[0] for g in e['groups']];stash.update(margin=float(y[0,answers[0]]-y[0,answers[1]]),top1=int(y.argmax(1)),correct_top1=bool(int(y.argmax(1))==answers[e['expected']]),correct_binary=bool((y[0,answers[0]]-y[0,answers[1]])*(1 if e['expected']==0 else -1)>0));return stash
for i,e in enumerate(examples):
 records.append(dict(row=i,pair=e['pair'],order=e['order'],format=e['format'],distractor=e['distractor'],distractor_class=e['distractor_class'],**evaluate(e)))
 if i%100==0:print('evaluated',i,flush=True)
summary=[];perword=[]
for fmt in templates:
 for cls,words in classes.items():
  rr=[r for r in records if r['format']==fmt and r['distractor_class']==cls];diff=[r['coefficient']-next(b['coefficient'] for b in records if b['format']=='baseline' and b['pair']==r['pair'] and b['order']==r['order']) for r in rr];both=[all(r['correct_binary'] for r in rr if r['pair']==p and r['distractor']==w) for p in sorted({r['pair'] for r in rr}) for w in words]
  for word in words:
   ww=[r for r in rr if r['distractor']==word];perword.append(dict(format=fmt,cls=cls,word=word,mean_coefficient=float(np.mean([r['coefficient'] for r in ww]))))
  means=[w['mean_coefficient'] for w in perword if w['format']==fmt and w['cls']==cls];summary.append(dict(format=fmt,cls=cls,mean_coefficient=float(np.mean([r['coefficient'] for r in rr])),mean_shift=float(np.mean(diff)),word_mean_min=min(means),word_mean_max=max(means),both_endpoint_accuracy=float(np.mean(both)),top1=float(np.mean([r['correct_top1'] for r in rr]))))
h1025=json.loads((A/'bilin18_routing_factor_diagnosis.json').read_text());control=max(abs(r['coefficient']-next(v['corners'][0] for v in h1025['records'] if v['format']=='baseline' and v['pair']==r['pair'] and v['order']==r['order'])) for r in records if r['format']=='baseline')
audit={'baseline_coefficient_replay_max_abs':control,'finite':bool(np.isfinite([r['coefficient'] for r in records]).all()),'prefix_sequences':392,'suffix_sequences':392,'unique_prompts':392,'fit_rows':0,'reserved_test_examples':0,'positions_matched_within_wording':True};assert control<1e-6 and audit['finite'];(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,perword=perword,records=records,audits=audit),indent=2)+'\n');print(json.dumps(dict(summary=summary,audits=audit),indent=2))
