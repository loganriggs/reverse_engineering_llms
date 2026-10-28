"""H1024 frozen routing challenge with distance and competing country mentions."""
from pathlib import Path
import sys,json,hashlib,importlib.util,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_lexical_routing_challenge';out=A/'lexical_routing_challenge';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
enc=runner.original_authority.task14.ENCODING;oldmeta=json.loads((A/'language_boundary_factorial/metadata.json').read_text());base=[e for e in oldmeta if e['format']=='A'];pairs=sorted({e['pair'] for e in base});examples=[]
templates={'baseline':'The main language spoken in {country} is','distance':'The main language spoken in {country} by most people is','long_distance':'The main language spoken in {country}, according to the guide, is','competitor_before':'Unlike {distractor}, the main language spoken in {country} is','competitor_after':'The main language spoken in {country}, rather than {distractor}, is'}
for fmt,template in templates.items():
 for p in pairs:
  endpoint=[e for e in base if e['pair']==p];distractor=next(e for e in base if e['pair']==pairs[(pairs.index(p)+1)%len(pairs)] and e['order']==0)['source_words'][0].strip();made=[]
  for e in endpoint:
   country=e['source_words'][0].strip();text=template.format(country=country,distractor=distractor);ids=enc.encode(text);made.append(dict(pair=p,order=e['order'],format=fmt,text=text,ids=ids,groups=e['groups'],country=country,distractor=distractor,expected=e['expected']))
  changed=[k for k,(a,b) in enumerate(zip(made[0]['ids'],made[1]['ids'])) if a!=b];assert len(changed)==1 and len(made[0]['ids'])==len(made[1]['ids'])
  for e in made:e['source']=changed[0];examples.append(e)
assert len(examples)==80 and len({e['text'] for e in examples})==80
config={'experiment':'H1024','templates':templates,'data':'8historical country pairs x2endpoints x5templates;baseline16repeatcontrols,newcontexts64;vocabulary familiar, no output-based filtering','arms':['native_cached13','exact_live_routing','export_literal_offsets','fixed_source_and_final','live_final_only'],'frozen':'H1023package hashes/weights/routing unchanged. Literal armwrites atsource/source+1; adapterwrites source/final using unchanged coefficients. Oracle live-routing diagnostic is not deployable.','gate':'performat both-endpoint accuracy>=.75 and intended fullvocabtop1>=.5;failedformatsreported diagnostic','coverage':'Compare isolatedpath with full donor-model change, not only ownrelativeerror.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'panel.json').write_text(json.dumps(examples,indent=2)+'\n');assert not (out/'responses.npz').exists()
package=A/'head13_lexical_program';manifest=json.loads((package/'manifest.json').read_text());hashes={k:hashlib.sha256((package/k).read_bytes()).hexdigest() for k in manifest['sha256']};assert hashes==manifest['sha256'];spec=importlib.util.spec_from_file_location('frozen_program',package/'execute.py');program=importlib.util.module_from_spec(spec);spec.loader.exec_module(program);weights={k:v.cuda() for k,v in torch.load(package/'weights.pt',weights_only=True).items()}
torch.set_num_threads(6);model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);F=torch.nn.functional;cache=[];native=[]
@torch.no_grad()
def suffix(raw,x0,bus,ans,edited=None,write=None):
 x=raw
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),edited if l==13 and edited is not None else bus)
  if l==13 and write is not None:at=at+write[None].to(at)
  x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 y=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);return float(y[0,ans[0]]-y[0,ans[1]]),int(y.argmax(1))
for e in examples:
 ids=torch.tensor([e['ids']],device='cuda')
 with torch.no_grad():initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float()
 raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);cache.append((raw,x0,bus));native.append(suffix(raw,x0,bus,[g[0] for g in e['groups']]))
records=[];coefficients=[];maxwrite=maxeffect=maxcache=0
for i,e in enumerate(examples):
 donor=next(j for j,f in enumerate(examples) if f['pair']==e['pair'] and f['format']==e['format'] and f['order']!=e['order']);raw,x0,bus=cache[i];donbus=cache[donor][2];source=e['source'];edited=bus.clone();edited[0,source,0]=donbus[0,source,0];att=model.transformer.h[13].attn;original=att.squared_attention;stash={}
 def capture(q,k,v,q2,k2):
  coef=(q[0,:,0].double()@k[0,source,0].double()/128)*(q2[0,:,0].double()@k2[0,source,0].double()/128);coef[:source]=0;stash['coeff']=coef;return original(q,k,v,q2,k2)
 att.squared_attention=capture
 try:ref,_=suffix(raw,x0,bus,[g[0] for g in e['groups']],edited=edited)
 finally:att.squared_attention=original
 with torch.no_grad():
  emb=model.transformer.wte(torch.tensor([e['ids'][source]],device='cuda'));de=model.transformer.wte(torch.tensor([examples[donor]['ids'][source]],device='cuda'));fixed=program.delta_writes(weights,emb,de)[0];dv=program.cache_value(weights,de)-program.cache_value(weights,emb);payload=F.linear(dv,weights['output13_head0'])[0]*weights['cache_mix13'];expected=donbus[0,source,0]-bus[0,source,0];maxcache=max(maxcache,float((dv[0]-expected).norm()/expected.norm()))
 live=stash['coeff'][:,None]*payload.double()[None,:];literal=torch.zeros_like(raw[0]);literal[source:source+2]=fixed;adapter=torch.zeros_like(raw[0]);adapter[source]=fixed[0];adapter[-1]=fixed[1];last=torch.zeros_like(live);last[-1]=live[-1];base=native[i][0];natural=native[donor][0]-base
 records.append(dict(row=i,format=e['format'],pair=e['pair'],arm='native_cached13',change=ref-base,natural_change=natural))
 for name,write in [('exact_live_routing',live),('export_literal_offsets',literal),('fixed_source_and_final',adapter),('live_final_only',last)]:
  val,_=suffix(raw,x0,bus,[g[0] for g in e['groups']],write=write);records.append(dict(row=i,format=e['format'],pair=e['pair'],arm=name,change=val-base,natural_change=natural))
  if name=='exact_live_routing':maxeffect=max(maxeffect,abs(val-ref))
 coefficients.append(dict(row=i,source_distance=len(e['ids'])-1-source,source_coeff=float(stash['coeff'][source]),final_coeff=float(stash['coeff'][-1])))
summary=[];competence=[]
for fmt in templates:
 ix=[i for i,e in enumerate(examples) if e['format']==fmt];both=[all(native[i][0]*(1 if examples[i]['expected']==0 else -1)>0 for i in ix if examples[i]['pair']==p) for p in pairs];top=[native[i][1]==examples[i]['groups'][examples[i]['expected']][0] for i in ix];competence.append(dict(format=fmt,both_endpoint_accuracy=float(np.mean(both)),top1=float(np.mean(top)),pass_gate=bool(np.mean(both)>=.75 and np.mean(top)>=.5)))
 rr=[r for r in records if r['format']==fmt];ref=np.array([r['change'] for r in rr if r['arm']=='native_cached13']);nat=np.array([r['natural_change'] for r in rr if r['arm']=='native_cached13'])
 for arm in config['arms']:
  y=np.array([r['change'] for r in rr if r['arm']==arm]);summary.append(dict(format=fmt,arm=arm,relative_l2=float(np.linalg.norm(y-ref)/np.linalg.norm(ref)),effect_rms=float(np.sqrt(np.mean(y*y))),native_path_natural_aligned_recovery=float(ref@nat/(nat@nat))))
audit={'export_hashes_unchanged':all(hashlib.sha256((package/k).read_bytes()).hexdigest()==h for k,h in hashes.items()),'cache_value_delta_relative_error_max':maxcache,'live_routing_effect_replay_max_abs':maxeffect,'finite':all(np.isfinite(r['change']) for r in records),'prefix_sequences':80,'suffix_sequences':480,'new_context_examples':64,'new_reserved_test_examples':0,'routing_fit_rows':0}
assert maxeffect<1e-4 and maxcache<1e-5 and audit['export_hashes_unchanged'];(A/f'{S}.json').write_text(json.dumps({'config':config,'competence':competence,'summary':summary,'records':records,'coefficients':coefficients,'audits':audit},indent=2)+'\n');np.savez(out/'responses.npz',changes=np.array([r['change'] for r in records]));print(json.dumps({'competence':competence,'summary':summary,'audits':audit},indent=2))
