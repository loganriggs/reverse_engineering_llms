"""H1018 third-task causal anchor: language lookup with output-vocabulary holdout."""
from pathlib import Path
import sys,json,numpy as np,torch,hashlib
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_language_functional_subspace';out=A/'language_functional_subspace';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
manifest=json.loads((A/'bilin18_multidomain_matched_competence_manifest.json').read_text());examples=[e for e in manifest['examples'] if e['domain']=='language' and e['family']=='task'];pairids=sorted({e['pair'] for e in examples});training_pairs=set(pairids[:4]);meta=[]
for e in examples:meta.append(dict(e,cohort='discovery' if e['pair'] in training_pairs and e['format']=='A' else 'template_transfer' if e['pair'] in training_pairs else 'lexical_'+e['format']))
train=[i for i,e in enumerate(meta) if e['cohort']=='discovery'];assert len(train)==8
train_answers={t for i in train for g in meta[i]['groups'] for t in g};novel_answers={t for e in meta if e['cohort'].startswith('lexical') for g in e['groups'] for t in g};assert not train_answers&novel_answers
config={'experiment':'H1018','task':'Country to language lookup','training':'First4historical language pairs, formatAonly:8states; test familiar pairs formatB and last4pairs in bothformats','selection':'Task selected from prior competence report; fixed first4split, no current outcome filtering. Novel pairs have disjoint source countries and answer languages. Historical evaluation screen was already opened.','bases':'language4/8discovery gradients; previous number4,sentiment4,joint4,union8; newbalanced3taskjoint4/6; random4. This turn tests language only, so no preservation claim for new joint spaces on prior tasks.','gate':'Per cohort both-endpoint binary accuracy>=.75 and fullvocabulary intended top1>=.5; allresults retained','scope':'Finite last-position preL11 donor-state causal projection. No trained BSF, inference suffix replacement, or new unobserved lexical dataset.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'responses.npz').exists();torch.set_num_threads(6)
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
F=torch.nn.functional

def capture(ids):
 ids=torch.tensor([ids],device='cuda')
 with torch.no_grad():initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float()
 raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);return raw,x0,bus

def suffix(raw,x0,bus):
 x=raw
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,bus=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30)
cache=[capture(e['ids']) for e in meta];native=[];gradients=[];native_replay=0
for i,e in enumerate(meta):
 raw,x0,bus=cache[i];ans=[g[0] for g in e['groups']]
 if i in train:
  xx=raw.detach().requires_grad_(True);log=suffix(xx,x0,bus);margin=log[0,ans[0]]-log[0,ans[1]];gradients.append(torch.autograd.grad(margin,xx)[0][0,-1].detach().cpu().double().numpy())
 else:
  with torch.no_grad():log=suffix(raw,x0,bus)
 native.append(log.detach().cpu().double().numpy()[0])
 if i==0:
  check=runner.component.native_margin(model,torch.tensor([e['ids']],device='cuda'),torch.tensor([len(e['ids'])-1],device='cuda'),torch.tensor([ans],device='cuda'),torch,F);native_replay=abs(float(check)-(native[0][ans[0]]-native[0][ans[1]]))
g=np.stack(gradients);_,_,vl=np.linalg.svd(g,full_matrices=False);old=np.load(A/'two_task_functional_subspaces/bases.npz');ng=np.load(A/'recipient_gradient_diagnosis/gradients.npz')['gradients'][0,:,0];sg=old['sentiment_gradients'];_,_,vj=np.linalg.svd(np.concatenate([x/np.linalg.norm(x) for x in [ng,sg,g]]),full_matrices=False);rng=np.random.default_rng(1018);random,_=np.linalg.qr(rng.normal(size=(1152,4)))
bases={k:old[k] for k in ['number4','sentiment4','joint4','union8']};bases.update(language4=vl[:4].T,language8=vl[:8].T,three_task_joint4=vj[:4].T,three_task_joint6=vj[:6].T,random4=random);np.savez(out/'bases.npz',**bases,language_gradients=g);frozen=hashlib.sha256((out/'bases.npz').read_bytes()).hexdigest()
records=[]
for i,e in enumerate(meta):
 donor=next(j for j,f in enumerate(meta) if f['pair']==e['pair'] and f['format']==e['format'] and f['order']!=e['order']);raw,x0,bus=cache[i];delta=cache[donor][0][0,-1].cpu().double().numpy()-raw[0,-1].cpu().double().numpy();ans=[gg[0] for gg in e['groups']];base=native[i][ans[0]]-native[i][ans[1]];don=native[donor][ans[0]]-native[donor][ans[1]]
 for arm,edit in {'full':delta,**{name:delta@b@b.T for name,b in bases.items()}}.items():
  xx=raw.clone();xx[0,-1]+=torch.tensor(edit,device='cuda',dtype=xx.dtype)
  with torch.no_grad():y=suffix(xx,x0,bus).cpu().double().numpy()[0]
  records.append(dict(row=i,pair=e['pair'],cohort=e['cohort'],arm=arm,change=float(y[ans[0]]-y[ans[1]]-base),native_donor_change=float(don-base),edit_norm=float(np.linalg.norm(edit)),full_norm=float(np.linalg.norm(delta))))
competence=[];summary=[]
for cohort in sorted({e['cohort'] for e in meta}):
 ix=[i for i,e in enumerate(meta) if e['cohort']==cohort];binary=[native[i][meta[i]['groups'][meta[i]['expected']][0]]>native[i][meta[i]['groups'][1-meta[i]['expected']][0]] for i in ix];correct={i:b for i,b in zip(ix,binary)};both=[all(correct[i] for i in ix if meta[i]['pair']==pair) for pair in sorted({meta[i]['pair'] for i in ix})];top=[native[i].argmax()==meta[i]['groups'][meta[i]['expected']][0] for i in ix];competence.append(dict(cohort=cohort,both_endpoint_accuracy=float(np.mean(both)),top1=float(np.mean(top)),pass_gate=bool(np.mean(both)>=.75 and np.mean(top)>=.5)))
 full=np.array([r['change'] for r in records if r['cohort']==cohort and r['arm']=='full']);don=np.array([r['native_donor_change'] for r in records if r['cohort']==cohort and r['arm']=='full'])
 for arm in ['full']+list(bases):
  rr=[r for r in records if r['cohort']==cohort and r['arm']==arm];y=np.array([r['change'] for r in rr]);summary.append(dict(cohort=cohort,arm=arm,n=len(rr),relative_l2=float(np.linalg.norm(y-full)/np.linalg.norm(full)),effect_rms=float(np.sqrt(np.mean(y*y))),full_patch_to_natural_donor_relative_l2=float(np.linalg.norm(full-don)/np.linalg.norm(don))))
result={'config':config,'competence':competence,'summary':summary,'records':records,'overlaps':{name:np.linalg.svd(bases['language4'].T@bases[name],compute_uv=False).tolist() for name in ['number4','sentiment4','joint4']},'audits':{'native_replay_max_abs':native_replay,'frozen_bases_unchanged':frozen==hashlib.sha256((out/'bases.npz').read_bytes()).hexdigest(),'train_test_answer_overlap':len(train_answers&novel_answers),'discovery_gradients':8,'evaluation_gradients':0,'prefix_sequences':32,'native_sequences':1,'suffix_sequences':352,'finite':all(np.isfinite(r['change']) for r in records),'new_reserved_test_examples':0}}
assert native_replay<1e-4 and result['audits']['finite'];(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');np.savez(out/'responses.npz',changes=np.array([r['change'] for r in records]));(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['competence','summary','overlaps','audits']},indent=2))
