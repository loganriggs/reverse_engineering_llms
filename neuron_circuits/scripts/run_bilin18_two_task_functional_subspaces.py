"""H1017 compare shared and separate number/sentiment read subspaces."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_two_task_functional_subspaces';out=A/'two_task_functional_subspaces';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
manifest=json.loads((A/'bilin18_factorial_valence_negation_manifest.json').read_text());sent=[];pairs=[]
for q in manifest['quartets']:
 if q['format'] not in ['A','B'] or (q['format']=='B' and q['source']<4):continue
 ii=[]
 for idx in q['indices']:
  e=manifest['examples'][idx]
  if e['n']!=1:continue
  ii.append(len(sent));sent.append(dict(e,cohort='discovery' if q['format']=='A' and q['source']<4 else 'historical_test_'+q['format'],source=q['source']))
 assert len(ii)==2;pairs.append(ii)
assert len(sent)==24
config={'experiment':'H1017','tasks':'Number agreement and plain adjective valence at same preL11residual boundary; different token positions and contexts','sentiment_training':'8plain states: first4adjective pairs, formatA; historical evaluation8formatA+8formatBstates never fit','bases':'number4 frozen; sentiment4 from8discovery gradients; balanced joint4 from each task gradient matrix normalized by Frobenius norm; union8 QR of task4bases; balanced activationPCA4; random4','gates':'Sentiment cohort binary positive/negative margin accuracy>=.75, fullvocab label-top1 reported separately. No individual-row filtering. Failed cohorts diagnostic only.','scope':'Fixed linear causal projection, not trained BSF or manifold. Joint4 tests same active dimension; union8 doubles budget. Known task selection for separate4 is supervised, not unsupervised routing.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'responses.npz').exists()
torch.set_num_threads(6);model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
F=torch.nn.functional;enc=runner.original_authority.task14.ENCODING;tokenpairs=[(' is',' are'),(' positive',' negative'),(' can',' will')];pi=torch.tensor([[enc.encode(x)[0] for x in pair] for pair in tokenpairs],device='cuda');assert all(len(enc.encode(x))==1 for pair in tokenpairs for x in pair)
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
 logits=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30);v=logits[:,pi.flatten()].reshape(-1,3,2);return v[:,:,0]-v[:,:,1],logits
cache=[capture(e['ids']) for e in sent];sg=[];native=[];top=[]
for i,(raw,x0,bus) in enumerate(cache):
 if sent[i]['cohort']=='discovery':
  xx=raw.detach().requires_grad_(True);v,log=suffix(xx,x0,bus);g=torch.autograd.grad(v[:,1].sum(),xx)[0][0,-1];sg.append(g.detach().cpu().double().numpy())
 else:
  with torch.no_grad():v,log=suffix(raw,x0,bus)
 native.append(v.detach().cpu().numpy()[0]);top.append(int(log.detach().argmax(1)))
sg=np.stack(sg);ng=np.load(A/'recipient_gradient_diagnosis/gradients.npz')['gradients'][0,:,0];number=np.load(A/'functional_projection_transfer/frozen_bases.npz')['gradient4'];_,_,vs=np.linalg.svd(sg,full_matrices=False);sentbasis=vs[:4].T;combined=np.concatenate([ng/np.linalg.norm(ng),sg/np.linalg.norm(sg)]);_,_,vj=np.linalg.svd(combined,full_matrices=False);union,_=np.linalg.qr(np.concatenate([number,sentbasis],1));rng=np.random.default_rng(1017);random,_=np.linalg.qr(rng.normal(size=(1152,4)))
nraw=np.load(A/'recipient_gradient_diagnosis/gradients.npz')['raw'];sraw=np.stack([c[0][0,-1].cpu().double().numpy() for i,c in enumerate(cache) if sent[i]['cohort']=='discovery']);nraw=nraw-nraw.mean(0);sraw=sraw-sraw.mean(0);_,_,vp=np.linalg.svd(np.concatenate([nraw/np.linalg.norm(nraw),sraw/np.linalg.norm(sraw)]),full_matrices=False)
bases={'number4':number,'sentiment4':sentbasis,'joint4':vj[:4].T,'union8':union,'activation4':vp[:4].T,'random4':random}
np.savez(out/'bases.npz',**bases,sentiment_gradients=sg);basis_hash=hashlib.sha256((out/'bases.npz').read_bytes()).hexdigest()
records=[];arrays={};delta_checks=[]
for pair in pairs:
 for i,donor in [pair,pair[::-1]]:
  raw,x0,bus=cache[i];delta=cache[donor][0][0,-1].cpu().double().numpy()-raw[0,-1].cpu().double().numpy();edits={'full':delta}|{name:delta@b@b.T for name,b in bases.items()}
  for name,e in edits.items():
   xx=raw.clone();xx[0,-1]+=torch.tensor(e,device='cuda',dtype=xx.dtype)
   with torch.no_grad():v,_=suffix(xx,x0,bus)
   change=v.cpu().numpy()[0]-native[i];records.append(dict(task='sentiment',cohort=sent[i]['cohort'],row=i,source=sent[i]['source'],arm=name,change=change.tolist(),edit_norm=float(np.linalg.norm(e)),full_edit_norm=float(np.linalg.norm(delta))))
# Number benchmark replay uses existing actual natural-token-derived edits, no evaluation gradients.
prior=np.load(A/'functional_projection_transfer/responses.npz');panel=json.loads((A/'bilin18_expanded_causal_panel.json').read_text());numids=prior['selected'][48:]
for i,idx in enumerate(numids):
 e=panel['rows'][idx];raw,x0,bus=capture(e['token_ids']);delta=prior['natural_token_delta'][48+i]
 with torch.no_grad():v0,_=suffix(raw,x0,bus)
 for name,edit in {'full':delta,**{n:delta@b@b.T for n,b in bases.items()}}.items():
  xx=raw.clone();xx[0,-1]+=torch.tensor(edit,device='cuda',dtype=xx.dtype)
  with torch.no_grad():v,_=suffix(xx,x0,bus)
  change=(v-v0).cpu().numpy()[0];records.append(dict(task='number',cohort='historical_test',row=i,source=e['subject'],arm=name,change=change.tolist(),edit_norm=float(np.linalg.norm(edit)),full_edit_norm=float(np.linalg.norm(delta))))
  if name=='full':delta_checks.append(abs(float(change[0]) + float(prior['natural_token_full'][48+i,0])*(1 if e['answer']==318 else -1)))
summary=[]
for task in ['sentiment','number']:
 for cohort in sorted({r['cohort'] for r in records if r['task']==task}):
  full=np.array([r['change'] for r in records if r['task']==task and r['cohort']==cohort and r['arm']=='full']);target=1 if task=='sentiment' else 0
  for arm in ['full']+list(bases):
   rr=[r for r in records if r['task']==task and r['cohort']==cohort and r['arm']==arm];y=np.array([r['change'] for r in rr]);summary.append(dict(task=task,cohort=cohort,arm=arm,n=len(rr),target_relative_l2=float(np.linalg.norm(y[:,target]-full[:,target])/np.linalg.norm(full[:,target])),rms=np.sqrt(np.mean(y*y,axis=0)).tolist(),mean_norm_fraction=float(np.mean([r['edit_norm']/r['full_edit_norm'] for r in rr]))))
competence=[]
for cohort in sorted({e['cohort'] for e in sent}):
 ix=[i for i,e in enumerate(sent) if e['cohort']==cohort];acc=float(np.mean([native[i][1]*sent[i]['a']>0 for i in ix]));fv=float(np.mean([top[i]==int(pi[1,0 if sent[i]['a']==1 else 1]) for i in ix]));competence.append(dict(cohort=cohort,n=len(ix),binary_accuracy=acc,label_top1=fv,pass_gate=acc>=.75))
cos=np.linalg.svd(number.T@sentbasis,compute_uv=False)
result={'config':config,'competence':competence,'summary':summary,'subspace_canonical_cosines':cos.tolist(),'records':records,'audits':{'number_full_replay_max_abs':max(delta_checks),'frozen_bases_unchanged':basis_hash==hashlib.sha256((out/'bases.npz').read_bytes()).hexdigest(),'sentiment_training_gradients':8,'evaluation_gradients':0,'finite':all(np.isfinite(r['change']).all() for r in records),'prefix_sequences':56,'suffix_sequences':448,'backward_sequences':8,'new_reserved_test_examples':0}}
assert max(delta_checks)<1e-4 and result['audits']['finite'];(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');np.savez(out/'responses.npz',changes=np.array([r['change'] for r in records]));print(json.dumps({k:result[k] for k in ['competence','summary','subspace_canonical_cosines','audits']},indent=2))
