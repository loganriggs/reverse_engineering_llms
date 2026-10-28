"""H1016 direct causal projection test; independent natural-token intervention family."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_functional_projection_transfer';out=A/'functional_projection_transfer';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
old=np.load(A/'source_context_factorial/responses.npz');grad=np.load(A/'recipient_gradient_diagnosis/gradients.npz');data=np.load(A/'expanded_causal_prediction/captures.npz');panel=json.loads((A/'bilin18_expanded_causal_panel.json').read_text());rows=panel['rows'];test=np.load(A/'factor_response_prediction/responses.npz');selected=list(old['chosen'])+list(test['recipient_ids']);rr=[rows[i] for i in selected];rho=float(old['rho']);enc=runner.original_authority.task14.ENCODING
_,_,vg=np.linalg.svd(grad['gradients'][0,:,0],full_matrices=False);_,_,vs=np.linalg.svd(old['directions'],full_matrices=False);rng=np.random.default_rng(1016);qr,_=np.linalg.qr(rng.normal(size=(1152,4)))
bases={'gradient1':vg[:1].T,'gradient4':vg[:4].T,'source1':vs[:1].T,'source4':vs[:4].T,'random1':qr[:,:1],'random4':qr}
np.savez(out/'frozen_bases.npz',**bases);basis_hash=hashlib.sha256((out/'frozen_bases.npz').read_bytes()).hexdigest()
config={'experiment':'H1016','training':'Is/are gradients from48discovery recipients only; top1/4 right singular vectors, no evaluation gradients','controls':'Uncentered discovery-source direction PCA1/4 and seeded random1/4; projection coordinates counted equally, projected norms not forcibly matched','interventions':'Original five-source rank1-decoder deltas and independent natural opposite-number subject-token replacement preL11residual deltas. Both normalized to fixedrho. Subject-position patch only; all other recipient states fixed.','evaluation':'48discovery and32historical H1012/H1014test recipient states; new projected interventions, not untouched lexical data','outputs':['is_are','was_were','can_will','period_comma'],'limits':'Tests preservation of finite scalar output changes, not prediction of gain or full-vector reconstruction. Natural-token family still probes grammatical number, not independent task transfer.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'responses.npz').exists()
torch.set_num_threads(6);model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);F=torch.nn.functional
ids=torch.tensor([r['token_ids'] for r in rr],device='cuda');pos=torch.tensor([5]*80,device='cuda');answers=torch.tensor([[r['answer'],r['foil']] for r in rr],device='cuda');opposites={a:b for pair in panel['training_pairs']+panel['test_pairs'] for a,b in [pair,pair[::-1]]};other=ids.clone();other[:,5]=torch.tensor([enc.encode(' '+opposites[r['subject']])[0] for r in rr],device='cuda')
with torch.no_grad():
 initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float();initial_other=F.rms_norm(model.transformer.wte(other),(1152,)).float()
raw,x0,first,_,_=runner.component._capture(model,initial,torch,F);raw_other,_,_,_,_=runner.component._capture(model,initial_other,torch,F)
pi=torch.tensor([[enc.encode(a)[0],enc.encode(b)[0]] for a,b in [(' is',' are'),(' was',' were'),(' can',' will'),('.',',')]],device='cuda')
@torch.no_grad()
def suffix(xx):
 x=xx;bus=first
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,bus=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 logits=30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,5])[:,pi.flatten()]/30);v=logits.reshape(80,4,2)
 return (v[:,:,0]-v[:,:,1]).cpu().double().numpy()
base=suffix(raw);native=runner.component.native_margin(model,ids,pos,answers,torch,F).cpu().numpy();sign=np.array([1 if r['answer']==318 else -1 for r in rr]);deltas={'decoder':data['delta'][selected].sum(1),'natural_token':(raw_other[:,5]-raw[:,5]).cpu().double().numpy()};responses={};normfractions={};saved={};maxortho=0
for family,d in deltas.items():
 original_norm=np.linalg.norm(d,axis=1);d=d/original_norm[:,None]*rho;saved[family+'_delta']=d;saved[family+'_original_norm']=original_norm;edits={'full':d}
 for name,b in bases.items():
  edits[name]=d@b@b.T;maxortho=max(maxortho,float(np.max(np.abs(b.T@b-np.eye(b.shape[1])))))
 for name in ['gradient1','gradient4']:edits[name+'_remainder']=d-edits[name]
 for name,e in edits.items():
  x=raw.clone();x[:,5]+=torch.tensor(e,device='cuda',dtype=x.dtype);responses[family+'_'+name]=(base-suffix(x))*sign[:,None];normfractions[family+'_'+name]=np.linalg.norm(e,axis=1)/rho
 print('completed',family,flush=True)
np.savez(out/'responses.npz',**responses,**saved,selected=selected,base=base)
summary=[]
for family in deltas:
 for split,ix in [('discovery',np.arange(48)),('historical_test',np.arange(48,80))]:
  full=responses[family+'_full'][ix]
  for name in edits:
   y=responses[family+'_'+name][ix];summary.append({'family':family,'split':split,'arm':name,'relative_l2_by_contrast':(np.linalg.norm(y-full,axis=0)/np.linalg.norm(full,axis=0)).tolist(),'rms_by_contrast':np.sqrt(np.mean(y*y,axis=0)).tolist(),'mean_edit_norm_fraction':float(np.mean(normfractions[family+'_'+name][ix]))})
result={'config':config,'summary':summary,'audits':{'native_margin_max_abs':float(np.max(np.abs(base[:,0]*sign-native))),'basis_orthogonality_max_abs':maxortho,'bases_hash_unchanged':basis_hash==hashlib.sha256((out/'frozen_bases.npz').read_bytes()).hexdigest(),'finite':all(np.isfinite(v).all() for v in responses.values()),'evaluation_gradients':0,'prefix_sequences':160,'native_sequences':80,'suffix_sequences':80*(1+2*9),'new_reserved_test_examples':0}}
assert result['audits']['native_margin_max_abs']<1e-4 and maxortho<1e-10 and result['audits']['finite'];(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
