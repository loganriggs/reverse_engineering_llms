"""H1013 discovery-only equal-norm source-direction by recipient factorial."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_source_context_factorial';out=A/'source_context_factorial';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
panel=json.loads((A/'bilin18_expanded_causal_panel.json').read_text());allrows=panel['rows'];data=np.load(A/'expanded_causal_prediction/captures.npz');chosen=[]
for pair in panel['training_pairs']:
 for word in pair:chosen.append(next(i for i,r in enumerate(allrows) if r['split']=='expanded_discovery' and r['subject']==word and r['template']=='near'))
rows=[allrows[i] for i in chosen];groups=[[i for i,r in enumerate(rows) if r['answer']==ans] for ans in [318,389]];assert list(map(len,groups))==[24,24]
delta=data['delta'][chosen].sum(1);norms=np.linalg.norm(delta,axis=1);rho=float(np.median(norms));directions=delta/norms[:,None];rng=np.random.default_rng(1013);random=rng.normal(size=(4,1152));random/=np.linalg.norm(random,axis=1,keepdims=True)
config={'experiment':'H1013','data':'48 expanded-discovery contexts:24noun pairs, both numbers, first near-template attractor; no test states','design':'Within each grammatical number, Cartesian24recipient x24source directions. Combined five-source delta normalized to one pooled discovery median norm. Four fixed random unit directions shared across recipients. Signed scales +/-0.25,+/-1. Recipient native prefix/context stays fixed.','rho':rho,'scales':[.25,-.25,1.,-1.],'random_seed':1013,'limits':'Transplanted residual edits may be off manifold. Discovery-only descriptive factorial, no learned predictor or new generalization claim. Same-number donors control effect sign; recipient context includes its own lexical state, not only attention routing.'}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'responses.npz').exists()
torch.set_num_threads(6);model,receipt=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);F=torch.nn.functional
ids=torch.tensor([r['token_ids'] for r in rows],device='cuda');pos=torch.tensor([5]*48,device='cuda');answers=torch.tensor([[r['answer'],r['foil']] for r in rows],device='cuda')
with torch.no_grad():initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float()
raw,x0,first,ports,audit=runner.component._capture(model,initial,torch,F)
base=runner.component._suffix_margin(model,raw,x0,first,pos,answers,torch,F);native=runner.component.native_margin(model,ids,pos,answers,torch,F)
rawerr=float(np.linalg.norm(raw[:,5].cpu().double().numpy()-data['raw'][chosen])/np.linalg.norm(data['raw'][chosen]));assert rawerr<1e-5
response=np.zeros((2,4,24,28));maxnorm=0
for gi,group in enumerate(groups):
 dd=np.concatenate([directions[group],random]);dd=torch.tensor(dd,device='cuda',dtype=raw.dtype)
 for ri,recipient in enumerate(group):
  # Every direction is evaluated in precisely the same recipient context.
  ix=torch.tensor([recipient]*28,device='cuda');p=pos[ix];aa=answers[ix]
  for si,scale in enumerate(config['scales']):
   edit=dd*(rho*scale);maxnorm=max(maxnorm,float((edit.double().norm(dim=1)/(rho*abs(scale))-1).abs().max()))
   xx=raw[ix].clone();xx[torch.arange(28,device='cuda'),p]+=edit
   margin=runner.component._suffix_margin(model,xx,x0[ix],first[ix],p,aa,torch,F)
   response[gi,si,ri]=base[recipient].item()-margin.cpu().numpy()
 print('group complete',gi,flush=True)
np.savez(out/'responses.npz',response=response,directions=directions,random_directions=random,chosen=chosen,groups=np.array(groups),rho=rho,base=base.cpu().numpy(),raw=raw[:,5].cpu().numpy())

def stats(m):
 grand=m.mean();row=m.mean(1,keepdims=True)-grand;col=m.mean(0,keepdims=True)-grand;interaction=m-grand-row-col;ss=np.sum((m-grand)**2);sv=np.linalg.svd(m,compute_uv=False)
 return {'mean':float(grand),'rms':float(np.sqrt(np.mean(m*m))),'row_fraction_centered_ss':float(m.shape[1]*np.sum(row*row)/ss),'column_fraction_centered_ss':float(m.shape[0]*np.sum(col*col)/ss),'interaction_fraction_centered_ss':float(np.sum(interaction**2)/ss),'rank1_raw_energy':float(sv[0]**2/np.sum(sv**2)),'diagonal_mean':float(np.diag(m).mean()),'offdiagonal_mean':float(m[~np.eye(24,dtype=bool)].mean())}
summary=[]
for gi in range(2):
 small=response[gi,0,:,:24];large=response[gi,2,:,:24];oddsmall=(response[gi,0,:,:24]-response[gi,1,:,:24])/2;oddlarge=(response[gi,2,:,:24]-response[gi,3,:,:24])/2;evensmall=(response[gi,0,:,:24]+response[gi,1,:,:24])/2;evenlarge=(response[gi,2,:,:24]+response[gi,3,:,:24])/2
 summary.append({'number':['singular','plural'][gi],'small_positive':stats(small),'large_positive':stats(large),'small_odd':stats(oddsmall),'large_odd':stats(oddlarge),'small_even_to_odd_norm':float(np.linalg.norm(evensmall)/np.linalg.norm(oddsmall)),'large_even_to_odd_norm':float(np.linalg.norm(evenlarge)/np.linalg.norm(oddlarge)),'odd_scale4_relative_error':float(np.linalg.norm(oddlarge-4*oddsmall)/np.linalg.norm(oddlarge)),'even_scale16_relative_error':float(np.linalg.norm(evenlarge-16*evensmall)/np.linalg.norm(evenlarge)),'random_large_rms':float(np.sqrt(np.mean(response[gi,2,:,24:]**2))),'source_large_rms':float(np.sqrt(np.mean(large**2)))})
result={'config':config,'summary':summary,'audits':{'native_replay_max_abs':float((base-native).abs().max()),'source_capture_relative_error':rawerr,'edit_norm_relative_error':maxnorm,'finite':bool(np.isfinite(response).all()),'discovery_only':all(r['split']=='expanded_discovery' for r in rows),'native_sequences':48,'partial_sequences':48,'suffix_sequences':48+2*4*24*28,'new_reserved_test_examples':0}}
assert result['audits']['native_replay_max_abs']<1e-5 and maxnorm<1e-6 and result['audits']['finite']
(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
