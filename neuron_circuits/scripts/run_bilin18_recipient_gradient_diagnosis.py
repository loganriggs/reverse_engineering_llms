"""H1015 discovery-only native sensitivity and final-normalization diagnostic."""
from pathlib import Path
import sys,json,numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_recipient_gradient_diagnosis';out=A/'recipient_gradient_diagnosis';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
old=np.load(A/'source_context_factorial/responses.npz');panel=json.loads((A/'bilin18_expanded_causal_panel.json').read_text());rows=[panel['rows'][i] for i in old['chosen']];enc=runner.original_authority.task14.ENCODING
names=['is_are','was_were','can_will','period_comma'];pairs=[(' is',' are'),(' was',' were'),(' can',' will'),('.',',')];pairids=[]
for pair in pairs:
 tok=[enc.encode(w) for w in pair];assert all(len(t)==1 for t in tok);pairids.append([t[0] for t in tok])
config={'experiment':'H1015','rows':'H1013 48discovery recipients only','diagnostics':'Native gradient of4logit contrasts with respect to subject-position preL11residual. Compare related number and unrelated contrasts. Stop gradient only through final RMS denominator in separate arm, preserving clean forward. Finite directional +/-0.001rho check.','scope':'Native suffix and backwards used for mechanism diagnosis, never student inference. Final RMS-only ablation does not isolate all normalization or attention transport.','contrasts':names}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'gradients.npz').exists();torch.set_num_threads(6)
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
for p in model.parameters():p.requires_grad_(False)
F=torch.nn.functional;ids=torch.tensor([r['token_ids'] for r in rows],device='cuda');pos=torch.tensor([5]*48,device='cuda');answer=torch.tensor([[r['answer'],r['foil']] for r in rows],device='cuda');pi=torch.tensor(pairids,device='cuda')
with torch.no_grad():initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float()
raw,x0,first,_,_=runner.component._capture(model,initial,torch,F)
def suffix(xx,freeze=False):
 x=xx;bus=first
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,bus=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 if freeze:
  denom=(x.square().mean(-1,keepdim=True)+torch.finfo(x.dtype).eps).sqrt().detach();norm=x/denom
 else:norm=F.rms_norm(x,(1152,))
 # Compute only required vocabulary coordinates, exactly as native linear readout.
 logits=30*torch.tanh(model.lm_head(norm[:,5])[:,pi.flatten()]/30);logits=logits.reshape(48,4,2)
 return logits[:,:,0]-logits[:,:,1],x[:,5]
grads=[];values=[];finalstates=None
for freeze in [False,True]:
 xx=raw.detach().requires_grad_(True);v,final=suffix(xx,freeze);gg=[]
 for c in range(4):gg.append(torch.autograd.grad(v[:,c].sum(),xx,retain_graph=c<3)[0][:,5].detach().cpu().double().numpy())
 grads.append(np.stack(gg,1));values.append(v.detach().cpu().numpy());finalstates=final.detach().cpu().numpy()
grad=np.stack(grads);sign=np.array([1 if r['answer']==318 else -1 for r in rows]);rho=float(old['rho']);epsilon=.001*rho
with torch.no_grad():
 dd=torch.tensor(old['directions'],device='cuda',dtype=raw.dtype);xp=raw.clone();xm=raw.clone();xp[:,5]+=epsilon*dd;xm[:,5]-=epsilon*dd;vp,_=suffix(xp);vm,_=suffix(xm);fd=((vp-vm)/(2*epsilon)).cpu().numpy()
 analytic=np.einsum('ncd,nd->nc',grad[0],old['directions']);fd_error=float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic))
 native=runner.component.native_margin(model,ids,pos,answer,torch,F).cpu().numpy()
np.savez(out/'gradients.npz',gradients=grad,values=np.stack(values),finite_difference=fd,analytic_directional=analytic,final_states=finalstates,raw=raw[:,5].cpu().numpy(),sign=sign)
summary=[]
for gi,group in enumerate(old['groups']):
 d=old['directions'][group];g=grad[0,group]*sign[group,None,None];gf=grad[1,group]*sign[group,None,None];response=-rho*.25*np.einsum('rcd,sd->rcs',g,d);odd=(old['response'][gi,0,:,:24]-old['response'][gi,1,:,:24])/2
 gg=g[:,0];unit=gg/np.linalg.norm(gg,axis=1,keepdims=True);gram=unit@unit.T;sv=np.linalg.svd(gg,compute_uv=False)
 row={'number':['singular','plural'][gi],'gradient_rank1_uncentered_energy':float(sv[0]**2/np.sum(sv*sv)),'gradient_pair_cosine_median':float(np.median(gram[~np.eye(24,dtype=bool)])),'quarter_odd_linear_prediction_error':float(np.linalg.norm(response[:,0]-odd)/np.linalg.norm(odd)),'contrast_projected_response_rms':{name:float(np.sqrt(np.mean(response[:,i]**2))) for i,name in enumerate(names)},'final_norm_gradient_relative_contribution':float(np.linalg.norm(g-gf)/np.linalg.norm(g))}
 m=old['response'][gi,2,:,:24];u,s,v=np.linalg.svd(m,full_matrices=False);gain=u[:,0]/u[:,0].mean();rawnorm=np.linalg.norm(raw[group,5].cpu().numpy(),axis=1);finalnorm=np.linalg.norm(finalstates[group],axis=1);row['gain_correlation']={name:float(np.corrcoef(gain,feature)[0,1]) for name,feature in [('inverse_input_norm',1/rawnorm),('inverse_final_norm',1/finalnorm),('gradient_norm',np.linalg.norm(gg,axis=1)),('mean_direction_sensitivity',response[:,0].mean(1))]};summary.append(row)
result={'config':config,'summary':summary,'audits':{'native_margin_max_abs':float(np.max(np.abs(native-values[0][:,0]*sign))),'frozen_final_norm_forward_max_abs':float(np.max(np.abs(values[0]-values[1]))),'finite_difference_relative_error':fd_error,'finite':bool(np.isfinite(grad).all()),'discovery_only':all(r['split']=='expanded_discovery' for r in rows),'prefix_sequences':48,'native_sequences':48,'suffix_sequences':192,'backward_sequence_contrasts':384,'student_training_runs':0}}
assert result['audits']['native_margin_max_abs']<1e-4 and result['audits']['frozen_final_norm_forward_max_abs']<1e-4 and fd_error<.02
(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
