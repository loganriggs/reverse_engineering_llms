"""H1014 freeze source/recipient regressions before new factorial response collection."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_factor_response_prediction';out=A/'factor_response_prediction';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
old=np.load(A/'source_context_factorial/responses.npz');data=np.load(A/'expanded_causal_prediction/captures.npz');panel=json.loads((A/'bilin18_expanded_causal_panel.json').read_text());rows=panel['rows'];rho=float(old['rho'])
# Source/test states were historically opened; these Cartesian interventions are new.
donor_ids=[next(i for i,r in enumerate(rows) if r['split']=='new_lexical_test' and r['subject']==word and r['template']=='near') for pair in panel['test_pairs'] for word in pair]
recipient_ids=[next(i for i,r in enumerate(rows) if r['subject']==word and r['template']==template and r['split'] in ['new_lexical_test','new_joint_test']) for template in ['near','under'] for pair in panel['test_pairs'] for word in pair]
assert not set(old['chosen'])&set(donor_ids+recipient_ids)
delta=data['delta'][donor_ids].sum(1);directions=delta/np.linalg.norm(delta,axis=1,keepdims=True)
def norm_features(ids):return np.concatenate([np.linalg.norm(data['clean'][ids],axis=2),np.linalg.norm(data['raw'][ids],axis=1,keepdims=True)],1)
def fit_predict(x,y,test,mode):
 xm=x.mean(0);xx=x-xm;tt=test-xm;basis=None
 if mode=='pca4':
  _,_,vh=np.linalg.svd(xx,full_matrices=False);basis=vh[:4].T;xx=xx@basis;tt=tt@basis
 if mode in ['pca4','norms']:
  scale=xx.std(0).clip(1e-12);xx=xx/scale;tt=tt/scale
 else:scale=np.ones(xx.shape[1])
 kernel_scale=float(np.mean(np.sum(xx*xx,axis=1)));xx=xx/np.sqrt(kernel_scale);tt=tt/np.sqrt(kernel_scale);ym=float(y.mean());alpha=np.linalg.solve(xx@xx.T+.1*np.eye(len(xx)),y-ym);weights=xx.T@alpha
 return ym+tt@weights,ym+xx@weights,dict(xmean=xm,scale=scale,basis=basis,kernel_scale=kernel_scale,ymean=ym,weights=weights)
config={'experiment':'H1014','target':'Full positive fixed rho damage; same-number donors','training':'H1013 24x24 discovery response matrix per grammatical number; rank1SVD factors, independent regressions on native upstream states','models':'constant; source linear with recipient mean; source linear x recipient norms6; source linear x recipientPCA4; sourcePCA4 x recipientPCA4; centered kernel ridge0.1 after trace normalization, fixed','evaluation':'8new donor nouns x16recipients(8near,8under) per number. Noun states used in H1012evaluation; cross interventions newly measured. No suffix inputs to students.','scope':'One causal channel from one rank1embedding intervention. No BSF/manifold or task-general claim.','rho':rho}
(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');assert not (out/'responses.npz').exists()
predictions={name:[] for name in ['constant','source_only','source_linear_recipient_norms','source_linear_recipient_pca4','source_pca4_recipient_pca4']};models=[];indices=[];trainerrors=[]
for gi,answer in enumerate([318,389]):
 group=old['groups'][gi];trainids=old['chosen'][group];dix=[i for i,j in enumerate(donor_ids) if rows[j]['answer']==answer];rix=[i for i,j in enumerate(recipient_ids) if rows[j]['answer']==answer];indices.append((dix,rix));m=old['response'][gi,2,:,:24];u,s,v=np.linalg.svd(m,full_matrices=False);gain=u[:,0]/u[:,0].mean();potency=s[0]*v[0]*u[:,0].mean()
 sp,st,sm=fit_predict(old['directions'][group],potency,directions[dix],'linear');sq,sqt,sqm=fit_predict(old['directions'][group],potency,directions[dix],'pca4');rn,rnt,rnm=fit_predict(norm_features(trainids),gain,norm_features(np.array(recipient_ids)[rix]),'norms');rp,rpt,rpm=fit_predict(data['raw'][trainids],gain,data['raw'][np.array(recipient_ids)[rix]],'pca4')
 predictions['constant'].append(np.full((16,8),m.mean()));predictions['source_only'].append(np.tile(sp,(16,1)));predictions['source_linear_recipient_norms'].append(rn[:,None]*sp[None,:]);predictions['source_linear_recipient_pca4'].append(rp[:,None]*sp[None,:]);predictions['source_pca4_recipient_pca4'].append(rp[:,None]*sq[None,:]);models.append(dict(source=sm,source_pca=sqm,recipient_norm=rnm,recipient_pca=rpm));trainerrors.append({'source_only':float(np.linalg.norm(np.tile(st,(24,1))-m)/np.linalg.norm(m)),'source_linear_recipient_norms':float(np.linalg.norm(rnt[:,None]*st-m)/np.linalg.norm(m)),'source_linear_recipient_pca4':float(np.linalg.norm(rpt[:,None]*st-m)/np.linalg.norm(m))})
# Persist frozen model predictions BEFORE evaluation model loading or suffix calls.
for k in predictions:predictions[k]=np.stack(predictions[k])
np.savez(out/'frozen_predictions.npz',**predictions);torch.save(models,out/'models.pt');frozen_hash=hashlib.sha256((out/'frozen_predictions.npz').read_bytes()).hexdigest()
torch.set_num_threads(6);model,receipt=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True);F=torch.nn.functional
rr=[rows[i] for i in recipient_ids];ids=torch.tensor([r['token_ids'] for r in rr],device='cuda');pos=torch.tensor([5]*len(rr),device='cuda');answers=torch.tensor([[r['answer'],r['foil']] for r in rr],device='cuda')
with torch.no_grad():initial=F.rms_norm(model.transformer.wte(ids),(1152,)).float()
raw,x0,first,_,_=runner.component._capture(model,initial,torch,F);base=runner.component._suffix_margin(model,raw,x0,first,pos,answers,torch,F);native=runner.component.native_margin(model,ids,pos,answers,torch,F)
response=np.empty((2,16,8));randomresponse=np.empty((2,16,4));random=old['random_directions']
for gi,(dix,rix) in enumerate(indices):
 dd=torch.tensor(np.concatenate([directions[dix],random]),device='cuda',dtype=raw.dtype)*rho
 for ri,recipient in enumerate(rix):
  ix=torch.tensor([recipient]*12,device='cuda');xx=raw[ix].clone();xx[:,5]+=dd;margin=runner.component._suffix_margin(model,xx,x0[ix],first[ix],pos[ix],answers[ix],torch,F);damage=base[recipient].item()-margin.cpu().numpy();response[gi,ri]=damage[:8];randomresponse[gi,ri]=damage[8:]
np.savez(out/'responses.npz',response=response,random_response=randomresponse,donor_ids=donor_ids,recipient_ids=recipient_ids,directions=directions)
summary=[]
for gi in range(2):
 for template,ix in [('near',np.arange(8)),('under',np.arange(8,16)),('all',np.arange(16))]:
  y=response[gi,ix]
  for name,p in predictions.items():summary.append(dict(number=['singular','plural'][gi],template=template,model=name,relative_l2=float(np.linalg.norm(p[gi,ix]-y)/np.linalg.norm(y)),bias=float(np.mean(p[gi,ix]-y))))
oracles=[]
for gi in range(2):
 u,s,v=np.linalg.svd(response[gi],full_matrices=False);oracles.append({'number':['singular','plural'][gi],'rank1_relative_l2':float(np.linalg.norm(response[gi]-s[0]*u[:,:1]@v[:1])/np.linalg.norm(response[gi])),'source_random_rms_ratio':float(np.sqrt(np.mean(response[gi]**2)/np.mean(randomresponse[gi]**2)))})
result={'config':config,'summary':summary,'train_errors':trainerrors,'evaluation_oracle_diagnostics':oracles,'audits':{'native_replay_max_abs':float((base-native).abs().max()),'frozen_predictions_unchanged':frozen_hash==hashlib.sha256((out/'frozen_predictions.npz').read_bytes()).hexdigest(),'finite':bool(np.isfinite(response).all() and all(np.isfinite(p).all() for p in predictions.values())),'student_suffix_inputs':0,'native_sequences':32,'prefix_sequences':32,'suffix_sequences':32+2*16*12,'new_reserved_test_examples':0}}
assert result['audits']['native_replay_max_abs']<1e-5 and result['audits']['frozen_predictions_unchanged'] and result['audits']['finite']
(A/f'{S}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
