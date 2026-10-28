"""H1032 frozen support least-squares and decoder-span diagnosis, train-only PCA."""
from pathlib import Path
import sys,json,hashlib,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_multistream_span_diagnosis';out=A/'multistream_span_diagnosis';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import bilin18_observed_model_facade as facade
from feature_structures.models import SharedGroups
torch.set_num_threads(6);F=torch.nn.functional
prev=json.loads((A/'bilin18_multistream_sparse_baselines.json').read_text());train=torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False);tp=json.loads((A/'multitask_boundary_contract/panel.json').read_text());devdata=torch.load(A/'multistream_sparse_baselines/development_states.pt',weights_only=False);dp=json.loads((A/'multistream_sparse_baselines/development_panel.json').read_text());dev={i:tuple(v.cuda() for v in s) for i,s in devdata['states'].items()};native={i:v.cuda() for i,v in devdata['native'].items()};families=sorted({p['family'] for p in tp});files=[A/f"multistream_sparse_baselines/{f['kind']}_{f['seed']}.pt" for f in prev['fits']];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};config={'experiment':'H1032','frozen_hashes':hashes,'methods':['PCA16 per stream,48totalcoordinates','PCA128 per stream,384totalcoordinates capacity reference','H1031 same active support with exact least-squares coefficients','H1031 full128atom decoder span,384totalcoordinates capacity diagnostic'],'selection':'PCA fit on H1030 discovery only with identical family/prompt weights; no development optimization. Full decoder SVD cutoff1e-8relative. Active least squares uses same input state and frozen selected support; not downstream-informed oracle.','metrics':'Same H1031 whole-state and paired-delta native-output KL; normalized state reconstruction; per-family results. Full span is not budget matched.','stop':'Do not rank-sweep. Determine whether encoder coordinates, learned span or low-dimensional input coverage dominates failure before proposing causal supervision.'};(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n')
# Exact same token weights and centers as H1031.
X=[[],[],[]];w=[];counts={f:sum(p['family']==f for p in tp) for f in families}
for p in tp:
 for i in p['indices']:
  st=train['states'][i];T=st[0].shape[1]
  for k in range(3):X[k].append(st[k].reshape(T,1152))
  w.extend([1/(len(families)*counts[p['family']]*2*T)]*T)
X=torch.stack([torch.cat(x) for x in X]).double().cuda();w=torch.tensor(w,device='cuda',dtype=torch.float64);reference=torch.load(files[0],weights_only=False);center=reference['center'].double().cuda();scale=reference['scale'].double().cuda();Z=(X-center)/scale;pca=[]
for k in range(3):u,s,vh=torch.linalg.svd(Z[k]*w.sqrt()[:,None],full_matrices=False);pca.append(vh[:128].T)
torch.save(dict(bases=[b.cpu() for b in pca],center=center.cpu(),scale=scale.cpu()),out/'pca.pt');candidates={};reconstruction=[];max_ls_violation=0;encoded_replay=0
for rank in [16,128]:
 name='PCA'+str(rank);pred={}
 for i,st in dev.items():
  parts=[];errs=[]
  for k in range(3):
   raw=st[k].reshape(-1,1152).double();z=(raw-center[k])/scale[k];B=pca[k][:,:rank];y=z@B@B.T;parts.append((y*scale[k]+center[k]).float().reshape_as(st[k]));errs.append(float((y-z).square().sum()/z.square().sum()))
  pred[i]=tuple(parts);reconstruction.append(dict(arm=name,row=i,errors=errs))
 candidates[name]=pred
with torch.no_grad():
 for path in files:
  state=torch.load(path,weights_only=False);kind=state['fit']['kind'];seed=state['fit']['seed'];mods=nn.ModuleList([SharedGroups(1152,128,32,4 if kind=='block' else 2,4 if kind=='block' else 8) for _ in range(3)]).cuda();mods.load_state_dict(state['models']);mu=state['center'].double().cuda();sc=state['scale'].double().cuda();M=[v.cuda() for v in state['memberships']];spans=[]
  for m in mods:
   u,s,vh=torch.linalg.svd(m.dictionary_matrix().double(),full_matrices=False);spans.append(u[:,s>s[0]*1e-8])
  for mode in ['active_ls','full_span']:
   name=f'{kind}_{seed}_{mode}';pred={}
   for i,st in dev.items():
    parts=[];errs=[]
    for k,m in enumerate(mods):
     z=(st[k].reshape(-1,1152).double()-mu[k])/sc[k];D=m.dictionary_matrix().double();bias=m.bias.double();target=z-bias
     if mode=='full_span':B=spans[k];y=target@B@B.T+bias
     else:
      # Match H1031 float32 gate arithmetic exactly; solve coefficients in float64.
      z32=(st[k].reshape(-1,1152)-state['center'][k].cuda())/state['scale'][k].cuda()
      if kind=='scalar':a=m.coordinate_encoder(z32);mask=torch.zeros_like(a).scatter_(1,a.abs().topk(16,1).indices,1);enc=(a*mask)@m.dictionary_matrix().T+m.bias
      else:r=m(z32,membership=M[k]);mask=r['support'];enc=r['reconstruction']
      rows=[]
      for t in range(len(z)):
       sub=D[:,mask[t].bool()];u,s,vh=torch.linalg.svd(sub,full_matrices=False);B=u[:,s>s[0]*1e-8];rows.append(target[t]@B@B.T+bias)
      y=torch.stack(rows);max_ls_violation=max(max_ls_violation,float(((y-z).square().sum(1)-(enc.double()-z).square().sum(1)).max()))
     parts.append((y*sc[k]+mu[k]).float().reshape_as(st[k]));errs.append(float((y-z).square().sum()/z.square().sum()))
    pred[i]=tuple(parts);reconstruction.append(dict(arm=name,row=i,errors=errs))
   candidates[name]=pred
  print('PROJECTED',kind,seed,flush=True)
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
@torch.no_grad()
def suffix(st):
 x,x0,bus=st
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 return (30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30))[0].double()
records=[];samples=[];replay=max(float((suffix(dev[i])-native[i]).abs().max()) for i in list(dev)[:4]);assert replay<1e-5
for name,pred in candidates.items():
 for p in dp:
  for i,d in [p['indices'],p['indices'][::-1]]:
   y=suffix(pred[i]);base=native[i];don=native[d];lp=F.log_softmax(base,0);kl=float((lp.exp()*(lp-F.log_softmax(y,0))).sum());yp=suffix(tuple(dev[i][k]+pred[d][k]-pred[i][k] for k in range(3)));dlp=F.log_softmax(don,0);pk=float((dlp.exp()*(dlp-F.log_softmax(yp,0))).sum());den=float((dlp.exp()*(dlp-lp)).sum());groups=devdata['examples'][i]['groups'];variant=devdata['examples'][i]['variant'];records.append(dict(arm=name,family=p['family'],row=i,whole_KL=kl,paired_KL=pk,paired_denominator=den,correct_top1=int(y.argmax()) in groups[variant]))
   if len(samples)<2*len(candidates) and i==dp[0]['indices'][0]:samples.append(dict(arm=name,row=i,base=base.cpu(),donor=don.cpu(),whole=y.cpu(),paired=yp.cpu()))
 print('EVALUATED',name,flush=True)
summary=[]
for name in candidates:
 for f in families:
  rr=[r for r in records if r['arm']==name and r['family']==f];summary.append(dict(arm=name,family=f,whole_KL=float(np.mean([r['whole_KL'] for r in rr])),paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),top1=float(np.mean([r['correct_top1'] for r in rr]))))
audit={'frozen_hashes_unchanged':all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)] for p in files),'active_ls_squared_error_increase_max':max_ls_violation,'native_logit_replay_max_abs':replay,'finite':bool(np.isfinite([r['whole_KL'] for r in records]).all()),'suffix_sequences':4+len(candidates)*len(dev)*2,'whole_task_holdout_reads':0};assert max_ls_violation<1e-5 and audit['finite'] and audit['frozen_hashes_unchanged'];torch.save(samples,out/'logit_audit_samples.pt');(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,reconstruction=reconstruction,records=records,audits=audit),indent=2)+'\n');print(json.dumps(audit,indent=2))
