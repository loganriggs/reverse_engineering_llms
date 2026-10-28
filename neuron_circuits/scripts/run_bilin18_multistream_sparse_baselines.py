"""H1031 fixed-budget multi-stream sparse representation baseline."""
from pathlib import Path
import sys,json,hashlib,time,numpy as np,torch
from torch import nn
R=Path('/home/loganriggs/Coding/neuron_circuits');A=R/'artifacts/bilin18_tensor_decomposition';S='bilin18_multistream_sparse_baselines';out=A/'multistream_sparse_baselines';out.mkdir(exist_ok=True)
repo=Path('/home/loganriggs/Coding/reverse_engineer/tensor_language')
for p in [repo,repo/'basis_aligned/polynomial_causal',repo/'basis_aligned/bilinear_quotient/ops']:sys.path.insert(0,str(p))
import run_subject_number_sparse_graph_token_extraction_v1 as runner
import bilin18_observed_model_facade as facade
from feature_structures.models import SharedGroups
torch.set_num_threads(6);F=torch.nn.functional;torch.set_float32_matmul_precision('highest')
old=torch.load(A/'multitask_boundary_contract/states.pt',weights_only=False);tp=json.loads((A/'multitask_boundary_contract/panel.json').read_text());bank=json.loads((A/'bilin18_context_training_bank.json').read_text());source=torch.load(A/'bilin18_context_training_bank_captures.pt',weights_only=False);families=['agreement','capital_recall','entity_binding','sentiment','pronoun_gender'];traintexts={e['text'] for e in old['examples'].values()};dp=[p for p in bank['records'] if p['family'] in families and p['context']==0 and p['split']=='lexical' and p['equal_length']];assert len(dp)==19 and not any(t in traintexts for p in dp for t in p['prompts']);devindices=sorted({i for p in dp for i in p['indices']});seeds=[10311,10312];kinds=['scalar','block','shared_fixed'];config={'experiment':'H1031','train':'H1030 60discovery pairs,120endpoint occurrences across five historical families','development':'context0 lexical fourpairs/family except three equal-length capital pairs,38endpoints; exact prompts disjoint, historically opened. Capital prior competence fails and stays diagnostic. Original whole-task holdouts not accessed.','ports':['pre-attention11 residual','x0 embedding recurrence','cached value bus'],'architecture':'Per stream128unit decoder atoms. ScalarTop16; block32disjoint groups of4 withTop4groups; overlapping32groups of8 withTop2groups and degree2 fixed membership. Shared union<=16 per stream; block/scalar exactly16. Three streams combined<=48 per token.','steps':400,'seeds':seeds,'objective':'Equal-family/equal-prompt/equal-stream normalized reconstruction plus0.1 normalized paired-state difference loss. Pair differences are activation supervision, NOT downstream causal supervision. Fixed Adam0.001 and final step, no model selection.','parameters':'Block/shared use identical SharedGroups encoder dimensions and same initial state; scalar uses coordinate encoder directly and has unused group encoder. Registered storage equal, effective trainable parameter count differs and is reported. Fixed membership logits unused in all.','metrics':'Normalized reconstruction and paired-state delta error; full-output KL and answer margin after replacing all three ports; paired decoded-difference interventions compared to full natural donor. All use native suffix with costs explicit.','stop':'No further architecture/rank sweep on this panel. Preserve all seeds and diagnostics; choose next objective based on fidelity failures, not best reconstruction alone.'};(R/f'configs/{S}.json').write_text(json.dumps(config,indent=2)+'\n');(out/'development_panel.json').write_text(json.dumps(dp,indent=2)+'\n')
model,_=facade.load_bilin18(snapshot=Path('/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240'),device='cuda',dtype=torch.float32,verify_weights_sha256=True)
@torch.no_grad()
def capture(i):
 e=source['examples'][i];initial=F.rms_norm(model.transformer.wte(torch.tensor([e['ids']],device='cuda')),(1152,)).float();raw,x0,bus,_,_=runner.component._capture(model,initial,torch,F);return raw,x0,bus
@torch.no_grad()
def suffix(st):
 x,x0,bus=st
 for l in range(11,18):
  b=model.transformer.h[l]
  if l>11:x=b.lambdas[0]*x+b.lambdas[1]*x0
  at,_=b.attn(F.rms_norm(x,(1152,)),bus);x=x+at;x=x+b.mlp(F.rms_norm(x,(1152,)))
 return (30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1])/30))[0].double()
dev={i:capture(i) for i in devindices};native={i:suffix(dev[i]) for i in devindices};torch.save(dict(states={i:tuple(v.cpu() for v in s) for i,s in dev.items()},native={i:v.cpu() for i,v in native.items()},examples={i:source['examples'][i] for i in devindices}),out/'development_states.pt')
# Token-level flattening keeps complete sequences; no padding or token selectors.
X=[[],[],[]];weights=[];offsets={};cursor=0;familycounts={f:sum(p['family']==f for p in tp) for f in families}
for p in tp:
 for i in p['indices']:
  st=old['states'][i];T=st[0].shape[1];offsets[i]=list(range(cursor,cursor+T));cursor+=T
  for port in range(3):X[port].append(st[port].reshape(T,1152))
  weights.extend([1/(len(families)*familycounts[p['family']]*2*T)]*T)
X=torch.stack([torch.cat(x) for x in X]).cuda().float();w=torch.tensor(weights,device='cuda');center=(X*w[None,:,None]).sum(1,keepdim=True);scale=((X-center).square().mean(-1)*w).sum(1).sqrt()[:,None,None];Z=(X-center)/scale;u=[];v=[];pw=[]
for p in tp:
 a,b=p['indices'];assert len(offsets[a])==len(offsets[b]);u+=offsets[a];v+=offsets[b];pw.extend([1/(len(families)*familycounts[p['family']]*len(offsets[a]))]*len(offsets[a]))
u=torch.tensor(u,device='cuda');v=torch.tensor(v,device='cuda');pw=torch.tensor(pw,device='cuda');target=Z[:,v]-Z[:,u];pair_energy=(target.square().mean(-1)*pw).sum(1).clamp_min(1e-8)
records=[];fitrecords=[];maxunion=0;allrecon={};start=time.time()
for seed in seeds:
 for kind in kinds:
  torch.manual_seed(seed);mods=nn.ModuleList([SharedGroups(1152,128,32,4 if kind=='block' else 2,4 if kind=='block' else 8) for _ in range(3)]).cuda();members=[]
  for port,m in enumerate(mods):
   m.membership_logits.requires_grad_(False);M=torch.zeros(32,128,device='cuda')
   if kind=='block':M[torch.arange(32)[:,None],torch.arange(128).reshape(32,4)]=1
   else:
    gen=torch.Generator().manual_seed(seed+port)
    for half in range(2):M[torch.arange(16)[:,None]+half*16,torch.randperm(128,generator=gen).reshape(16,8)]=1
   members.append(M)
  def encode(m,z,M):
   if kind=='scalar':
    a=m.coordinate_encoder(z);ix=a.abs().topk(16,dim=1).indices;mask=torch.zeros_like(a).scatter_(1,ix,1);return (a*mask)@m.dictionary_matrix().T+m.bias,mask
   r=m(z,membership=M);return r['reconstruction'],r['support']
  opt=torch.optim.Adam(mods.parameters(),lr=.001);history=[]
  for step in range(400):
   pred=torch.stack([encode(m,Z[k],members[k])[0] for k,m in enumerate(mods)]);recon=((pred-Z).square().mean(-1)*w).sum(1).mean();change=(((pred[:,v]-pred[:,u]-target).square().mean(-1)*pw).sum(1)/pair_energy).mean();loss=recon+.1*change;opt.zero_grad();loss.backward();assert torch.isfinite(loss);opt.step()
   if step%100==0:history.append(dict(step=step,recon=float(recon.detach()),pair=float(change.detach())))
  reconstructed={};active=[];errors=[]
  with torch.no_grad():
   for i in devindices:
    st=dev[i];parts=[];counts=[];er=[]
    for k,m in enumerate(mods):
     raw=st[k].reshape(-1,1152);z=(raw-center[k])/scale[k];y,mask=encode(m,z,members[k]);parts.append((y*scale[k]+center[k]).reshape_as(st[k]));counts.append(mask.sum(1));er.append(float((y-z).square().sum()/z.square().sum()))
     if kind!='scalar':
      r=m(z,membership=members[k]);maxunion=max(maxunion,float((r['support']-(r['gates']@members[k]>0).float()).abs().max()))
    reconstructed[i]=tuple(parts);active.extend(torch.stack(counts).sum(0).cpu().tolist());errors.append(er)
   for p in dp:
    for i,d in [p['indices'],p['indices'][::-1]]:
     y=suffix(reconstructed[i]);base=native[i];don=native[d];lp=F.log_softmax(base,0);kl=float((lp.exp()*(lp-F.log_softmax(y,0))).sum());groups=source['examples'][i]['groups']
     def margin(y):return float(torch.logsumexp(y[groups[0]],0)-torch.logsumexp(y[groups[1]],0))
     patch=tuple(dev[i][k]+reconstructed[d][k]-reconstructed[i][k] for k in range(3));yp=suffix(patch);g=float((F.log_softmax(don,0).exp()*(F.log_softmax(don,0)-F.log_softmax(yp,0))).sum());den=float((F.log_softmax(don,0).exp()*(F.log_softmax(don,0)-lp)).sum());records.append(dict(kind=kind,seed=seed,family=p['family'],row=i,donor=d,whole_KL=kl,native_margin=margin(base),reconstructed_margin=margin(y),natural_delta=margin(don)-margin(base),predicted_delta=margin(yp)-margin(base),paired_KL=g,paired_denominator=den,native_correct=int(base.argmax()) in groups[source['examples'][i]['variant']],reconstructed_correct=int(y.argmax()) in groups[source['examples'][i]['variant']]))
  fr=dict(kind=kind,seed=seed,registered_parameters=sum(p.numel() for p in mods.parameters()),effective_trainable_parameters=sum(p.numel() for n,p in mods.named_parameters() if p.requires_grad and not(kind=='scalar' and 'group_encoder' in n)),mean_active=float(np.mean(active)),max_active=max(active),min_active=min(active),reconstruction_error_by_stream=np.mean(errors,axis=0).tolist(),history=history);fitrecords.append(fr);torch.save(dict(models=mods.state_dict(),memberships=[m.cpu() for m in members],center=center.cpu(),scale=scale.cpu(),fit=fr),out/f'{kind}_{seed}.pt');print('FIT',kind,seed,fr['mean_active'],flush=True)
summary=[]
for kind in kinds:
 for f in families:
  rr=[r for r in records if r['kind']==kind and r['family']==f];nd=np.array([r['natural_delta'] for r in rr]);pd=np.array([r['predicted_delta'] for r in rr]);summary.append(dict(kind=kind,family=f,whole_KL=float(np.mean([r['whole_KL'] for r in rr])),paired_relative_KL=sum(r['paired_KL'] for r in rr)/sum(r['paired_denominator'] for r in rr),paired_margin_relative_l2=float(np.linalg.norm(pd-nd)/np.linalg.norm(nd)),native_top1=float(np.mean([r['native_correct'] for r in rr])),reconstructed_top1=float(np.mean([r['reconstructed_correct'] for r in rr]))))
audit={'finite':bool(np.isfinite([r['whole_KL'] for r in records]).all()),'union_max_abs':maxunion,'fits_completed':len(fitrecords),'train_dev_prompt_overlap':0,'original_whole_task_holdout_reads':0,'dev_prefix_sequences':38,'suffix_sequences':494,'train_token_occurrences':cursor,'max_active_all':max(r['max_active'] for r in fitrecords)};assert audit['finite'] and maxunion==0 and audit['max_active_all']<=48;(A/f'{S}.json').write_text(json.dumps(dict(config=config,summary=summary,records=records,fits=fitrecords,audits=audit,seconds=time.time()-start),indent=2)+'\n');print(json.dumps(dict(summary=summary,audits=audit),indent=2))
