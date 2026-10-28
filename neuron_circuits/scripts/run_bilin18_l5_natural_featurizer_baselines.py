"""H754 natural L5H3/H5 activations and task-blind linear/adaptive baselines."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_natural_featurizer_baselines'
def reconstruct(x,fit,method,k,seed=None):
 mu=fit['mean'];xc=x.double()-mu
 if method=='coordinate_pca':E=fit['coordinate_pca'];return mu+(xc@E[:,:k])@E[:,:k].T
 E=fit['write_pca']['encode'];B=fit['write_pca']['decode']
 if method=='random_topk':Q=fit['random'][seed];E=E@Q;B=Q.T@B
 z=xc@E
 if method=='write_pca':z=z.clone();z[:,k:]=0
 else:
  mask=torch.zeros_like(z);mask.scatter_(1,z.abs().topk(k,dim=-1).indices,1);z=z*mask
 return mu+z@B
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'data':ART/'bilin18_verified_head_pca_data.pt','task_heads':ART/'bilin18_operation_l5_head_mediation_captures.pt','task_contexts':ART/'bilin18_fresh_wording_subspace_effects_captures.pt','task_results':ART/'bilin18_fresh_wording_subspace_effects.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','source':Path('/home/loganriggs/Coding/reverse_engineer/tensor_language/jacclust/tt_model.py')};hashes={k:sha(p) for k,p in paths.items()};ranks=[1,2,4,8,16,32,64,128];seeds=[7541,7542,7543];sites={'L5H3':3,'L5H5':5}
 cfg={'experiment':'H754','hashes':hashes,'sites':sites,'ranks':ranks,'random_seeds':seeds,'train_documents':2048,'validation_documents':256,'positions':list(range(7,128,8)),'design':'Reuse H722 natural train/validation token documents, no task examples in fits. Capture L5 pre-output-projection head activations using exact six-block prefix. Fit train-centered coordinate PCA and residual-write-metric PCA, plus adaptive top-K in write PCA and three random orthogonal write-coordinate controls. All128 directions retained in fit objects; linear rank and adaptive active-coordinate budget distinguished. Natural validation MSE and reused H747 operation-difference write MSE, separately by native competence/format/query or source position. Adaptive difference is F(donor)-F(receiver), never F(delta).','limitations':'Task-informed head selection, task-blind fits. Reused validation and task panels. No new causal fidelity evaluations or manifold conclusions from reconstruction. Fixed PCA subspaces have rankK; adaptive topK can use all128 directions across examples. Original reserved test untouched.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 old=torch.load(paths['data'],weights_only=False);ids=old['all_token_ids'];positions=old['positions'];assert ids.shape==(2304,128) and positions==cfg['positions'];del old
 doc_hash=[hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in ids];overlap=len(set(doc_hash[:2048])&set(doc_hash[2048:]));assert overlap==0
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 captures={s:[] for s in sites}
 for lo in range(0,len(ids),16):
  saved={}
  def capture(m,args):
   for site,h in sites.items():saved[site]=args[0][:,positions,h*128:(h+1)*128].detach().cpu()
  hook=model.blocks[5].attn.c_proj.register_forward_pre_hook(capture)
  try:
   x0=F.rms_norm(model.embedding(ids[lo:lo+16].cuda()),(1152,));x=x0;bus=None
   for block in model.blocks[:6]:x,bus=block(x,bus,x0)
  finally:hook.remove()
  for s in sites:captures[s].append(saved[s])
 captures={s:torch.cat(v) for s,v in captures.items()};print('CAPTURED',flush=True)
 fits={};audits_fit={};spectra=[]
 for site,h in sites.items():
  train=captures[site][:2048].reshape(-1,128).double();val=captures[site][2048:].reshape(-1,128).double();mu=train.mean(0);xc=train-mu;vc=val-mu;W=model.blocks[5].attn.c_proj.weight[:,h*128:(h+1)*128].cpu().double();U,sv,Vt=torch.linalg.svd(W,full_matrices=False);A=Vt.T*sv;Ai=Vt/sv[:,None];ev,P=torch.linalg.eigh((xc@A).T@(xc@A)/len(xc));ev=ev.flip(0);P=P.flip(1);_,Pc=torch.linalg.eigh(xc.T@xc/len(xc));Pc=Pc.flip(1);E=A@P;B=P.T@Ai
  random={seed:torch.linalg.qr(torch.randn(128,128,generator=torch.Generator().manual_seed(seed),dtype=torch.double)).Q for seed in seeds};fit={'mean':mu,'write':W,'coordinate_pca':Pc,'write_pca':{'encode':E,'decode':B},'write_eigenvalues':ev,'random':random};fits[site]=fit
  audits_fit[site]={'full_rank_inverse_max_abs':float((E@B-torch.eye(128)).abs().max()),'write_isometry_max_relative':float(((vc@A)@U.T-vc@W.T).norm()/(vc@W.T).norm()),'projection_condition':float(sv.max()/sv.min())}
  for method,seed in [('coordinate_pca',None),('write_pca',None),('adaptive_write_pca',None)]+[('random_topk',seed) for seed in seeds]:
   for k in ranks:
    err=reconstruct(val,fit,method,k,seed)-val;spectra.append({'site':site,'method':method,'seed':seed,'budget':k,'coordinate_relative_MSE':float(err.square().sum()/vc.square().sum()),'write_relative_MSE':float((err@W.T).square().sum()/(vc@W.T).square().sum())})
 heads=torch.load(paths['task_heads'],weights_only=False)['native_heads'];ctx=torch.load(paths['task_contexts'],weights_only=False);task=json.loads(paths['task_results'].read_text());meta={(r['id'],r['receiver']):r for r in task['records'] if r['scope']=='native'};task_records=[]
 for site,h in sites.items():
  fit=fits[site];W=fit['write'];keys=list(heads);stack=torch.stack([heads[k][:,h*128:(h+1)*128] for k in keys]).double();index={k:i for i,k in enumerate(keys)}
  for method,seed in [('coordinate_pca',None),('write_pca',None),('adaptive_write_pca',None)]+[('random_topk',seed) for seed in seeds]:
   for k in ranks:
    rec=reconstruct(stack.reshape(-1,128),fit,method,k,seed).view_as(stack)
    for rk,dk in ctx['pairs']:
     if meta[rk]['family']!='entity_binding':continue
     m=meta[rk];cell='|'.join(m[v] for v in ['name_set','wording','format']);competent=all(task['competence'][cell+'|'+fam]['pass'] for fam in ['entity_binding','sequence_copy']);delta=stack[index[dk]]-stack[index[rk]];err=rec[index[dk]]-rec[index[rk]]-delta
     for pos,label in enumerate(['query','source']):task_records.append({'id':rk[0],'receiver':rk[1],'site':site,'method':method,'seed':seed,'budget':k,'position':label,'format':m['format'],'competent':competent,'write_error_squared':float((err[pos]@W.T).square().sum()),'full_write_squared':float((delta[pos]@W.T).square().sum())})
 groups=defaultdict(list)
 for r in task_records:groups[r['site'],r['method'],r['seed'],r['budget'],r['position'],r['format'],r['competent']].append(r)
 task_summary=[dict(zip(['site','method','seed','budget','position','format','competent'],key),write_relative_MSE=sum(r['write_error_squared'] for r in rr)/sum(r['full_write_squared'] for r in rr),pairs=len(rr)) for key,rr in groups.items()]
 fullerr=max(s['write_relative_MSE'] for s in spectra if s['budget']==128);audits={'hashes_unchanged':{k:sha(p)==hashes[k] for k,p in paths.items()},'train_validation_exact_document_overlap':overlap,'fit':audits_fit,'full_rank_validation_write_relative_MSE_max':fullerr,'finite':all(torch.isfinite(v).all().item() for v in captures.values()),'six_block_prefix_evaluations':len(ids),'whole_model_forwards':0,'test_evaluations':0};assert all(audits['hashes_unchanged'].values()) and audits['finite'] and fullerr<1e-20 and max(v['full_rank_inverse_max_abs'] for v in audits_fit.values())<1e-10
 torch.save({'activations':captures,'positions':positions,'all_token_ids':ids,'document_hashes':doc_hash},ART/f'{STEM}_data.pt');torch.save({'fits':fits},ART/f'{STEM}_fits.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'spectra':spectra,'task_records':task_records,'task_summary':task_summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
 for s in spectra:
  if s['budget'] in [4,8,16,32] and s['method']!='random_topk':print(s,flush=True)
if __name__=='__main__':main()
