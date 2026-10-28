"""H723: task-blind PCA in the projected residual-write metric."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch,torch.nn.functional as F,numpy as np
from transformers import AutoTokenizer
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_diverse_causal_tasks import ROOT,ART,groups as answer_groups,measure
from run_bilin18_broad_task_competence import continuation_id
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_reader_operation_format import run
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_verified_head_write_pca'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_verified_head_pca_data.pt','manifest':ART/'bilin18_group_task_transfer_manifest.json','heads':ART/'bilin18_agreement_capital_heads_captures.pt','head_results':ART/'bilin18_agreement_capital_heads.json','suite':ROOT/'scripts/run_bilin18_diverse_causal_tasks.py','tokenization':ROOT/'scripts/run_bilin18_broad_task_competence.py','model':SNAPSHOT/'pytorch_model.bin','runner':Path(__file__),'forward':ROOT/'scripts/run_bilin18_reader_operation_format.py','loader':ROOT/'scripts/run_bilin18_qkov_causal_edges.py'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);ids=data['all_token_ids'][:2304].clone();positions=data['positions'];captures=data['activations'];del data;sites={'agreement':(11,3),'capital_recall':(13,0)};ranks=[1,2,4,8,16,32,64,96,128];cfg=dict(experiment='H723',hashes=hashes,sites=sites,ranks=ranks,design='ReuseH722naturalheadcaptures/splits. ForheadoutputmatrixW=U diag(s) Vt, mapcoordinatesh to isometricwritecoordinates y=h V diag(s). Fittrain-onlyPCAiny, decodeback withdiag(1/s)Vt. Thisminimizesresidual-writeMSEatfixedrank, notcoordinateMSEoroutputKL. No tasklabels. ReuseexactH722task/nuisance and32validationdocfidelitypanels, same ranks. Verifyfullrankreplay andexactlinearalgebra.',limitations='Task-informedsitechoice, task-blindfit. Reusedresearchpanels. WritePCAislinearbaseline, notmanifold orsemanticdiscovery. Retainedheadbasisnonorthogonal underheadmetric. NativeoutputKL/CEandtaskdifferencepreservation distinct. Originalreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 partialforwards=0;fits={};spectra=[];fit_audits={}
 for f,(l,h) in sites.items():
  train=captures[f][:2048].reshape(-1,128).double();val=captures[f][2048:].reshape(-1,128).double();mu=train.mean(0);xc=train-mu;vc=val-mu;W=model.blocks[l].attn.c_proj.weight[:,h*128:(h+1)*128].cpu().double();U,sv,Vt=torch.linalg.svd(W,full_matrices=False);A=Vt.T*sv;Ai=Vt/sv[:,None];y=xc@A;cov=y.T@y/len(y);eig,P=torch.linalg.eigh(cov);eig=eig.flip(0);P=P.flip(1);E=A@P;B=P.T@Ai;fits[f]=dict(mean=mu,eigenvalues=eig,write_pca=dict(encode=E,decode=B),write=W,svd_singular_values=sv,write_isometry=U)
  fit_audits[f]=dict(output_projection_condition=float(sv.max()/sv.min()),full_rank_inverse_max_abs=float((E@B-torch.eye(128)).abs().max()),write_isometry_relative_error=float(((vc@A)@U.T-vc@W.T).norm()/(vc@W.T).norm()))
  assert fit_audits[f]['full_rank_inverse_max_abs']<1e-10 and fit_audits[f]['write_isometry_relative_error']<1e-10
  for k in ranks:
   err=vc-(vc@E[:,:k])@B[:k];spectra.append(dict(site=f,method='write_pca',rank=k,validation_coordinate_relative_mse=float(err.square().sum()/vc.square().sum()),validation_write_relative_mse=float((err@W.T).square().sum()/(vc@W.T).square().sum())))
  print('WRITE_FIT',f,fit_audits[f],flush=True)
 torch.save(dict(fits=fits),ART/f'{STEM}_fits.pt');old=torch.load(paths['heads'],weights_only=False);oldj=json.loads(paths['head_results'].read_text());rows=json.loads(paths['manifest'].read_text());tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True);examples={}
 for r in rows:
  if r['family'] not in sites:continue
  for v,text in enumerate(r['prompts']+[r['nuisance']]):
   gs=[[continuation_id(tok,text,a) for a in g] for g in answer_groups(r['family'],r['answers'])];gs=[[a for a in g if a is not None] for g in gs];examples[r['id'],v]=dict(ids=tok.encode(text,add_special_tokens=False),groups=gs)
 records=[];forwards=0;fullerrors=[]
 for method in ['write_pca']:
  for k in ranks:
   buckets=defaultdict(list)
   for row in rows:
    if row['family'] not in sites:continue
    for rv,dv in [(0,1),(1,0),(0,2)]:buckets[row['family'],len(examples[row['id'],rv]['ids'])].append(dict(id=row['id'],family=row['family'],split=row['split'],receiver=rv,donor=dv))
   for (f,length),jobs in buckets.items():
    l,h=sites[f];E=fits[f][method]['encode'][:,:k];B=fits[f][method]['decode'][:k]
    for st in range(0,len(jobs),12):
     bb=jobs[st:st+12];delta=torch.stack([old['head_coordinates'][r['id'],r['donor']][h]-old['head_coordinates'][r['id'],r['receiver']][h] for r in bb]).double();patch=((delta@E)@B).float().cuda()
     def edit(m,args):
      y=args[0].clone();y[:,-1,h*128:(h+1)*128]+=patch.to(y);return (y,)
     hook=model.blocks[l].attn.c_proj.register_forward_pre_hook(edit)
     try:logits,_=run(model,torch.tensor([examples[r['id'],r['receiver']]['ids'] for r in bb],device='cuda'))
     finally:hook.remove()
     forwards+=len(bb)
     for n,r in enumerate(bb):
      rk=(r['id'],r['receiver']);gs=examples[rk]['groups'];base=measure(old['native_logits'][rk],gs);ed=measure(logits[n],gs);sign=1 if r['donor'] in [0,2] else -1;change=ed['margin']-base['margin'];records.append(dict(**r,method=method,rank=k,margin_change=change,donor_directed_margin_change=sign*change,edited_donor_correct=ed['top1'][0 if r['donor']==2 else r['donor']]))
      if k==128:
       prev=next(v for v in oldj['records'] if v['id']==r['id'] and v['receiver']==r['receiver'] and v['donor']==r['donor'] and v['head']==h);fullerrors.append(abs(change-prev['margin_change']))
   print('TASK',method,k,flush=True)
 valdocs=sorted(range(2048,2304),key=lambda d:__import__('hashlib').sha256(f'H722:{d}'.encode()).hexdigest())[:32];native=[]
 for st in range(0,32,8):
  logits,_=run(model,ids[valdocs[st:st+8],:64].cuda());native.extend(logits.cpu());forwards+=len(logits)
 fidelity=[]
 for f,(l,h) in sites.items():
  mu=fits[f]['mean'];raw=captures[f][valdocs,7].double()
  for method in ['write_pca']:
   for k in [8,32,64,128]:
    E=fits[f][method]['encode'][:,:k];B=fits[f][method]['decode'][:k];replacement=(mu+((raw-mu)@E)@B).float()
    for st in range(0,32,8):
     repl=replacement[st:st+8].cuda()
     def edit(m,args):
      y=args[0].clone();y[:,-1,h*128:(h+1)*128]=repl.to(y);return (y,)
     hook=model.blocks[l].attn.c_proj.register_forward_pre_hook(edit)
     try:logits,_=run(model,ids[valdocs[st:st+8],:64].cuda())
     finally:hook.remove()
     forwards+=len(logits)
     for n in range(len(logits)):
      base=native[st+n].double().log_softmax(-1);ed=logits[n].cpu().double().log_softmax(-1);nextid=int(ids[valdocs[st+n],64]);fidelity.append(dict(site=f,method=method,rank=k,doc=valdocs[st+n],native_to_reconstruction_kl=float((base.exp()*(base-ed)).sum()),observed_next_token_ce_change=float(base[nextid]-ed[nextid])))
  print('FIDELITY',f,flush=True)
 zero=max(r['native_to_reconstruction_kl'] for r in fidelity if r['rank']==128);audits=dict(fit_audits=fit_audits,hashes_unchanged={a:sha(p)==hashes[a] for a,p in paths.items()},finite=all(np.isfinite(r['native_to_reconstruction_kl']) for r in fidelity),full_rank_head_swap_margin_replay_max_abs=max(fullerrors),full_rank_reconstruction_max_kl=zero,fourteen_block_prefix_evaluations=partialforwards,whole_model_forwards=forwards,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(fullerrors)<1e-4 and zero<1e-7;(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,spectra=spectra,task_records=records,fidelity=fidelity,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
