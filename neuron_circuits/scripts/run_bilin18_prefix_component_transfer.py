"""H604 leave-task-and-wording-out prefix component prediction."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_prefix_component_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();src=ART/'bilin18_metric_wording_edit_factorial_captures.pt';d=torch.load(src,weights_only=False);ex=d['examples'];families=sorted({e['family'] for e in ex.values()});cfg=dict(experiment='H604',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),design='Leave each task out of fitting. Train prefix context differences on old wording (template1) from the other four tasks, 32 endpoint vectors. Evaluate held-out task/new wording (template3), eight endpoints x six signed signals. Fit fixed mean, mean direction, uncentered PCA8/32 and seeded random8/32 controls. Same-task-old-wording rank8 is a separate diagnostic and not task holdout. Per-case oracle8/32 sees response targets. Reused H603 panel; adaptive cross-validation, not independent confirmation.',seed=604,rank_rule='Exclude singular values <=1e-6 of largest; shared32 means up to32, same_task8 means up to8, random32 matched to effective shared rank. Repeated capital prompts create exact dependencies.',scope='Predict one prefix edit response in MLP11. Projection modes still observe actual dc; only fixed_mean predicts dc without case-specific context difference. All modes use observed incoming e. Same lexical items across wording; task folds not independent corpus samples. No semantic-unit or global compression claim.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));folds={};gen=torch.Generator().manual_seed(604);random=torch.linalg.qr(torch.randn(1152,32,generator=gen,dtype=torch.float64)).Q.T.float()
 for fam in families:
  train=[k for k,e in ex.items() if k[1]<2 and e['template']==1 and e['family']!=fam];own=[k for k,e in ex.items() if k[1]<2 and e['template']==1 and e['family']==fam];test=[k for k,e in ex.items() if k[1]<2 and e['template']==3 and e['family']==fam];assert len(train)==32 and len(own)==len(test)==8
  def contexts(kk):return torch.stack([d['inputs'][k[0],k[1]+2]-d['inputs'][k] for k in kk])
  X=contexts(train);Y=contexts(own);_,sv,V=torch.linalg.svd(X.double(),full_matrices=False);_,sy,Vy=torch.linalg.svd(Y.double(),full_matrices=False);V=V[sv>sv[0]*1e-6];Vy=Vy[sy>sy[0]*1e-6];assert len(V)>=8;mean=X.mean(0);folds[fam]=dict(train=train,own=own,test=test,mean=mean,mean_axis=mean/mean.norm(),pca=V.float(),own8=Vy.float(),singular_values=sv,effective_rank=len(V),same_task_rank=len(Vy))
 fitted=ART/f'{STEM}_locked_folds.pt';torch.save(dict(folds=folds,random=random,config=cfg),fitted);fh=hashlib.sha256(fitted.read_bytes()).hexdigest();print('FROZEN',len(folds),'folds',flush=True)
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 m=model.blocks[11].mlp
 def T(c,e):
  c=c.cuda();e=e.cuda();v=m.Left(c)*m.Right(e)+m.Left(e)*m.Right(c);return (m.Down(v)-m.Down(torch.zeros_like(v))).cpu()
 modes=['full','native','fixed_mean','mean_projection','shared8','shared32','random8','random32','same_task8','oracle8','oracle32'];vectors={};geometry=[];buckets=defaultdict(list)
 for job,c in d['cache'].items():
  k,signal,sign=job;e=ex[k]
  if e['template']!=3:continue
  fold=folds[e['family']];assert k in fold['test'];dc=c['dc'];de=c['de'];full=d['vectors'][job]['full'];axis=fold['mean_axis'];vv={'full':full,'native':torch.zeros_like(full),'fixed_mean':T(fold['mean'],de),'mean_projection':T(axis*(axis@dc),de)};cg={}
  for label,B in [('shared8',fold['pca'][:8]),('shared32',fold['pca']),('random8',random[:8]),('random32',random[:len(fold['pca'])]),('same_task8',fold['own8'])]:
   p=B.T@(B@dc);vv[label]=T(p,de);cg[label]=float((p-dc).norm()/dc.norm())
  for rank in [8,32]:
   B=fold['pca'][:rank];M=T(B,de).T.double();U,S,Vh=torch.linalg.svd(M,full_matrices=False);keep=S>S[0]*1e-10;coef=Vh[keep].T@((U[:,keep].T@full.double())/S[keep]);pred=M@coef;vv[f'oracle{rank}']=pred.float();cg[f'oracle{rank}_normal_error']=float((M.T@(pred-full.double())).norm()/(M.norm()*full.norm()))
  cg['fixed_mean']=float((fold['mean']-dc).norm()/dc.norm());cg['mean_projection']=float((axis*(axis@dc)-dc).norm()/dc.norm());geometry.append(dict(id=k[0],variant=k[1],family=e['family'],signal=signal,sign=sign,context_errors=cg));vectors[job]=vv;buckets[len(e['ids'])].append(job)
 sourcej=json.loads((ART/'bilin18_metric_wording_edit_factorial.json').read_text());competence={fam:sourcej['competence'][fam+'_3'] for fam in families};result=dict(config=cfg,competence=competence,geometry=geometry,interventions=[],summary={},status='running');refs={};replay=[]
 nb=torch.load(ART/'bilin18_agreement_write_subspace_locked_bases.pt',weights_only=False);axis=nb['pca'][0]*torch.sign(nb['pca'][0]@nb['mean']);size=float(axis@nb['mean'])*.0625;rand=torch.randn(1152,generator=torch.Generator().manual_seed(600));rand/=rand.norm()
 def delta(j):
  k,s,sgn=j;v=axis if s=='number' else rand if s=='random' else d['attn'][k[0],1-k[1]]-d['attn'][k]
  if s=='task':v=v/v.norm()
  return v*size*sgn
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save();print('INTERVENTIONS',len(vectors)*len(modes),flush=True)
 for mode in modes:
  for jj in buckets.values():
   for st in range(0,len(jj),16):
    bb=jj[st:st+16];rep=torch.stack([d['cache'][j]['write']+vectors[j][mode] for j in bb]).cuda()
    def hook(m,args,out):y=out.clone();y[:,-1]=rep.to(y);return y
    h=m.register_forward_hook(hook)
    try:z,_=run(model,torch.tensor([ex[j[0]]['ids'] for j in bb],device='cuda'),patch=(5,torch.stack([delta(j) for j in bb])))
    finally:h.remove()
    for i,j in enumerate(bb):
     k,signal,sign=j;e=ex[k];zz=z[i].cpu()
     if mode=='full':refs[j]=zz
     if mode=='native':replay.append(float((zz-d['cache'][j]['logits']).abs().max()))
     rp=refs[j].double().log_softmax(0);lp=zz.double().log_softmax(0);f=vectors[j]['full'];result['interventions'].append(dict(id=k[0],variant=k[1],family=e['family'],signal=signal,sign=sign,mode=mode,margin_error=measure(zz,e['groups'])['margin']-measure(refs[j],e['groups'])['margin'],reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce_error=float(rp[e['target']]-lp[e['target']]),internal_response_error=float((vectors[j][mode]-f).norm()/f.norm())))
  save();print('COMPLETE',mode,flush=True)
 for fam in families:
  for signal in ['number','task','random']:
   cc=[c for c in result['interventions'] if c['family']==fam and c['signal']==signal];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
   for mode in modes:
    vv=[c for c in cc if c['mode']==mode];result['summary'][f'{fam}_{signal}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd))
 result['audits']=dict(source_unchanged=cfg['source_sha256']==hashlib.sha256(src.read_bytes()).hexdigest(),runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fitted_folds_unchanged=fh==hashlib.sha256(fitted.read_bytes()).hexdigest(),folds=len(folds),effective_ranks={f:v['effective_rank'] for f,v in folds.items()},same_task_ranks={f:v['same_task_rank'] for f,v in folds.items()},per_fold_training_contexts=32,per_fold_evaluation_contexts=8,training_task_overlap=0,training_wording_overlap=0,same_task8_is_exception=True,case_count=len(vectors),intervention_count=len(result['interventions']),replay_max_logit_error=max(replay),oracle_normal_error=max(g['context_errors'][f'oracle{r}_normal_error'] for g in geometry for r in [8,32]),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['margin_error','reference_kl','target_ce_error','internal_response_error']));torch.save(dict(vectors=vectors,folds=folds),ART/f'{STEM}_vectors.pt');result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
