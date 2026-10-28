"""H601 exploratory item holdout: context span by signal metric at fixed rank."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_reader_operation_format import run
from run_bilin18_qkov_causal_edges import load_full_model
STEM='bilin18_metric_coverage_factorial'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);start=time.time();source=ART/'bilin18_metric_cross_task_captures.pt';bp=ART/'bilin18_number_fixed_response_metric_metric.pt';old=torch.load(bp,weights_only=False);d=torch.load(source,weights_only=False);ex=d['examples'];train=[k for k in ex if k[1]<2 and ex[k]['item']<4];test=[k for k in ex if k[1]<2 and ex[k]['item']>=4];assert len(train)==len(test)==80 and not set(train)&set(test)
 cfg=dict(experiment='H601',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),old_metric_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rank=64,discovery_items=[0,1,2,3],evaluation_items=[4,5,6,7],design='Old number-trained versus broad five-task context basis, both rank64. Euclidean, number-signal and equal-case three-signal response metrics in each basis; plus per-case oracle and historical H598 metric. Fit solely H600 item0-3 native same-answer context differences and incoming responses. Evaluate all item4-7 cases including failed competence. Reused previously observed panel, exploratory item exclusion not independent confirmation. Shared words/templates may occur across item split.',signals=['number','task','random'],metrics='Internal norm, reference KL, contrast margin and intended-token CE separately. Same nuisance-context switch and upstream norm as H600.',fineweb_new_documents=0)
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2))
 D=torch.stack([d['inputs'][k[0],k[1]+2]-d['inputs'][k] for k in train]);U,S,Vh=torch.linalg.svd(D.double(),full_matrices=False);assert S[63]>S[0]*1e-8;bases={'old':old['basis'],'broad':Vh[:64].float()};basis=torch.cat(list(bases.values())).cuda();trainjobs=[j for j in d['cache'] if j[0] in set(train)];testjobs=[j for j in d['cache'] if j[0] in set(test)];assert len(trainjobs)==len(testjobs)==480
 model=load_full_model('cuda');assert not model.config.gated
 for mod in [model.embedding,model.blocks,model.lm_head]:
  mod.float().eval()
  for p in mod.parameters():p.requires_grad_(False)
 m=model.blocks[11].mlp
 def T(c,e):
  c=c.cuda();e=e.cuda();v=m.Left(c)*m.Right(e)+m.Left(e)*m.Right(c);return (m.Down(v)-m.Down(torch.zeros_like(v))).cpu()
 cross={n:torch.zeros(128,1152,dtype=torch.float64) for n in ['number','multi']};counts={'number':0,'multi':0};la=m.Left(basis);ra=m.Right(basis)
 for j in trainjobs:
  e=d['cache'][j]['de'].cuda();le=m.Left(e);re=m.Right(e);v=la*re+ra*le;out=m.Down(v)-m.Down(torch.zeros_like(v));back=out@m.Down.weight;ag=((back*re)@m.Left.weight+(back*le)@m.Right.weight).cpu().double();cross['multi']+=ag;counts['multi']+=1
  if j[1]=='number':cross['number']+=ag;counts['number']+=1
 maps={};audits=[]
 for name in cross:
  cross[name]/=counts[name]
  for bi,(space,A) in enumerate(bases.items()):
   C=cross[name][bi*64:(bi+1)*64];H=C@A.double().T;K=torch.linalg.solve(H,C).float();maps[space,name]=K;audits.append(dict(space=space,metric=name,condition=float(torch.linalg.cond(H)),solve_error=float((H@K.double()-C).norm()/C.norm()),projector_identity_error=float((K@A.T-torch.eye(64)).abs().max())))
 fitted=ART/f'{STEM}_locked_maps.pt';torch.save(dict(bases=bases,maps=maps,training_keys=train,training_jobs=trainjobs,singular_values=S,config=cfg),fitted);fh=hashlib.sha256(fitted.read_bytes()).hexdigest();print('FIT',counts,audits,flush=True)
 # Independent forward normal-equation audit on seeded vectors, all fitted metrics.
 probes=torch.randn(3,1152,generator=torch.Generator().manual_seed(601));rhs={key:torch.zeros(64,3,dtype=torch.float64) for key in maps}
 for j in trainjobs:
  e=d['cache'][j]['de'];target=T(probes,e).T.double()
  for space,A in bases.items():
   inc=T(A,e).double()@target;rhs[space,'multi']+=inc/counts['multi']
   if j[1]=='number':rhs[space,'number']+=inc/counts['number']
 for q in audits:
  space=q['space'];name=q['metric'];bi=list(bases).index(space);C=cross[name][bi*64:(bi+1)*64];q['forward_adjoint_error']=float((rhs[space,name]-C@probes.double().T).norm()/rhs[space,name].norm());assert q['forward_adjoint_error']<2e-6 and q['projector_identity_error']<2e-6
 modes=['full','native','historical']+[f'{space}_{mode}' for space in bases for mode in ['projection','number','multi','oracle']];vec={};geometry=[];buckets=defaultdict(list)
 for j in testjobs:
  k,signal,sign=j;c=d['cache'][j];dc=c['dc'];e=c['de'];full=d['vectors'][j]['full'];vv={'full':full,'native':torch.zeros_like(full),'historical':d['vectors'][j]['metric']}
  for space,A in bases.items():
   vv[space+'_projection']=T(A.T@(A@dc),e)
   for met in ['number','multi']:vv[space+'_'+met]=T(A.T@(maps[space,met]@dc),e)
   M=T(A,e).T.double();u,s,vh=torch.linalg.svd(M,full_matrices=False);keep=s>s[0]*1e-10;coef=vh[keep].T@((u[:,keep].T@full.double())/s[keep]);pred=M@coef;vv[space+'_oracle']=pred.float();geometry.append(dict(space=space,id=k[0],signal=signal,sign=sign,normal_error=float((M.T@(pred-full.double())).norm()/(M.norm()*full.norm()))))
  vec[j]=vv;buckets[len(ex[k]['ids'])].append(j)
 # Competence of held-out endpoints, not the whole source panel.
 competence={}
 for fam in sorted({ex[k]['family'] for k in test}):
  for template in [0,1]:
   kk=[k for k in test if ex[k]['family']==fam and ex[k]['template']==template];rr=[k[0] for k in kk if k[1]==0];both=np.mean([measure(d['native'][r,0],ex[r,0]['groups'])['margin']>0 and measure(d['native'][r,1],ex[r,1]['groups'])['margin']<0 for r in rr]);top=np.mean([measure(d['native'][k],ex[k]['groups'])['top1'][k[1]] for k in kk]);competence[f'{fam}_{template}']=dict(pairs=len(rr),both_endpoint_choice=float(both),full_vocab_top1=float(top),passes=bool(len(rr)>=4 and both>=.75 and top>=.5))
 print('COMPETENCE',competence,flush=True)
 nb=torch.load(ART/'bilin18_agreement_write_subspace_locked_bases.pt',weights_only=False);axis=nb['pca'][0]*torch.sign(nb['pca'][0]@nb['mean']);size=float(axis@nb['mean'])*.0625;rand=torch.randn(1152,generator=torch.Generator().manual_seed(600));rand/=rand.norm()
 def delta(j):
  k,s,sign=j
  v=axis if s=='number' else rand if s=='random' else d['attn'][k[0],1-k[1]]-d['attn'][k]
  if s=='task':v=v/v.norm()
  return v*size*sign
 result=dict(config=cfg,competence=competence,fitting_audits=audits,geometry=geometry,interventions=[],summary={},status='running');refs={};replay=[]
 def save():result['elapsed_seconds']=time.time()-start;(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2))
 save()
 for mode in modes:
  for jj in buckets.values():
   for st in range(0,len(jj),16):
    bb=jj[st:st+16];rep=torch.stack([d['cache'][j]['write']+vec[j][mode] for j in bb]).cuda()
    def h(m,args,out):y=out.clone();y[:,-1]=rep.to(y);return y
    hook=m.register_forward_hook(h)
    try:z,_=run(model,torch.tensor([ex[j[0]]['ids'] for j in bb],device='cuda'),patch=(5,torch.stack([delta(j) for j in bb])))
    finally:hook.remove()
    for i,j in enumerate(bb):
     k,signal,sign=j;e=ex[k];zz=z[i].cpu()
     if mode=='full':refs[j]=zz
     if mode=='native':replay.append(float((zz-d['cache'][j]['logits']).abs().max()))
     rp=refs[j].double().log_softmax(0);lp=zz.double().log_softmax(0);f=vec[j]['full'];result['interventions'].append(dict(id=k[0],variant=k[1],family=e['family'],template=e['template'],signal=signal,sign=sign,mode=mode,margin_error=measure(zz,e['groups'])['margin']-measure(refs[j],e['groups'])['margin'],reference_kl=float((rp.exp()*(rp-lp)).sum()),target_ce_error=float(rp[e['target']]-lp[e['target']]),internal_response_error=float((vec[j][mode]-f).norm()/f.norm())))
  print('COMPLETE',mode,flush=True);save()
 for cell in competence:
  fam,t=cell.rsplit('_',1)
  for signal in cfg['signals']:
   cc=[c for c in result['interventions'] if c['family']==fam and c['template']==int(t) and c['signal']==signal];base=[c for c in cc if c['mode']=='native'];md=np.mean([abs(c['margin_error']) for c in base]);kd=np.mean([c['reference_kl'] for c in base]);cd=np.mean([abs(c['target_ce_error']) for c in base])
   for mode in modes:
    vv=[c for c in cc if c['mode']==mode];result['summary'][f'{cell}_{signal}_{mode}']=dict(internal_response_error=float(np.mean([c['internal_response_error'] for c in vv])),relative_margin_error=float(np.mean([abs(c['margin_error']) for c in vv])/md),relative_reference_kl=float(np.mean([c['reference_kl'] for c in vv])/kd),reference_kl=float(np.mean([c['reference_kl'] for c in vv])),relative_target_ce_error=float(np.mean([abs(c['target_ce_error']) for c in vv])/cd))
 result['audits']=dict(source_unchanged=cfg['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest(),old_metric_unchanged=cfg['old_metric_sha256']==hashlib.sha256(bp.read_bytes()).hexdigest(),runner_unchanged=cfg['runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fitted_maps_unchanged=fh==hashlib.sha256(fitted.read_bytes()).hexdigest(),train_test_item_overlap=0,training_contexts=len(train),training_number_responses=counts['number'],training_multi_responses=counts['multi'],heldout_responses=len(testjobs),interventions=len(result['interventions']),replay_max_logit_error=max(replay),oracle_normal_error=max(g['normal_error'] for g in geometry),broad_basis_rank=64,broad_training_energy=float(S[:64].square().sum()/S.square().sum()),finite=all(np.isfinite(c[n]) for c in result['interventions'] for n in ['reference_kl','margin_error','target_ce_error','internal_response_error']));result['status']='complete';save();print('AUDITS',result['audits'],flush=True)
if __name__=='__main__':main()
