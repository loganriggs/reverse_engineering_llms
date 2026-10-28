"""H946 discovery-only mechanism anchors for real causal distillation."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_real_causal_distillation import Student
from pathlib import Path
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_supervised_relay_anchors';ARMS=['continued','anchor_endpoints','anchor_effects']
@torch.inference_mode()
def teacher_targets(mlp,x):return torch.cat([mlp(F.rms_norm(b.cuda(),(1152,))).cpu() for b in x.split(64)])
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'student':ROOT/'scripts/run_bilin18_real_causal_distillation.py','natural':ART/'bilin18_real_causal_distillation_targets.pt','initial':ART/'bilin18_real_causal_distillation_fits.pt','discovery':ART/'bilin18_same_token_context_transfer_captures.pt','selection':ART/'bilin18_context_relay_positions.json','examples':ART/'bilin18_candidate_task_screen_captures.pt','evaluation':ART/'bilin18_distilled_relay_mechanisms_captures.pt','evaluation_examples':ART/'bilin18_relay_fresh_distance_captures.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H946','hashes':hashes,'arms':ARMS,'seeds':[0,1],'steps':2000,'starting_point':'Same-seed H943 effect-trained checkpoint, fresh Adam .003 for allarms. Same64products/222336parameters. Continued natural+random-effect objective in everyarm.','anchors':'Original H848 discovery only:8 agreement/7 sentiment lexicalpairs, both donor directions; exclude entire discovery pair if either prompt exactly overlaps evaluation (input-only rule), full rawpreMLP4 relay contrast. No projected axes or H849 evaluation states in optimizer. Equal-task weighting; all anchor rows each step.','anchor_losses':'Endpoints: per-task average native/edited outputMSE normalized by centered discovery native output energy. Effects: per-task finite-effectMSE normalized by discovery effect energy. Each added with totalweight1. Endpoint/effect differ in weighting and share endpoints.','evaluation':'Existing H849 full/rank1 contrasts and H943 natural random/opposite/double-dose panels, no modelselection; every finalfit retained. Known-mechanism recovery is supervised validation, not discovery. Input prompt sets checked disjoint.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 natural=torch.load(paths['natural'],weights_only=False);initial=torch.load(paths['initial'],weights_only=False);source=torch.load(paths['discovery'],weights_only=False);selection=json.loads(paths['selection'].read_text());examples=torch.load(paths['examples'],weights_only=False)['examples'];evaluation=torch.load(paths['evaluation'],weights_only=False);evalex=torch.load(paths['evaluation_examples'],weights_only=False)['examples'];offset={'agreement':-3,'sentiment':-4};anchor={};manifest=[];trainprompts=set();testprompts={tuple(e['ids']) for e in evalex};model=load_full_model('cuda');mlp=model.blocks[4].mlp.float().eval()
 for fam in offset:
  jobs=[r for r in selection['records'] if r['family']==fam and r['from_end']==offset[fam] and r['split']=='discovery'];assert len(jobs)==16;excluded={r['id'] for r in jobs if tuple(examples[r['i']]['ids']) in testprompts or tuple(examples[r['j']]['ids']) in testprompts};jobs=[r for r in jobs if r['id'] not in excluded];assert len(jobs)==(16 if fam=='agreement' else 14)
  x=torch.stack([source['premlp'][r['i']][4][r['position']] for r in jobs]).float();b=torch.stack([source['premlp'][r['j']][4][r['position']] for r in jobs]).float();y=teacher_targets(mlp,x);w=teacher_targets(mlp,b);d=b-x;e=w-y;anchor[fam]=dict(x=x,delta=d,y=y,effect=e,native_den=(y-y.mean(0)).square().mean(),effect_den=e.square().mean())
  for r in jobs:trainprompts.add(tuple(examples[r['i']]['ids']));manifest.append({k:r[k] for k in ['id','family','split','recipient','donor','i','j','position']})
 assert not trainprompts&testprompts
 del model,mlp,source;torch.cuda.empty_cache();torch.save(dict(anchors=anchor,manifest=manifest),ART/f'{S}_anchors.pt');xx=natural['train_x'].cuda();dd=natural['train_delta'].cuda();yy=natural['train_y'].cuda();ee=natural['train_effect'].cuda();mu=initial['mean'];nd=initial['native_den'];ed=initial['effect_den'];ag={fam:{k:v.cuda() for k,v in a.items()} for fam,a in anchor.items()};records=[];curves=[];models={};preds={};train_scores=[]
 for seed in [0,1]:
  for arm in ARMS:
   m=Student(seed,mu).cuda();m.load_state_dict(initial['models']['effects',seed]);opt=torch.optim.Adam(m.parameters(),lr=.003);gen=torch.Generator().manual_seed(94600+seed)
   for step in range(2000):
    ii=torch.randint(len(xx),(128,),generator=gen).cuda();a=m(xx[ii]);b=m(xx[ii]+dd[ii]);base=(a-yy[ii]).square().mean()/nd+(b-a-ee[ii]).square().mean()/ed;al=torch.zeros((),device='cuda')
    if arm!='continued':
     for fam,p in ag.items():
      c=m(p['x']);v=m(p['x']+p['delta'])
      if arm=='anchor_endpoints':al+=.25*((c-p['y']).square().mean()+(v-p['y']-p['effect']).square().mean())/p['native_den']
      else:al+=.5*(v-c-p['effect']).square().mean()/p['effect_den']
    loss=base+al;assert bool(torch.isfinite(loss));opt.zero_grad();loss.backward();opt.step()
    if step%250==0 or step==1999:curves.append(dict(arm=arm,seed=seed,step=step+1,base_loss=float(base.detach()),anchor_loss=float(al.detach())))
   m=m.cpu();models[arm,seed]=m.state_dict()
   with torch.no_grad():
    for fam,p in anchor.items():train_scores.append(dict(arm=arm,seed=seed,family=fam,anchor_effect_error_ratio=float((m(p['x']+p['delta'])-m(p['x'])-p['effect']).norm()/p['effect'].norm())))
    for name,p in natural['panels'].items():
     y=m(p['x']);e=m(p['x']+p['delta'])-y;records.append(dict(arm=arm,seed=seed,domain='natural',panel=name,native_error_ratio=float((y-p['y']).norm()/(p['y']-mu).norm()),effect_error_ratio=float((e-p['effect']).norm()/p['effect'].norm())))
    X=evaluation['inputs'];D=evaluation['deltas'];E=evaluation['true_effects'];p=m(X);pe=m(X+D)-p;preds[arm,seed]=pe
    for fam in ['agreement','sentiment']:
     for kind in ['full','rank1']:
      ix=[n for n,j in enumerate(evaluation['jobs']) if j['family']==fam and j['arm']==kind];records.append(dict(arm=arm,seed=seed,domain=fam,panel=kind,effect_error_ratio=float((pe[ix]-E[ix]).norm()/E[ix].norm())))
   print('DONE',arm,seed,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'anchor_prompt_eval_prompt_disjoint':not bool(trainprompts&testprompts),'anchor_rows':len(manifest),'anchor_pairs':len({r['id'] for r in manifest}),'discovery_only':all(r['split']=='discovery' for r in manifest),'models':len(models),'finite':all(np.isfinite(r['effect_error_ratio']) for r in records),'excluded_discovery_pair_ids':sorted({r['id'] for r in selection['records'] if r['split']=='discovery' and r['from_end']==offset[r['family']] and tuple(examples[r['i']]['ids']) in testprompts}),'local_teacher_evaluations':2*len(manifest),'whole_model_forwards':0,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['finite'];torch.save(dict(models=models,predicted_effects=preds),ART/f'{S}_fits.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,train_scores=train_scores,curves=curves,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(records,train_scores,audits,flush=True)
if __name__=='__main__':main()
