"""H959 downstream decomposition of held-out relay responses and frozen students."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import measure
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_metric_coordinate_transfer'

@torch.inference_mode()
def prepare_students(c,fits):
 states=torch.load(ART/'bilin18_factorial_number_contexts_captures.pt',weights_only=False)
 graph=json.loads((ART/'bilin18_fisher_state_consistency.json').read_text())
 records=[r for r in graph['state_potentials'] if r['radius']==1 and r['path']=='shared'];targets={(r['seed'],r['template'],r['source_pair']):torch.tensor(r['states'],dtype=torch.double)*r['common_cap_scale'] for r in records}
 raw=torch.stack([states['premlp_relay'][i] for i in range(len(states['examples']))]);u=F.rms_norm(raw,(1152,));students={};models=[];foldscores=[];predictions={};stationarity=0.;consistency=0.;seen=set()
 for seed in [0,1]:
  state=fits['models'][.1,seed];ix=fits['orders'][.1,seed]['agreement'][:8];O=state['out'].double();W=O[ix]/O[ix].norm(dim=1,keepdim=True);_,_,V=torch.linalg.svd(W,full_matrices=False);B=V.T
  L=F.normalize(state['left'],dim=0);R=F.normalize(state['right'],dim=0);codes=(u@L)*(u@R)
  rec=[j['index'] for j in c['jobs']];don=[j['donor_index'] for j in c['jobs']];Y=torch.stack([targets[seed,j['template'],j['source_pair']][j['donor']]-targets[seed,j['template'],j['source_pair']][j['recipient']] for j in c['jobs']])
  students['original8',seed]=((codes[don][:,ix]-codes[rec][:,ix]).double()@O[ix]).float();students['state_oracle8',seed]=(Y@B.T).float()
  for feature,Phi in [('normalized_state',u.double()),('bilinear64',codes.double())]:
   X=Phi[don]-Phi[rec]
   for protocol,folds in [('lexical',list(range(7))),('template',['prepositional','relative_clause'])]:
    pred=torch.zeros_like(Y);assigned=torch.zeros(len(Y),dtype=torch.int64)
    for heldout in folds:
     key='source_pair' if protocol=='lexical' else 'template';train=[n for n,j in enumerate(c['jobs']) if j[key]!=heldout];test=[n for n,j in enumerate(c['jobs']) if j[key]==heldout]
     trainstates={c['jobs'][n][k] for n in train for k in ['index','donor_index']};teststates={c['jobs'][n][k] for n in test for k in ['index','donor_index']};assert not trainstates&teststates and train and test
     if protocol=='lexical':assert {c['jobs'][n]['source_pair'] for n in train}.isdisjoint({c['jobs'][n]['source_pair'] for n in test})
     scale=X[train].square().sum(1).mean().sqrt();assert scale>0;Xt=X[train]/scale;Yt=Y[train];reg=.001*len(train)
     dual=torch.linalg.solve(Xt@Xt.T+reg*torch.eye(len(train),dtype=torch.double),Yt);coef=Xt.T@dual
     residual=Xt.T@(Xt@coef-Yt)+reg*coef;stationarity=max(stationarity,float(residual.norm()/(Xt.T@Yt).norm().clamp_min(1e-30)))
     pred[test]=(X[test]/scale)@coef;assigned[test]+=1
     models.append(dict(seed=seed,feature=feature,protocol=protocol,heldout=heldout,scale=float(scale),ridge_alpha=.001,weights=coef,train_job_indices=train,test_job_indices=test))
     foldscores.append(dict(seed=seed,feature=feature,protocol=protocol,heldout=heldout,train_edges=len(train),test_edges=len(test),train_coefficient_error=float((Xt@coef-Yt).norm()/Yt.norm()),heldout_coefficient_error=float((pred[test]-Y[test]).norm()/Y[test].norm())))
    assert bool((assigned==1).all());mode=protocol+':'+feature;students[mode,seed]=(pred@B.T).float();predictions[mode,seed]=pred
    for template in ['prepositional','relative_clause']:
     for source in range(7):
      ids=[n for n,j in enumerate(c['jobs']) if j['template']==template and j['source_pair']==source];D=torch.zeros(len(ids),4,dtype=torch.double)
      for k,n in enumerate(ids):D[k,c['jobs'][n]['recipient']]=-1;D[k,c['jobs'][n]['donor']]=1
      consistency=max(consistency,float((D@torch.linalg.pinv(D)@pred[ids]-pred[ids]).norm()/pred[ids].norm().clamp_min(1e-30)))
 assert len(models)==36 and stationarity<1e-9 and consistency<1e-10
 prepare_students.audit={'ridge_normal_equation_relative_error':stationarity,'within_quartet_consistency_relative_error':consistency,'fits':len(models),'complete_endpoint_and_noun_holdout_checks':True,'heldout_teacher_used_for_prediction':False,'teacher_norm_prediction_clipping':False,'scope':'Graph targets are teacher-informed; fold predictions use only training target edges and input activation features. Shared scalar feature scale fitted on training edges. Fixed alpha .001, no heldout model selection. Lexical and template protocols are separate.'}
 prepare_students.folds=foldscores
 torch.save(dict(models=models,predicted_coefficients=predictions,predicted_effects=students),ART/f'{S}_fits.pt')
 return students

@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'graph_targets':ART/'bilin18_fisher_state_consistency.json','target_preparation':ROOT/'scripts/run_bilin18_factorial_number_contexts.py','original_replay_inputs':ART/'bilin18_distilled_relay_mechanisms_captures.pt','effects':ART/'bilin18_factorial_number_contexts_effects.pt','anchored_fits':ART/'bilin18_product_complexity_training_fits.pt','tasks':ART/'bilin18_factorial_number_contexts_captures.pt','prior':ART/'bilin18_factorial_number_contexts.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H959','hashes':hashes,'design':'Ridge encoder on normalized activation or frozen64 bilinear codes predicts differences of H958 budget1 shared consistent state coordinates. Fixed outputspan .1/top8 bothseeds. Seven complete noun-family folds, each withholding both templates; separate two-way template transfer. Scalar training-edge RMS normalization and alpha .001 fixed before evaluation. Original encoder and teacher-informed state oracle as controls. All per-fold predictions replayed live on isolated MLP and joint paths.','limits':'Coefficient supervision derives from teacher outputs. Not unsupervised discovery. Only seven source nouns and two templates; this repeatedly examined development panel is not a pristine confirmation set. No heldout teacher coefficients/norms enter predictions; holdout means the current encoder training split, not pretraining.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 c=torch.load(paths['effects'],weights_only=False);task=torch.load(paths['tasks'],weights_only=False);prior=json.loads(paths['prior'].read_text());ex=task['examples'];jobs=c['jobs'];students=prepare_students(c,torch.load(paths['anchored_fits'],weights_only=False));modes=['identity','residual','mlp','joint']+[(mode,seed,path) for mode,seed in students for path in ['mlp','joint']];model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 def run(ids,positions,addition):
  x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
  for l,block in enumerate(model.blocks):
   raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att;x=pre+block.mlp(F.rms_norm(pre,(1152,)))
   if l==4:x[torch.arange(len(ids),device='cuda'),positions]+=addition
  return (30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30)).double().cpu()
 hist={(r['id'],r['recipient'],'full' if r['kind']=='full' else 'rank1'):r for r in prior['records'] if r['kind']=='full' or (r['kind']=='own' and r['rank']==1)};buckets=defaultdict(list)
 for n,j in enumerate(jobs):buckets[len(ex[j['index']]['ids'])].append(n)
 actual=0;identity=0.;replay=0.;records=[];components=[]
 for length,inds in sorted(buckets.items()):
  for st in range(0,len(inds),16):
   ns=inds[st:st+16];batch=[jobs[n] for n in ns];ids=torch.tensor([ex[j['index']]['ids'] for j in batch],device='cuda');pos=torch.tensor([ex[j['index']]['position'] for j in batch],device='cuda');dd=c['deltas'][ns].cuda();ee=c['true_effects'][ns].cuda();outs={}
   for mode in modes:
    if mode=='identity':a=torch.zeros_like(dd)
    elif mode=='residual':a=dd
    elif mode=='mlp':a=ee
    elif mode=='joint':a=dd+ee
    else:kind,seed,path=mode;a=students[kind,seed][ns].cuda()+(dd if path=='joint' else 0.)
    outs[mode]=run(ids,pos,a);actual+=len(ns)
   for k,(n,job) in enumerate(zip(ns,batch)):
    g=ex[job['index']]['groups'];base=outs['identity'][k];margin=lambda z:measure(z,g)['margin'];b=margin(base);sgn=1 if job['recipient']==1 else -1;identity=max(identity,float((base-task['native'][job['index']]).abs().max()));joint=margin(outs['joint'][k])-b;replay=max(replay,abs(sgn*joint-hist[job['id'],job['recipient'],job['arm']]['donor_directed_delta']));vals={p:margin(outs[p][k])-b for p in ['residual','mlp','joint']};components.append(job|vals|{'nonadditivity':vals['joint']-vals['mlp']-vals['residual']})
    for kind,seed,path in [m for m in modes if isinstance(m,tuple)]:
     out=outs[kind,seed,path][k];ref=outs[path][k];lp=ref.log_softmax(0);lz=out.log_softmax(0);refeff=margin(ref)-b;eff=margin(out)-b;records.append(job|dict(mode=kind,seed=seed,path=path,reference_effect=refeff,predicted_effect=eff,effect_error=eff-refeff,KL=float((lp.exp()*(lp-lz)).sum()),zero_MLP_KL=float((lp.exp()*(lp-outs['identity' if path=='mlp' else 'residual'][k].log_softmax(0))).sum())))
   print('DONE',length,st,actual,flush=True)
 groups=defaultdict(list)
 for r in records:groups[r['family'],r['format'],r['arm'],r['mode'],r['seed'],r['path']].append(r)
 summary=[]
 for key,rr in groups.items():
  rms=lambda k:float(np.mean([r[k]**2 for r in rr])**.5);summary.append(dict(zip(['family','format','arm','mode','seed','path'],key))|dict(reference_RMS=rms('reference_effect'),effect_RMSE=rms('effect_error'),relative_error=rms('effect_error')/max(rms('reference_effect'),1e-12),KL=float(np.mean([r['KL'] for r in rr])),zero_MLP_KL=float(np.mean([r['zero_MLP_KL'] for r in rr])),KL_ratio=float(sum(r['KL'] for r in rr)/max(sum(r['zero_MLP_KL'] for r in rr),1e-20))))
 component_summary=[]
 for f in sorted({r['family'] for r in components}):
  for arm in sorted({r['arm'] for r in components if r['family']==f}):
   rr=[r for r in components if r['family']==f and r['arm']==arm];component_summary.append(dict(family=f,arm=arm,**{k+'_RMS':float(np.mean([r[k]**2 for r in rr])**.5) for k in ['residual','mlp','joint','nonadditivity']}))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':identity,'historical_joint_margin_max_abs':replay,'whole_model_forwards':actual,'expected_forwards':len(jobs)*len(modes),'finite':all(np.isfinite(r['KL']) and np.isfinite(r['effect_error']) for r in records),'coefficient_encoder_fitted':True,'backbone_and_span_frozen':True,'student_preparation':prepare_students.audit,'test_evaluations':0,'local_teacher_evaluations_preparation':prior['audits']['local_teacher_evaluations']};assert audits['hashes_unchanged'] and identity<1e-4 and replay<1e-4 and actual==audits['expected_forwards'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,components=components,summary=summary,component_summary=component_summary,competence=prior['competence'],fold_scores=prepare_students.folds,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(component_summary,summary,audits,flush=True)
if __name__=='__main__':main()
