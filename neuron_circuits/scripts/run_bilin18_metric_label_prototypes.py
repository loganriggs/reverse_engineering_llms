"""H960 downstream decomposition of held-out relay responses and frozen students."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import measure
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_metric_label_prototypes'

@torch.inference_mode()
def prepare_students(c,fits):
 graph=json.loads((ART/'bilin18_fisher_state_consistency.json').read_text());records=[r for r in graph['state_potentials'] if r['radius']==1 and r['path']=='shared'];targets={(r['seed'],r['template'],r['source_pair']):torch.tensor(r['states'],dtype=torch.double)*r['common_cap_scale'] for r in records}
 H=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=torch.double);interaction=torch.tensor([1.,-1.,-1.,1.],dtype=torch.double);students={};models=[];scores=[];center=0.;ortho=0.;closure=0.
 for seed in [0,1]:
  state=fits['models'][.1,seed];ix=fits['orders'][.1,seed]['agreement'][:8];W=state['out'][ix].double();W=W/W.norm(dim=1,keepdim=True);_,_,V=torch.linalg.svd(W,full_matrices=False);B=V.T
  Y=torch.stack([targets[seed,j['template'],j['source_pair']][j['donor']]-targets[seed,j['template'],j['source_pair']][j['recipient']] for j in c['jobs']])
  for protocol,folds in [('lexical',list(range(7))),('template',['prepositional','relative_clause'])]:
   for conditioning in (['template','pooled'] if protocol=='lexical' else ['pooled']):
    for kind in ['full','additive']:
     pred=torch.zeros_like(Y);assigned=torch.zeros(len(Y),dtype=torch.int64)
     for heldout in folds:
      key='source_pair' if protocol=='lexical' else 'template';train=[n for n,j in enumerate(c['jobs']) if j[key]!=heldout];test=[n for n,j in enumerate(c['jobs']) if j[key]==heldout]
      trstates={c['jobs'][n][k] for n in train for k in ['index','donor_index']};teststates={c['jobs'][n][k] for n in test for k in ['index','donor_index']};assert not trstates&teststates
      prototypes={}
      for template in sorted({c['jobs'][n]['template'] for n in test}):
       training_quartets=sorted({(c['jobs'][n]['template'],c['jobs'][n]['source_pair']) for n in train if conditioning=='pooled' or c['jobs'][n]['template']==template});assert training_quartets
       Z=torch.stack([targets[seed,t,source] for t,source in training_quartets]).mean(0);Z-=Z.mean(0);add=H@(H.T@Z)/4;inter=interaction[:,None]*(interaction@Z)[None,:]/4
       center=max(center,float(Z.sum(0).abs().max()));ortho=max(ortho,float((Z-add-inter).abs().max()))
       chosen=Z if kind=='full' else add;prototypes[template]=chosen
       models.append(dict(seed=seed,protocol=protocol,conditioning=conditioning,kind=kind,heldout=heldout,evaluation_template=template,training_quartets=training_quartets,states=chosen,interaction_fraction=float(inter.square().sum()/Z.square().sum())))
      for n in test:
       j=c['jobs'][n];Z=prototypes[j['template']];pred[n]=Z[j['donor']]-Z[j['recipient']];assigned[n]+=1
      scores.append(dict(seed=seed,protocol=protocol,conditioning=conditioning,kind=kind,heldout=heldout,coefficient_error=float((pred[test]-Y[test]).norm()/Y[test].norm())))
     assert bool((assigned==1).all());mode=f'{protocol}:{conditioning}:{kind}';students[mode,seed]=(pred@B.T).float()
     for template in ['prepositional','relative_clause']:
      for source in range(7):
       ii=[n for n,j in enumerate(c['jobs']) if j['template']==template and j['source_pair']==source];D=torch.zeros(len(ii),4,dtype=torch.double)
       for k,n in enumerate(ii):D[k,c['jobs'][n]['recipient']]=-1;D[k,c['jobs'][n]['donor']]=1
       closure=max(closure,float((D@torch.linalg.pinv(D)@pred[ii]-pred[ii]).norm()/pred[ii].norm()))
 assert center<1e-9 and ortho<1e-9 and closure<1e-10
 prepare_students.audit={'centered_state_sum_max_abs':center,'additive_interaction_decomposition_max_abs':ortho,'within_quartet_consistency_relative_error':closure,'training_test_endpoint_disjoint':True,'heldout_targets_used_for_prediction':False,'scope':'Predictions use explicit factorial number labels and template (when present in training), not activations. Heldout targets used only for scoring; no teacher norm clipping. Pooled prototype supplies unseen-template control.'}
 prepare_students.folds=scores
 torch.save(dict(models=models,predicted_effects=students),ART/f'{S}_fits.pt')
 return students

@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'graph_targets':ART/'bilin18_fisher_state_consistency.json','target_preparation':ROOT/'scripts/run_bilin18_factorial_number_contexts.py','original_replay_inputs':ART/'bilin18_distilled_relay_mechanisms_captures.pt','effects':ART/'bilin18_factorial_number_contexts_effects.pt','anchored_fits':ART/'bilin18_product_complexity_training_fits.pt','tasks':ART/'bilin18_factorial_number_contexts_captures.pt','prior':ART/'bilin18_factorial_number_contexts.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H960','hashes':hashes,'design':'Same H959 lexical and template holdouts; average training quartet H958 shared budget1 state coordinates. Lexical controls template-conditioned or pooled, each full four-state versus additive subject+distractor state effects. Template holdout uses pooled states from training template. Frozen .1/top8 outputspan, bothseeds, allpredictions live MLP/joint. No activation inputs, no heldout target or norm use.','limits':'Explicit semantic labels and template IDs are privileged controls, not discovered features. Additive/interaction refers to state coefficients on this factorial panel, not unique mechanisms of the original model. No downstream-based selection.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
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
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':identity,'historical_joint_margin_max_abs':replay,'whole_model_forwards':actual,'expected_forwards':len(jobs)*len(modes),'finite':all(np.isfinite(r['KL']) and np.isfinite(r['effect_error']) for r in records),'label_prototypes_fitted':True,'backbone_and_span_frozen':True,'student_preparation':prepare_students.audit,'test_evaluations':0,'local_teacher_evaluations_preparation':prior['audits']['local_teacher_evaluations']};assert audits['hashes_unchanged'] and identity<1e-4 and replay<1e-4 and actual==audits['expected_forwards'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,components=components,summary=summary,component_summary=component_summary,competence=prior['competence'],fold_scores=prepare_students.folds,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(component_summary,summary,audits,flush=True)
if __name__=='__main__':main()
