"""H958 downstream decomposition of held-out relay responses and frozen students."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import measure
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_fisher_state_consistency'

@torch.inference_mode()
def prepare_students(c,fits):
 import itertools
 oracle=json.loads((ART/'bilin18_fisher_span_oracle.json').read_text());lookup={(r['index'],r['seed'],r['path'],r['radius']):torch.tensor(r['coefficients'],dtype=torch.double) for r in oracle['coefficients']}
 students={};diagnostics=[];potentials=[];max_projection_error=0.;control=0.;closure=0.;budget=0.;quartets=defaultdict(list)
 for n,j in enumerate(c['jobs']):quartets[j['template'],j['source_pair']].append(n)
 for seed in [0,1]:
  state=fits['models'][.1,seed];ix=fits['orders'][.1,seed]['agreement'][:8];W=state['out'][ix].double();W=W/W.norm(dim=1,keepdim=True);_,_,V=torch.linalg.svd(W,full_matrices=False);B=V.T
  for radius in [1.,2.]:
   labels=['edge_mlp','edge_joint']+[f'{kind}_{path}' for kind in ['state','capped'] for path in ['mlp','joint','shared']]
   effects={label:torch.zeros_like(c['true_effects']) for label in labels}
   for (template,source),ii in quartets.items():
    D=torch.zeros(len(ii),4,dtype=torch.double);pairs={}
    for row,n in enumerate(ii):
     j=c['jobs'][n];a,b=j['recipient'],j['donor'];D[row,a]=-1;D[row,b]=1;pairs[a,b]=row
    assert len(pairs)==12
    P=torch.linalg.pinv(D);truth=c['true_effects'][ii].double();en=truth.norm(dim=1)
    for ctrl in [truth,truth@B]:control=max(control,float((D@P@ctrl-ctrl).norm()/ctrl.norm()))
    edge={path:torch.stack([lookup[n,seed,path,radius] for n in ii]) for path in ['mlp','joint']}
    for path,C in edge.items():
     effects['edge_'+path][ii]=(C@B.T).float();rms=(C.square().sum()/len(C)).sqrt()
     anti=torch.stack([C[pairs[a,b]]+C[pairs[b,a]] for a,b in itertools.combinations(range(4),2)])
     cycle=torch.stack([C[pairs[a,b]]+C[pairs[b,d]]-C[pairs[a,d]] for a,b,d in itertools.permutations(range(4),3)])
     diagnostics.append(dict(seed=seed,radius=radius,template=template,source_pair=source,path=path,reversal_relative_RMS=float((anti.square().sum()/len(anti)).sqrt()/rms),triangle_relative_RMS=float((cycle.square().sum()/len(cycle)).sqrt()/rms),projection_relative_error=float((D@P@C-C).norm()/C.norm())))
    for path in ['mlp','joint','shared']:
     C=edge[path] if path!='shared' else (edge['mlp']+edge['joint'])/2;Z=P@C;fitted=D@Z
     max_projection_error=max(max_projection_error,float((Z-D.T@C/8).abs().max()));closure=max(closure,float((D@P@fitted-fitted).norm()/fitted.norm()))
     scale=min(1.,float((radius*en/fitted.norm(dim=1).clamp_min(1e-30)).min()));bounded=fitted*scale;budget=max(budget,float((bounded.norm(dim=1)/(radius*en)).max()))
     effects['state_'+path][ii]=(fitted@B.T).float();effects['capped_'+path][ii]=(bounded@B.T).float()
     potentials.append(dict(seed=seed,radius=radius,template=template,source_pair=source,path=path,states=Z.tolist(),common_cap_scale=scale,raw_max_norm_over_budget=float((fitted.norm(dim=1)/(radius*en)).max()),path_disagreement_relative_norm=float((edge['mlp']-edge['joint']).norm()/edge['mlp'].norm())))
   for label,eff in effects.items():students[f'r={int(radius)}:{label}',seed]=eff
 assert control<1e-5 and closure<1e-10 and max_projection_error<1e-9 and budget<1+1e-10,(control,closure,max_projection_error,budget)
 prepare_students.audit={'exact_and_euclidean_control_projection_relative_error':control,'consistent_projection_closure_relative_error':closure,'pinv_vs_complete_graph_solution_max_abs':max_projection_error,'capped_norm_max_over_budget':budget,'scope':'Teacher-informed graph potentials fit on all edges within each quartet. Uniform edge coefficient metric, not globally KL-optimal consistent representation. Separate path and shared-path variants; no weight training. Common quartet scaling preserves consistency while enforcing each edge norm bound.'}
 prepare_students.diagnostics=diagnostics;prepare_students.potentials=potentials
 torch.save(dict(predicted_effects=students),ART/f'{S}_effects.pt')
 return students

@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'oracle_coefficients':ART/'bilin18_fisher_span_oracle.json','target_preparation':ROOT/'scripts/run_bilin18_factorial_number_contexts.py','original_replay_inputs':ART/'bilin18_distilled_relay_mechanisms_captures.pt','effects':ART/'bilin18_factorial_number_contexts_effects.pt','anchored_fits':ART/'bilin18_product_complexity_training_fits.pt','tasks':ART/'bilin18_factorial_number_contexts_captures.pt','prior':ART/'bilin18_factorial_number_contexts.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H958','hashes':hashes,'design':'Fit centered four-state potentials to H957 directed-edge oracle coefficients on each complete factorial quartet. Fixed outputspan .1/top8 bothseeds. Separate MLP/joint and shared-path potentials; raw and common-scaled budget-preserving variants, original directed-edge controls. Both original normbudgets retained. Evaluate every variant on both paths, including cross-path transfer. No model training; all coefficients teacher-informed.', 'limits':'A consistency diagnostic on observed states, not an encoder or unseen-input predictor. Uniform coefficient-space graph projection may sacrifice KL unnecessarily; failure is not a lower bound on consistent KL optimum. Input-state additivity is necessary only for fixed linear decoder representations.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
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
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':identity,'historical_joint_margin_max_abs':replay,'whole_model_forwards':actual,'expected_forwards':len(jobs)*len(modes),'finite':all(np.isfinite(r['KL']) and np.isfinite(r['effect_error']) for r in records),'no_model_training':True,'teacher_informed_state_potentials':True,'student_preparation':prepare_students.audit,'test_evaluations':0,'local_teacher_evaluations_preparation':prior['audits']['local_teacher_evaluations']};assert audits['hashes_unchanged'] and identity<1e-4 and replay<1e-4 and actual==audits['expected_forwards'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,components=components,summary=summary,component_summary=component_summary,competence=prior['competence'],consistency_diagnostics=prepare_students.diagnostics,state_potentials=prepare_students.potentials,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(component_summary,summary,audits,flush=True)
if __name__=='__main__':main()
