"""H953 downstream decomposition of held-out relay responses and frozen students."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_diverse_causal_tasks import measure
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_compact_fresh_transfer'

@torch.inference_mode()
def prepare_students(c,fits):
 from run_bilin18_real_causal_distillation import Student
 students={}; max_replay=0.; max_algebra=0.; replay_inputs=torch.load(ART/'bilin18_distilled_relay_mechanisms_captures.pt',weights_only=False)
 for (lam,seed),state in fits['models'].items():
  m=Student(seed,state['bias']);m.load_state_dict(state)
  direct=m(c['inputs']+c['deltas'])-m(c['inputs'])
  replayed=m(replay_inputs['inputs']+replay_inputs['deltas'])-m(replay_inputs['inputs']);max_replay=max(max_replay,float((replayed-fits['predicted_effects'][lam,seed]).abs().max()))
  u=F.rms_norm(c['inputs'],(1152,));v=F.rms_norm(c['inputs']+c['deltas'],(1152,))
  l=F.normalize(m.left,dim=0);r=F.normalize(m.right,dim=0)
  dc=(v@l)*(v@r)-(u@l)*(u@r);full=dc@m.out
  max_algebra=max(max_algebra,float((full-direct).norm()/direct.norm().clamp_min(1e-12)))
  subset=torch.zeros_like(full)
  for family,order in fits['orders'][lam,seed].items():
   ii=[i for i,j in enumerate(c['jobs']) if j['family']==family];ix=order[:8]
   subset[ii]=dc[ii][:,ix]@m.out[ix]
  students[f'lambda={lam}:all64',seed]=direct
  students[f'lambda={lam}:top8',seed]=subset
 assert max_replay==0 and max_algebra<1e-3,(max_replay,max_algebra)
 prepare_students.audit={'checkpoint_replay_max_abs':max_replay,'whole_product_algebra_relative_error':max_algebra,'selection':'H949 discovery-only family-specific greedy order; no subset refitting; all penalties and seeds retained'}
 return students

@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'target_preparation':ROOT/'scripts/prepare_bilin18_compact_fresh_transfer.py','original_replay_inputs':ART/'bilin18_distilled_relay_mechanisms_captures.pt','effects':ART/'bilin18_compact_fresh_transfer_captures.pt','anchored_fits':ART/'bilin18_product_complexity_training_fits.pt','tasks':ART/'bilin18_compact_fresh_contexts_captures.pt','prior':ART/'bilin18_compact_fresh_contexts.json','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H953','hashes':hashes,'design':'All212 fresh H952 full/rank1 relay jobs, including competence failures, grouped by context. Frozen H949 products and discovery orders; no fitting or selection on this panel. At layer4 selected position separately add raw input delta to residual output, exact MLP finite delta, or both. H949 frozen all-64 and discovery-selected top8 product MLP deltas replace exact delta with native MLP baseline retained; test with/without residual delta. Downstream model fully live. Not native MLP replacement.','metrics':'Compare student effect to exact matching path, full distribution KL; quantify residual/MLP/joint effects and downstream nonadditivity. Identity and full/rank1 historical margin replay. All seeds/cells; no fitting or task selection.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
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
 for f in ['agreement','sentiment']:
  for arm in ['full','rank1']:
   rr=[r for r in components if r['family']==f and r['arm']==arm];component_summary.append(dict(family=f,arm=arm,**{k+'_RMS':float(np.mean([r[k]**2 for r in rr])**.5) for k in ['residual','mlp','joint','nonadditivity']}))
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':identity,'historical_joint_margin_max_abs':replay,'whole_model_forwards':actual,'expected_forwards':len(jobs)*len(modes),'finite':all(np.isfinite(r['KL']) and np.isfinite(r['effect_error']) for r in records),'no_refitting':True,'student_preparation':prepare_students.audit,'test_evaluations':0,'local_teacher_evaluations_preparation':424};assert audits['hashes_unchanged'] and identity<1e-4 and replay<1e-4 and actual==audits['expected_forwards'] and audits['finite'];(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,components=components,summary=summary,component_summary=component_summary,competence=prior['competence'],audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(component_summary,summary,audits,flush=True)
if __name__=='__main__':main()
