"""H944 known relay mechanisms held out from causal distillation."""
from run_bilin18_binding_role_order_panel import ROOT,ART,sha,SNAPSHOT,load_full_model
from run_bilin18_real_causal_distillation import Student
from pathlib import Path
from collections import defaultdict
import torch,numpy as np,json,time
import torch.nn.functional as F
S='bilin18_distilled_relay_mechanisms'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'runner':Path(__file__),'student':ROOT/'scripts/run_bilin18_real_causal_distillation.py','fits':ART/'bilin18_real_causal_distillation_fits.pt','tasks':ART/'bilin18_relay_fresh_distance_captures.pt','task_results':ART/'bilin18_relay_fresh_distance.json','directions':ART/'bilin18_relay_contrast_subspaces_fits.pt','model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(v) for k,v in paths.items()};cfg={'experiment':'H944','hashes':hashes,'design':'Frozen H943 students, no fitting. Known agreement/sentiment relay tasks and rank1 axes used only for this evaluation; native task competence fromH849. Compare actual MLP finite changes for full raw donor contrast and its own-task rank1 projection. Same token/position recipients. Local update fidelity, not downstream task fidelity. All seeds,formats,gaps retained.','oracles':'Evaluation-only best output-rank1/4/8/16/64 residual floors; true effect projection onto each learned output span. Neither uses deployable coefficients.','limits':'Task panel previously used in broader research; independent of H943 optimization, not a pristine research test. Full preMLP relay edits include a residual route beyond this MLP-only evaluation.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 fits=torch.load(paths['fits'],weights_only=False);tasks=torch.load(paths['tasks'],weights_only=False);results=json.loads(paths['task_results'].read_text());axes=torch.load(paths['directions'],weights_only=False)['bases'];ex=tasks['examples'];caps=tasks['premlp_relay'];lookup={(e['id'],e['variant']):i for i,e in enumerate(ex)};jobs=[];xs=[];ds=[]
 for r in results['records']:
  if r['kind']!='full':continue
  i=lookup[r['id'],r['recipient']];j=lookup[r['id'],r['donor']];assert ex[i]['position']==ex[j]['position'] and ex[i]['ids'][ex[i]['position']]==ex[j]['ids'][ex[j]['position']];x=caps[i].double();d=caps[j].double()-x;q=axes[r['family']][:,0].double()
  for arm,delta in [('full',d),('rank1',q*(d@q))]:
   jobs.append({k:r[k] for k in ['id','family','format','gap','source_pair','recipient','donor']}|dict(arm=arm,index=i));xs.append(x);ds.append(delta)
 X=torch.stack(xs).float();D=torch.stack(ds).float();model=load_full_model('cuda');mlp=model.blocks[4].mlp.float().eval()
 def truth(x):return torch.cat([mlp(F.rms_norm(b.cuda(),(1152,))).cpu() for b in x.split(64)])
 Y=truth(X);E=truth(X+D)-Y;actual=2*len(X);del model,mlp;pred={};records=[];span={};native={}
 for (mode,seed),state in fits['models'].items():
  m=Student(seed,fits['mean']).float();m.load_state_dict(state);p=m(X);pe=m(X+D)-p;pred[mode,seed]=pe;native[mode,seed]=p;_,_,V=torch.linalg.svd(m.out.double(),full_matrices=False);span[mode,seed]=V.T
  for n,job in enumerate(jobs):
   err=pe[n]-E[n];records.append(job|dict(mode=mode,seed=seed,error_sq=float(err.square().sum()),reference_sq=float(E[n].square().sum()),native_error_sq=float((p[n]-Y[n]).square().sum()),native_reference_sq=float((Y[n]-fits['mean']).square().sum())))
 grouped=defaultdict(list)
 for r in records:grouped[r['family'],r['format'],r['gap'],r['arm'],r['mode'],r['seed']].append(r)
 summary=[]
 for key,rr in grouped.items():summary.append(dict(zip(['family','format','gap','arm','mode','seed'],key))|dict(n=len(rr),effect_error_ratio=(sum(r['error_sq'] for r in rr)/sum(r['reference_sq'] for r in rr))**.5,native_error_ratio=(sum(r['native_error_sq'] for r in rr)/sum(r['native_reference_sq'] for r in rr))**.5))
 oracles=[]
 for family in ['agreement','sentiment']:
  for arm in ['full','rank1']:
   ix=[n for n,j in enumerate(jobs) if j['family']==family and j['arm']==arm];Z=E[ix].double();sv=torch.linalg.svdvals(Z);energy=sv.square();den=energy.sum();r90=int(torch.searchsorted(energy.cumsum(0),.9*den))+1
   for rank in [1,4,8,16,64]:oracles.append(dict(family=family,arm=arm,kind='evaluation_rank_floor',rank=rank,error_ratio=float((energy[rank:].sum()/den).sqrt()),rank90=r90))
   for (mode,seed),B in span.items():oracles.append(dict(family=family,arm=arm,kind='learned_span_oracle',mode=mode,seed=seed,error_ratio=float((Z-Z@B@B.T).norm()/Z.norm())))
 audits={'hashes_unchanged':all(sha(v)==hashes[k] for k,v in paths.items()),'finite':all(np.isfinite(r['effect_error_ratio']) for r in summary),'jobs':len(jobs),'local_teacher_evaluations':actual,'whole_model_forwards':0,'test_evaluations':0,'no_refitting':True,'all_competence_cells_pass':all(c['pass'] for c in results['competence'])};assert audits['hashes_unchanged'] and audits['finite'];torch.save(dict(jobs=jobs,inputs=X,deltas=D,native_updates=Y,true_effects=E,predicted_effects=pred,predicted_native=native),ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,records=records,summary=summary,oracles=oracles,competence=results['competence'],audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(oracles,audits,flush=True)
if __name__=='__main__':main()
