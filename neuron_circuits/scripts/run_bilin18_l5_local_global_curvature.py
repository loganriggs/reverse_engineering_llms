"""H767: separate context averaging from finite-step error using frozen directions."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_l5_refined_output_geometry import forward
from run_bilin18_l5_geometry_output_fidelity import geometry_reconstruct
from run_bilin18_l5_natural_featurizer_baselines import reconstruct
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_local_global_curvature'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','base':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','geometry':ART/'bilin18_l5_refined_output_geometry_captures.pt','native':ART/'bilin18_l5_geometry_output_fidelity_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'}
 hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);base=torch.load(paths['base'],weights_only=False)['fits']['L5H5'];geo=torch.load(paths['geometry'],weights_only=False);cached=torch.load(paths['native'],weights_only=False)
 docs=sorted({k[0] for k in cached['native_heads']},key=lambda d:hashlib.sha256(f'H767:{d}'.encode()).hexdigest())[:16];contexts=[(d,n) for d in docs for n in [32,120]];scales=[-.02,-.01,0.,.01,.02,.1,.25,.5,1.]
 cfg={'experiment':'H767','hashes':hashes,'contexts':contexts,'scales':scales,'budgets':[8,64],'design':'Locked16 reused validation docs x2 lengths; adaptive write PCA and both Fisher ridges, budgets8/64. Add alpha times frozen reconstruction error to native L5H5 final head. Estimate directional logit derivative by central differences at .01 and .02; local Fisher is exact vocabulary-probability variance of that numerical derivative. Compare actual KL against local and H764 global quadratic predictions. No fitted parameters or outcome selection.','limitations':'Numerical directional derivative, not analytic JVP. Reused validation documents. Local approximation diagnosed only along six reconstruction errors per context. Global metric has Monte Carlo and context sampling uncertainty.'}
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];derivatives=[];replay=[];count=0
 for ci,(doc,length) in enumerate(contexts):
  raw=cached['native_heads'][doc,length].double()[None];native=cached['native_logits'][doc,length].double();lp=native.log_softmax(-1);p=lp.exp();tokens=data['all_token_ids'][doc,:length]
  for k in cfg['budgets']:
   variants={'adaptive_write_pca':reconstruct(raw,base,'adaptive_write_pca',k)}
   variants.update({name:geometry_reconstruct(raw,fit,k,True) for name,fit in geo['fits'].items()})
   for name,rec in variants.items():
    error=(rec-raw)[0];out=[]
    for lo in range(0,len(scales),4):
     ss=scales[lo:lo+4];delta=torch.stack([s*error for s in ss]).float().cuda();logits=forward(model,tokens[None].expand(len(ss),-1).cuda(),delta).cpu().double();out.extend(logits.unbind());count+=len(ss)
    by=dict(zip(scales,out));replay.append(float((by[0.]-native).abs().max()));lp=by[0.].log_softmax(-1);p=lp.exp();g1=(by[.01]-by[-.01])/.02;g2=(by[.02]-by[-.02])/.04
    def var(g):return float((p*(g-(p*g).sum()).square()).sum())
    local=var(g1);other=var(g2);glob=float(error@geo['metric']@error);derivatives.append({'doc':doc,'length':length,'variant':name,'budget':k,'local_curvature':local,'local_curvature_eps02':other,'global_curvature':glob,'derivative_weighted_relative_difference':(var(g1-g2)/max(local,1e-30))**.5})
    for s in [.01,.02,.1,.25,.5,1.]:
     kl=float((p*(lp-by[s].log_softmax(-1))).sum());records.append({'doc':doc,'length':length,'variant':name,'budget':k,'scale':s,'actual_KL':kl,'local_prediction':.5*s*s*local,'global_prediction':.5*s*s*glob})
  print('context',ci+1,len(contexts),'forwards',count,flush=True)
 groups=defaultdict(list)
 for r in records:groups[r['budget'],r['scale']].append(r)
 summary=[]
 for (k,s),rr in groups.items():
  actual=torch.tensor([r['actual_KL'] for r in rr],dtype=torch.double);row={'budget':k,'scale':s,'mean_actual_KL':float(actual.mean())}
  for kind in ['local','global']:
   pred=torch.tensor([r[kind+'_prediction'] for r in rr],dtype=torch.double);row[kind+'_mean_prediction']=float(pred.mean());row[kind+'_relative_RMS_prediction_error']=float((pred-actual).norm()/actual.norm())
  summary.append(row)
 diff=torch.tensor([r['derivative_weighted_relative_difference'] for r in derivatives]);audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'native_logit_replay_max_abs':max(replay),'derivative_step_agreement_median':float(diff.median()),'derivative_step_agreement_max':float(diff.max()),'finite':all(torch.isfinite(torch.tensor([r['actual_KL'],r['local_prediction'],r['global_prediction']])).all() for r in records),'whole_model_forwards':count,'test_evaluations':0}
 assert audits['hashes_unchanged'] and audits['finite'] and max(replay)<1e-4
 result={'status':'complete','config':cfg,'records':records,'directional_curvature':derivatives,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start};(ART/f'{STEM}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()
