"""H771 matched sample sensitivity-weighted covariance fits, actual validation KL."""
import json,time
from pathlib import Path
from collections import defaultdict
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_l5_refined_output_geometry import forward
from run_bilin18_l5_geometry_output_fidelity import geometry_reconstruct
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_sensitivity_weighted_fits'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','base':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','fisher':ART/'bilin18_l5_refined_output_geometry_captures.pt','probes':ART/'bilin18_l5_disjoint_sensitivity_probes_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);base=torch.load(paths['base'],weights_only=False)['fits']['L5H5'];fish=torch.load(paths['fisher'],weights_only=False)['fits']['fisher_ridge_0.1'];probe=torch.load(paths['probes'],weights_only=False);train=[c for c in probe['contexts'] if c['split']=='train'];val=[c for c in probe['contexts'] if c['split']=='validation'];nt=len(train);weights=probe['gain'][:nt];perm=torch.randperm(nt//2,generator=torch.Generator().manual_seed(7711));shuffled=weights.reshape(-1,2)[perm].reshape(-1);cfg={'experiment':'H771','hashes':hashes,'training_contexts':train,'validation_contexts':val,'budgets':[8,32,64,128],'shrinkage_to_large_train_second_moment':.5,'weight_shuffle_seed':7711,'geometries':['write','fisher_ridge_0.1'],'design':'Same64 training activations, fixed original training mean, equal0.5shrinkage toward32768sample training second moment. Compare unweighted, measured scalar weights from first12 training probes, and document-shuffled weights preserving length pairing. Freeze all fits before output evaluation. Both fixed and adaptive coordinates; original large natural fit references. Evaluate actual replacement KL and observed-token CE on32 validation contexts.','limitations':'Small reused validation panel; probe weights estimated from finite direction bank; shrinkage predetermined not tuned; no reserved test or causal transfer; fixed-mean weighted second moment, not recentered weighted covariance.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 mu=base['mean'];raw=torch.stack([data['activations']['L5H5'][c['doc'],c['length']//8-1] for c in train]).double();large=data['activations']['L5H5'][:2048].reshape(-1,128).double();fits={};algebra={}
 for name,b in [('write',{'mean':mu,**base['write_pca']}),('fisher',fish)]:
  E=b['encode'];B=b['decode'];y=(raw-mu)@E;yl=(large-mu)@E;C=yl.T@yl/len(yl);fits[name+'_large']={'mean':mu,'encode':E,'decode':B}
  for weightname,w in [('unweighted',torch.ones(nt,dtype=torch.double)),('measured',weights),('shuffled',shuffled)]:
   small=y.T@(y*w[:,None])/w.sum();cov=.5*small+.5*C;ev,V=torch.linalg.eigh(cov);V=V.flip(1);fit={'mean':mu,'encode':E@V,'decode':V.T@B};key=name+'_'+weightname;fits[key]=fit;algebra[key]=float((fit['encode']@fit['decode']-torch.eye(128)).abs().max())
 # All fit artifacts are frozen before any model output is inspected.
 torch.save({'fits':fits,'training_weights':weights,'shuffled_weights':shuffled,'config':cfg},ART/f'{STEM}_fits.pt');fit_hash=sha(ART/f'{STEM}_fits.pt');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];fullerrors=[];captureerrors=[];count=0;cache={}
 for ci,c in enumerate(val):
  token=data['all_token_ids'][c['doc'],:c['length']][None].cuda();saved={};native=forward(model,token,capture=saved)[0].cpu().double();head=saved['head'][0].double();count+=1;cache[c['doc'],c['length']]={'native':native,'head':head};captureerrors.append(float((head-data['activations']['L5H5'][c['doc'],c['length']//8-1]).abs().max()));lp=native.log_softmax(-1);p=lp.exp();jobs=[]
  for name,fit in fits.items():
   for k in cfg['budgets']:
    for adaptive in [False,True]:jobs.append((name,k,adaptive,geometry_reconstruct(head[None],fit,k,adaptive)[0]-head))
  for lo in range(0,len(jobs),8):
   bb=jobs[lo:lo+8];delta=torch.stack([r[3] for r in bb]).float().cuda();out=forward(model,token.expand(len(bb),-1),delta).cpu().double();count+=len(bb)
   for i,(name,k,adaptive,error) in enumerate(bb):
    q=out[i].log_softmax(-1);target=int(data['all_token_ids'][c['doc'],c['length']]);records.append({**c,'fit':name,'budget':k,'adaptive':adaptive,'KL':float((p*(lp-q)).sum()),'CE_change':float(lp[target]-q[target]),'write_error_squared':float((error@base['write'].T).square().sum())})
    if k==128:fullerrors.append(float((out[i]-native).abs().max()))
  print('validation',ci+1,len(val),'forwards',count,flush=True)
 groups=defaultdict(list)
 for r in records:groups[r['fit'],r['budget'],r['adaptive']].append(r)
 summary=[{'fit':n,'budget':k,'adaptive':a,'mean_KL':sum(r['KL'] for r in rr)/len(rr),'mean_CE_change':sum(r['CE_change'] for r in rr)/len(rr),'mean_write_error_squared':sum(r['write_error_squared'] for r in rr)/len(rr)} for (n,k,a),rr in groups.items()];audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'fit_hash_unchanged':sha(ART/f'{STEM}_fits.pt')==fit_hash,'full_rank_logit_replay_max_abs':max(fullerrors),'cached_head_replay_max_abs':max(captureerrors),'inverse_max_abs':max(algebra.values()),'weight_effective_sample_size':float(weights.sum().square()/weights.square().sum()),'document_weight_effective_sample_size':float(weights.reshape(-1,2).sum(1).sum().square()/weights.reshape(-1,2).sum(1).square().sum()),'finite':all(torch.isfinite(torch.tensor(r['KL'])) for r in records),'whole_model_forwards':count,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['fit_hash_unchanged'] and audits['finite'] and max(fullerrors)<1e-4 and max(captureerrors)<1e-4 and max(algebra.values())<1e-10
 torch.save(cache,ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'records':records,'summary':summary,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps(audits,indent=2),flush=True)
if __name__=='__main__':main()
