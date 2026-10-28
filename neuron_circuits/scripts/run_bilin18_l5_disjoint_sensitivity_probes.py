"""H769 disjoint random-probe prediction on separate documents."""
import json,time,hashlib
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_l5_refined_output_geometry import forward
from run_bilin18_l5_natural_featurizer_baselines import reconstruct
from run_bilin18_coactivation_group_swaps import sha
STEM='bilin18_l5_disjoint_sensitivity_probes'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_l5_natural_featurizer_baselines_data.pt','base':ART/'bilin18_l5_natural_featurizer_baselines_fits.pt','old_geometry':ART/'bilin18_l5_refined_output_geometry_captures.pt','old_validation':ART/'bilin18_l5_geometry_output_fidelity_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);base=torch.load(paths['base'],weights_only=False)['fits']['L5H5'];old=torch.load(paths['old_geometry'],weights_only=False);oldval=torch.load(paths['old_validation'],weights_only=False);excluded={r['doc'] for r in old['sample_records']};excludedval={d for d,n in oldval['native_heads']}
 order=lambda ds:sorted(ds,key=lambda d:hashlib.sha256(f'H769:{d}'.encode()).hexdigest());train=order(set(range(2048))-excluded)[:32];val=order(set(range(2048,2304))-excludedval)[:16];contexts=[{'doc':d,'length':n,'split':split} for split,ds in [('train',train),('validation',val)] for d in ds for n in [32,120]]
 raw=data['activations']['L5H5'][train].reshape(-1,128).double();radius=float((reconstruct(raw,base,'adaptive_write_pca',8)-raw).norm(dim=1).median());gen=torch.Generator().manual_seed(7691);probes=torch.randn(24,128,generator=gen,dtype=torch.double);probes=probes/probes.norm(dim=1,keepdim=True)*radius
 cfg={'experiment':'H769','hashes':hashes,'contexts':contexts,'probe_seed':7691,'probe_count':24,'observed_probe_indices':list(range(12)),'target_probe_indices':list(range(12,24)),'probe_radius':radius,'steps':[.01,.02],'ridge':.1,'directional_rank':2,'design':'32train and16validation documents, disjoint from H764 gradient and H765 validation panels respectively. Same24 independent isotropic random directions across96contexts. Observe first12 curvatures to predict remaining12. Fit static per-direction mean, scalar context gain, scalar plus rank2 ridge residual, and scalar plus full ridge residual on training documents only. No validation model selection.','limitations':'Documents already belong to natural research pool. Requires model probes at evaluation time; not an activation-only predictor or featurizer. Fixed small random bank not full metric. Pooled two lengths; no original reserved-test usage.'};(ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2)+'\n');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 allC=[];allD=[];count=0
 for ci,c in enumerate(contexts):
  tokens=data['all_token_ids'][c['doc'],:c['length']][None].expand(5,-1).cuda();cur=[];cur2=[]
  for probe in probes:
   delta=torch.stack([s*probe for s in [-.02,-.01,0.,.01,.02]]).float().cuda();out=forward(model,tokens,delta).cpu().double();count+=5;p=out[2].softmax(-1)
   def var(g):return (p*(g-(p*g).sum()).square()).sum()
   cur.append(var((out[3]-out[1])/.02));cur2.append(var((out[4]-out[0])/.04))
  allC.append(torch.stack(cur));allD.append(torch.stack(cur2));print('context',ci+1,len(contexts),'forwards',count,flush=True)
 C=torch.stack(allC);D=torch.stack(allD);nt=len(train)*2;means=C[:nt].mean(0);gain=(C[:,:12]/means[:12]).mean(1);X=C[:,:12]/means[:12]/gain[:,None]-1;Y=C[:,12:]/means[12:]/gain[:,None]-1;xm=X[:nt].mean(0);ym=Y[:nt].mean(0);xx=X[:nt]-xm;yy=Y[:nt]-ym;coef=torch.linalg.solve(xx.T@xx/nt+.1*torch.eye(12,dtype=torch.double),xx.T@yy/nt);_,_,V=torch.linalg.svd(xx@coef,full_matrices=False);coef2=coef@V[:2].T@V[:2];preds={'static':means[12:].expand(len(C),-1),'scalar':gain[:,None]*means[12:]};clipping={}
 for name,W in [('scalar_rank2',coef2),('scalar_full_ridge',coef)]:
  unclip=1+ym+(X-xm)@W;clipping[name]=int((unclip[nt:]<0).sum());preds[name]=gain[:,None]*means[12:]*unclip.clamp_min(0)
 summary=[];records=[]
 for name,pred in preds.items():
  for split,idx in [('train',slice(0,nt)),('validation',slice(nt,None))]:
   a=C[idx,12:];p=pred[idx];summary.append({'method':name,'split':split,'relative_RMS_error':float((p-a).norm()/a.norm()),'median_relative_error':float(((p-a).abs()/a).median()),'mean_actual_curvature':float(a.mean()),'mean_predicted_curvature':float(p.mean())})
  for i,c in enumerate(contexts):records.append({**c,'method':name,'actual':C[i,12:].tolist(),'prediction':pred[i].tolist()})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'document_splits_disjoint':not bool(set(train)&set(val)),'fresh_relative_to_previous_gradient_and_validation_panels':not bool(set(train)&excluded or set(val)&excludedval),'finite':bool(torch.isfinite(C).all() and torch.isfinite(D).all()),'curvature_step_relative_RMS':float((C-D).norm()/D.norm()),'median_step_relative_error':float(((C-D).abs()/D).median()),'validation_negative_residual_factors_clipped':clipping,'whole_model_forwards':count,'test_evaluations':0};assert audits['hashes_unchanged'] and audits['document_splits_disjoint'] and audits['fresh_relative_to_previous_gradient_and_validation_panels'] and audits['finite']
 torch.save({'curvatures':C,'curvatures_eps02':D,'probes':probes,'contexts':contexts,'means':means,'gain':gain,'coef':coef,'coef_rank2':coef2,'xm':xm,'ym':ym,'predictions':preds},ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps({'status':'complete','config':cfg,'summary':summary,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(json.dumps({'summary':summary,'audits':audits},indent=2),flush=True)
if __name__=='__main__':main()
