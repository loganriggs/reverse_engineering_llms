"""H912 explicit task-minus-copy operator stability and low-rank diagnostics."""
from run_bilin18_agreement_partner_number import ROOT,ART,SNAPSHOT,sha,load_full_model
from pathlib import Path
import torch,numpy as np,json,time
S='bilin18_context_operator_contrast'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);paths={'runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin','panel':ART/'bilin18_language_transfer_panel_captures.pt',**{'means_'+str(k):ART/f'bilin18_language_operator_stability_l{k}_means.pt' for k in [13,14]}};hashes={k:sha(p) for k,p in paths.items()};cfg=dict(experiment='H912',hashes=hashes,layers=[13,14],ranks=[1,2,4,8,16,32,64,128,256],design='Fixed disjoint-discovery means. Contrast J(taskmean)-J(copymean). Explicit operator SVD and action diagnostics on all96 reused newcountry endpoints. No newfit on evaluation, no layer/rank selection or behavioral fidelity claim. Contexts differ in more than task; contrast not pure taskcause.');(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');panel=torch.load(paths['panel'],weights_only=False);ex=panel['examples'];ii=[i for i,e in enumerate(ex) if e['cohort']=='new'];matched={(e['family'],e['format'],e['pair'],e['order']):i for i,e in enumerate(ex)};model=load_full_model('cuda');rows=[];spectra=[];geometry=[];saved={};identity=0.;svderr=0.
 for layer in cfg['layers']:
  mod=model.blocks[layer].mlp;L,R,D=[m.weight.double() for m in [mod.Left,mod.Right,mod.Down]];means={k:v.cuda() for k,v in torch.load(paths['means_'+str(layer)],weights_only=False).items()};ops={k:D@((R@m)[:,None]*L+(L@m)[:,None]*R) for k,m in means.items()};contrast={split:ops['task_'+split]-ops['copy_'+split] for split in ['first','second']};a,b=contrast['first'],contrast['second'];rec=dict(layer=layer,operator_cosine=float((a*b).sum()/a.norm()/b.norm()),operator_relative_difference=float((a-b).norm()/a.norm()),contrast_to_task_norm={split:float(contrast[split].norm()/ops['task_'+split].norm()) for split in contrast},actions={})
  X=torch.stack([panel['normalized_states'][i][layer] for i in range(len(ex))]).double().cuda();jj=[matched[ex[i]['family'],ex[i]['format'],ex[i]['pair'],1-ex[i]['order']] for i in ii];changes=X[jj]-X[ii]
  for family in ['task','copy']:
   sel=[n for n,i in enumerate(ii) if ex[i]['family']==family];da=changes[sel]@a.T;db=changes[sel]@b.T;rec['actions'][family]=dict(cosine=float((da*db).sum()/da.norm()/db.norm()),relative_difference=float((da-db).norm()/da.norm()))
  geometry.append(rec)
  for split,C in contrast.items():
   diff=means['task_'+split]-means['copy_'+split];ref=D@((R@diff)[:,None]*L+(L@diff)[:,None]*R);identity=max(identity,float((C-ref).norm()/C.norm()));U,s,V=torch.linalg.svd(C,full_matrices=False);svderr=max(svderr,float((C-(U*s)@V).norm()/C.norm()));energy=s.square();cum=energy.cumsum(0)/energy.sum();spectra.append(dict(layer=layer,split=split,rank50=int((cum<.5).sum())+1,rank90=int((cum<.9).sum())+1,rank99=int((cum<.99).sum())+1,singular_values=s.cpu().tolist()));saved[layer,split]=dict(U=U[:,:256].cpu(),s=s[:256].cpu(),V=V[:256].cpu())
   for rank in cfg['ranks']:
    approx=(changes@V[:rank].T*s[:rank])@U[:,:rank].T;exact=changes@C.T
    for family in ['task','copy']:
     sel=[n for n,i in enumerate(ii) if ex[i]['family']==family];rows.append(dict(layer=layer,split=split,rank=rank,family=family,operator_energy_retained=float(cum[rank-1]),action_relative_RMS_error=float((approx[sel]-exact[sel]).norm()/exact[sel].norm())))
  print('DONE',layer,flush=True)
 audits=dict(hashes_unchanged=all(sha(p)==hashes[k] for k,p in paths.items()),Jacobian_linearity_relative=identity,SVD_reconstruction_relative=svderr,whole_model_forwards=0,test_evaluations=0,finite=all(np.isfinite(r['action_relative_RMS_error']) for r in rows));assert audits['hashes_unchanged'] and identity<1e-10 and svderr<1e-10 and audits['finite'];torch.save(saved,ART/f'{S}_svd.pt');(ART/f'{S}.json').write_text(json.dumps(dict(config=cfg,geometry=geometry,spectra=spectra,records=rows,audits=audits,elapsed_seconds=time.time()-start),indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
