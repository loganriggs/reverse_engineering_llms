"""H822: decoded same-token local geometry, with Gaussian selection controls."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from run_bilin18_early_bsf_extended import BSF
from run_bilin18_diverse_causal_tasks import ROOT,ART
S='bilin18_early_local_geometry'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def errors(x,y,k,rank=4):
 mu=x.mean(0);_,_,v=np.linalg.svd(x-mu,full_matrices=False);d=y-mu;glob=np.sum((d-(d@v[:rank].T)@v[:rank])**2,axis=1)
 dist=np.sum((y[:,None]-x[None])**2,axis=2);near=np.argsort(dist,axis=1)[:,:k];loc=[]
 for a,ids in zip(y,near):
  xx=x[ids];m=xx.mean(0);_,_,vv=np.linalg.svd(xx-m,full_matrices=False);dd=a-m;loc.append(np.sum((dd-(dd@vv[:rank].T)@vv[:rank])**2))
 return glob,np.array(loc)
@torch.no_grad()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');rng=np.random.default_rng(822)
 paths={'data':ART/'bilin18_early_bsf_data_captures.pt','atlas':ART/'bilin18_bsf_lexical_audit.json','runner':Path(__file__)}
 for s in ['value0','residual0','residual1']:paths[s]=ART/'early_bsf_extended'/f'{s}_r16_seed681.pt'
 hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H822','hashes':hashes,'sites':['value0','residual0','residual1'],'blocks':'All H685 selected blocks, unchanged','split':'First2048 documents train; next256 validation; reserved test untouched. Reused exploratory validation.','geometry':'Orthonormal coordinates of decoded16D block; same-token active occurrences; train>=64,validation>=8; global and local affine PCA rank4, neighbors32 and64 chosen by Euclidean query distance among train only.','null':'Five independent Gaussian train/validation draws per token with empirical train mean/covariance and same counts; identical neighborhood selection. This controls adaptive locality, not every lexical or distributional confound.','degeneracy':'Skip token clouds with train variance / mean squared norm <1e-10; report coverage.','posthoc_control':'After first Gaussian results, add elliptical random directions scaled to each observed train/validation centered radius. This preserves radial magnitudes, not exact covariance; diagnostic only.','inference':'Descriptive item metrics. No semantic or causal manifold claim.'}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['data'],weights_only=False);atlas=json.loads(paths['atlas'].read_text());tids=data['token_ids'][:2304].flatten().numpy();records=[];coverage=[];pointrecords=[];isometry=[]
 for site in cfg['sites']:
  ck=torch.load(paths[site],weights_only=False);model=BSF(1152,16).cuda();model.load_state_dict(ck['state_dict']);chosen=[r['block'] for r in atlas['records'] if r['site']==site];zz=[];mm=[]
  for raw in data['activations'][site][:2304].reshape(-1,1152).split(512):
   z,m=model.encode((raw.cuda()-ck['mean'].cuda())/ck['scale'].cuda());zz.append(z[:,chosen].cpu());mm.append(m[:,chosen].cpu())
  z=torch.cat(zz).double().numpy();mask=torch.cat(mm).bool().numpy();D=model.decoder.detach().cpu().double().numpy().reshape(144,16,1152)
  for bi,block in enumerate(chosen):
   q,r=np.linalg.qr(D[block].T,mode='reduced');coords=z[:,bi]@r.T
   # QR preserves decoded Euclidean geometry including nonorthogonal decoder scaling.
   a=z[:8,bi]@D[block];b=coords[:8]@q.T;isometry.append(float(np.max(np.abs(a-b))))
   trainids=np.flatnonzero(mask[:32768,bi]);validids=np.flatnonzero(mask[32768:,bi])+32768;tokens,counts=np.unique(tids[trainids],return_counts=True);eligible=0;degenerate=0
   for token,count in zip(tokens,counts):
    va=validids[tids[validids]==token]
    if count<64 or len(va)<8:continue
    tr=trainids[tids[trainids]==token];x=coords[tr];y=coords[va];var=np.mean(np.sum((x-x.mean(0))**2,axis=1));rel=var/max(np.mean(np.sum(x*x,axis=1)),1e-30)
    if rel<1e-10:degenerate+=1;continue
    eligible+=1;cov=np.cov(x,rowvar=False);ev,U=np.linalg.eigh(cov);root=U@np.diag(np.sqrt(np.maximum(ev,0)));null={k:[] for k in [32,64]};radial={k:[] for k in [32,64]}
    for rep in range(5):
     xn=rng.normal(size=x.shape)@root.T;yn=rng.normal(size=y.shape)@root.T
     for k in null:
      g,l=errors(xn,yn,k);null[k].append(float(l.sum()/max(g.sum(),1e-30)))
     xn=xn/np.maximum(np.linalg.norm(xn,axis=1,keepdims=True),1e-30)*np.linalg.norm(x-x.mean(0),axis=1,keepdims=True);yn=yn/np.maximum(np.linalg.norm(yn,axis=1,keepdims=True),1e-30)*np.linalg.norm(y-x.mean(0),axis=1,keepdims=True)
     for k in radial:
      g,l=errors(xn,yn,k);radial[k].append(float(l.sum()/max(g.sum(),1e-30)))
    for k in null:
     g,l=errors(x,y,k);ratio=float(l.sum()/max(g.sum(),1e-30));records.append({'site':site,'block':block,'token_id':int(token),'train':len(tr),'validation':len(va),'neighbors':k,'local_over_global_error':ratio,'global_error_over_train_variance':float(g.mean()/var),'local_error_over_train_variance':float(l.mean()/var),'gaussian_ratios':null[k],'radial_ratios':radial[k],'excess_local_gain_over_radial':float(np.mean(radial[k])-ratio),'excess_local_gain_over_gaussian':float(np.mean(null[k])-ratio)})
     for idx,gg,ll in zip(va,g,l):pointrecords.append({'site':site,'block':block,'token_id':int(token),'neighbors':k,'validation_document':int(idx//16-2048),'global_error':float(gg),'local_error':float(ll)})
   coverage.append({'site':site,'block':block,'eligible_tokens':eligible,'degenerate_tokens':degenerate});print(site,block,eligible,degenerate,flush=True)
 summaries=[]
 for site in cfg['sites']:
  for k in [32,64]:
   rows=[r for r in records if r['site']==site and r['neighbors']==k]
   if rows:summaries.append({'site':site,'neighbors':k,'token_block_clouds':len(rows),'median_local_over_global':float(np.median([r['local_over_global_error'] for r in rows])),'median_gaussian_ratio':float(np.median([np.mean(r['gaussian_ratios']) for r in rows])),'median_excess_gain':float(np.median([r['excess_local_gain_over_gaussian'] for r in rows])),'median_radial_ratio':float(np.median([np.mean(r['radial_ratios']) for r in rows])),'clouds_better_than_radial_mean':sum(r['excess_local_gain_over_radial']>0 for r in rows),'clouds_better_than_gaussian_mean':sum(r['excess_local_gain_over_gaussian']>0 for r in rows)})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'decoded_isometry_max_abs':max(isometry),'finite':all(np.isfinite(r['local_over_global_error']) for r in records),'whole_model_forwards':0,'test_evaluations':0,'training_validation_document_disjoint':True};assert audits['hashes_unchanged'] and audits['finite'] and max(isometry)<1e-10
 out={'config':cfg,'records':records,'pointrecords':pointrecords,'coverage':coverage,'summary':summaries,'audits':audits,'elapsed_seconds':time.time()-start};(ART/f'{S}.json').write_text(json.dumps(out,indent=2)+'\n');print(summaries,flush=True);print(audits,flush=True)
if __name__=='__main__':main()
