"""H823: token-conditioned full residual Gaussian passed through frozen TopK BSF."""
import json,time
from pathlib import Path
import numpy as np
import torch
from run_bilin18_early_local_geometry import errors,sha
from run_bilin18_early_bsf_extended import BSF
from run_bilin18_diverse_causal_tasks import ROOT,ART
S='bilin18_gate_geometry_control'
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time()
 paths={'data':ART/'bilin18_early_bsf_data_captures.pt','prior':ART/'bilin18_early_local_geometry.json','runner':Path(__file__),'geometry_runner':ROOT/'scripts/run_bilin18_early_local_geometry.py','encoder_runner':ROOT/'scripts/run_bilin18_early_bsf_extended.py'}
 for site in ['residual0','residual1']:paths[site]=ART/'early_bsf_extended'/f'{site}_r16_seed681.pt'
 hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H823','hashes':hashes,'selection':'All21 nondegenerate H822 residual token/block clouds, unchanged.','null':'Gaussian full standardized residual with exact empirical mean and sample covariance of ALL training occurrences of token, before block selection. Use unchanged encoder Top2 block-L2 gates and decoder.','sampling':'Three seeds8230-8232 per cloud. Draw until accepted count equals observed active train+validation counts, cap65536 draws. First accepted train count fits local/global PCA; next accepted validation count scores it. Ungated comparison uses first same-count draws from identical stream.','geometry':'Rank4 affine PCA;32/64 nearest training neighbors; matched observed selected sample counts.','limitations':'Exploratory reused validation. Null sample covariance may be rank-deficient; no shrinkage or independent covariance fit. Gaussian selection is one null, not proof of smoothness/semantics.','numerics':'Double precision covariance SVD and draws; cast to float32 before unchanged encoder. Earlier float32 factor audit failed at9.94e-4, rejected.','whole_model_forwards':0}
 (ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['data'],weights_only=False);old=json.loads(paths['prior'].read_text());tids=data['token_ids'][:2048].flatten();rows=[r for r in old['records'] if r['neighbors']==32];records=[];coverage=[];replay=[];factorerr=[]
 for site in ['residual0','residual1']:
  ck=torch.load(paths[site],weights_only=False);m=BSF(1152,16).cuda();m.load_state_dict(ck['state_dict']);raw=data['activations'][site][:2048].reshape(-1,1152);cached={}
  for row in [r for r in rows if r['site']==site]:
   token=row['token_id'];block=row['block'];nt=row['train'];nv=row['validation'];need=nt+nv
   if token not in cached:
    x=(raw[tids==token].cuda()-ck['mean'].cuda())/ck['scale'].cuda();x=x.double();mu=x.mean(0);xc=x-mu
    # SVD yields Gaussian covariance factor without assuming independent residual coordinates.
    _,sv,vh=torch.linalg.svd(xc,full_matrices=False);factor=vh.T*(sv/np.sqrt(len(x)-1));cached[token]=(mu,factor,len(x))
    # Frobenius covariance identity checked in residual space.
    factorerr.append(float((factor@factor.T-xc.T@xc/(len(x)-1)).norm()/(xc.T@xc/(len(x)-1)).norm()))
   mu,factor,alln=cached[token];D=m.decoder.reshape(144,16,1152)[block].double();q,r=torch.linalg.qr(D.T,mode='reduced')
   for rep in range(3):
    gen=torch.Generator(device='cuda').manual_seed(8230+rep+100*token+100000*int(site[-1]));accepted=[];ungated=[];na=0;nu=0;draws=0
    while na<need and draws<65536:
     n=min(512,65536-draws);xx=torch.randn(n,factor.shape[1],generator=gen,device='cuda',dtype=torch.float64)@factor.T+mu;xx=xx.float();pre=m.encoder(xx).reshape(-1,144,16);ids=pre.square().sum(-1).topk(2,dim=1).indices;gate=(ids==block).any(1);co=pre[:,block].double()@r.T
     if draws==0:
      z,mask=m.encode(xx);replay.append(float((z[:,block]-pre[:,block]*gate[:,None]).abs().max()));assert torch.equal(mask[:,block].bool(),gate)
     if nu<need:ungated.append(co[:need-nu].cpu().numpy());nu+=min(n,need-nu)
     accepted.append(co[gate].cpu().numpy());na+=int(gate.sum());draws+=n
    coverage.append({'site':site,'block':block,'token_id':token,'rep':rep,'all_training_token_occurrences':alln,'required':need,'accepted':na,'draws':draws,'complete':na>=need})
    if na<need:continue
    gated=np.concatenate(accepted)[:need];ung=np.concatenate(ungated)[:need]
    for k in [32,64]:
     base=next(z for z in old['records'] if z['site']==site and z['block']==block and z['token_id']==token and z['neighbors']==k)
     gg,gl=errors(gated[:nt],gated[nt:],k);ug,ul=errors(ung[:nt],ung[nt:],k)
     records.append({'site':site,'block':block,'token_id':token,'rep':rep,'neighbors':k,'train':nt,'validation':nv,'observed_ratio':base['local_over_global_error'],'gated_gaussian_ratio':float(gl.sum()/gg.sum()),'ungated_gaussian_ratio':float(ul.sum()/ug.sum()),'acceptance':na/draws})
    print(site,block,token,rep,na,draws,flush=True)
 summaries=[]
 for site in ['residual0','residual1']:
  for k in [32,64]:
   rs=[r for r in records if r['site']==site and r['neighbors']==k];clouds=[]
   for b,t in sorted({(r['block'],r['token_id']) for r in rs}):
    rr=[r for r in rs if r['block']==b and r['token_id']==t]
    if len(rr)==3:clouds.append((rr[0]['observed_ratio'],np.mean([r['gated_gaussian_ratio'] for r in rr]),np.mean([r['ungated_gaussian_ratio'] for r in rr])))
   if clouds:
    a=np.array(clouds);summaries.append({'site':site,'neighbors':k,'complete_clouds':len(a),'median_observed':float(np.median(a[:,0])),'median_gated_null':float(np.median(a[:,1])),'median_ungated_null':float(np.median(a[:,2])),'observed_better_than_gated_mean':int((a[:,0]<a[:,1]).sum()),'median_gate_ratio_change':float(np.median(a[:,1]-a[:,2]))})
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'encoder_gate_replay_max_abs':max(replay),'gaussian_covariance_factor_max_relative':max(factorerr),'all_coverage_complete':all(r['complete'] for r in coverage),'finite':all(np.isfinite(r['gated_gaussian_ratio']) and np.isfinite(r['ungated_gaussian_ratio']) for r in records),'whole_model_forwards':0,'test_evaluations':0};print(audits,flush=True);assert audits['hashes_unchanged'] and audits['finite'] and max(replay)==0 and max(factorerr)<1e-5
 (ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'coverage':coverage,'summary':summaries,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(summaries,flush=True);print(audits,flush=True)
if __name__=='__main__':main()
