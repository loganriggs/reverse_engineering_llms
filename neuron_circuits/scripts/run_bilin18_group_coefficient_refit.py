"""H729: coefficient refitting and exhaustive group-pair subspace projection."""
import json,time
from pathlib import Path
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_verified_head_sparse import SignedBSF
STEM='bilin18_group_coefficient_refit'
def support_decoders(D,q):
 k=int(q.sum(-1).max());ids=q.topk(k,dim=-1).indices;return D[ids]*q.gather(-1,ids)[:,:,None]
@torch.no_grad()
def project_supports(x,D,bias,q):
 # Double precision SVD gives the orthogonal projection even for singular supports.
 dec=support_decoders(D.double(),q.double());u,s,vh=torch.linalg.svd(dec,full_matrices=False);keep=s>s[:,:1]*1e-10;basis=vh*keep[:,:,None];y=x.double()-bias.double();pred=bias.double()+torch.einsum('nk,nkd->nd',torch.einsum('nd,nkd->nk',y,basis),basis);return pred
@torch.no_grad()
def best_refit_pair(x,D,bias,supports,chunk=256):
 y=x-bias;best=torch.full((len(x),),-torch.inf,device=x.device);idx=torch.zeros(len(x),dtype=torch.long,device=x.device);orth=0.
 for lo in range(0,len(supports),chunk):
  dec=support_decoders(D,supports[lo:lo+chunk]);u,s,vh=torch.linalg.svd(dec,full_matrices=False);keep=s>s[:,:1]*1e-6;basis=vh*keep[:,:,None];coords=y@basis.flatten(0,1).T;scores=coords.reshape(len(y),len(basis),-1).square().sum(-1);v,i=scores.max(-1);new=v>best;best=torch.where(new,v,best);idx=torch.where(new,i+lo,idx)
  gram=basis@basis.transpose(-1,-2);orth=max(orth,float((gram-torch.diag_embed(keep.float())).abs().max()))
 return supports[idx],y.square().sum(-1)-best,orth
@torch.no_grad()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_verified_head_pca_data.pt','fits':ART/'bilin18_verified_head_write_pca_fits.pt','models':ART/'bilin18_verified_head_sparse_captures.pt','memberships':ART/'bilin18_frozen_dictionary_groups_captures.pt','previous':ART/'bilin18_reconstruction_group_gates_captures.pt','sparse':ROOT/'scripts/run_bilin18_verified_head_sparse.py','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);fits=torch.load(paths['fits'],weights_only=False)['fits'];states=torch.load(paths['models'],weights_only=False)['states'];members=torch.load(paths['memberships'],weights_only=False)['memberships'];previous=torch.load(paths['previous'],weights_only=False);docs=previous['documents']
 cfg=dict(experiment='H729',hashes=hashes,documents=docs,design='Same512validationvectors/head asH728, all54memberships and6dictionaries. Orthogonal least-squares projection onto frozen scalarTop8 supports and H728best fixed-coefficient pair supports. Additionally enumerate everydistinctgroup pair, optimize coefficients bySVD projection, choose global bestpair. Encoder,decoder,bias,memberships frozen. Singular supports handled by rankthreshold. Inference uses availableinputactivation, not tasklabels. No fitting across examples.',limitations='Finite reusedvalidationpanel. Global optimum onlywithin fixeddictionary two-groupunion candidate family. ScalarTop8 supports areencoder-selected, not globalbest8atom subsets. More expensive inference; nooutput/semanticclaims.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));records=[];captures={};errors=[];dominance=[];orths=[]
 for site,fit in fits.items():
  p=(data['activations'][site][docs].reshape(-1,128).cuda()-fit['mean'].cuda().float())@fit['write_pca']['encode'].cuda().float()
  for seed in [7241,7242,7243]:
   st=states[site,'bsf1',seed];model=SignedBSF(1).cuda();model.load_state_dict(st['model']);x=p/st['input_scale'];z=model.encoder(x);D=model.decoder;bias=model.bias;den=x.double().square().sum();ranks=z.abs().argsort(-1,descending=True).argsort(-1);scalarq=(ranks<8).float();scalar=project_supports(x,D,bias,scalarq);scalarmse=float((scalar-x).square().sum()/den)
   for key,M in members.items():
    ss,sd,kind,ns=key
    if (ss,sd)!=(site,seed):continue
    M=M.cuda();oldq=previous['reconstructions'][key]['exact_mask'].cuda();fixed=project_supports(x,D,bias,oldq);ij=torch.triu_indices(len(M),len(M),1,device='cuda');supports=torch.maximum(M[ij[0]],M[ij[1]]);q,sse,orth=best_refit_pair(x,D,bias,supports);pred=project_supports(x,D,bias,q);direct=(pred-x).square().sum(-1);norm=x.double().square().sum(-1).clamp_min(1e-12);errors.append(float(((direct-sse).abs()/norm).max()));dominance.append(float(((direct-(fixed-x).square().sum(-1))/norm).max()));orths.append(orth);k=q.sum(-1);matched=project_supports(x,D,bias,(ranks<k[:,None]).float());old=((z*oldq)@D+bias-x).double().square().sum()/den
    rec=dict(site=site,seed=seed,kind=kind,null_seed=ns,scalar_refit_mse=scalarmse,old_fixed_coefficient_mse=float(old),old_support_refit_mse=float((fixed-x).square().sum()/den),best_pair_refit_mse=float(direct.sum()/den),occupancy_matched_scalar_refit_mse=float((matched-x).square().sum()/den),best_pair_mean_unique_atoms=float(k.mean()),same_support_fraction=float((q==oldq).all(-1).float().mean()));records.append(rec);captures[key]=dict(mask=q.cpu(),reconstruction=(pred*st['input_scale']).cpu());print(site,seed,kind,ns,rec,flush=True)
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['best_pair_refit_mse']) for r in records),max_float32_score_vs_double_projection_relative_error=max(errors),max_best_minus_old_support_relative_error=max(dominance),max_basis_orthogonality_error=max(orths),whole_model_forwards=0,test_evaluations=0);print(audits,flush=True);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(errors)<1e-5 and max(dominance)<1e-5 and max(orths)<1e-5
 torch.save(dict(documents=docs,reconstructions=captures),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,audits=audits,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
