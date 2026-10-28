"""H851 task-blind exact preMLP4 natural-data capture and covariance baselines."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_natural_data'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'training':ART/'bilin18_early_bsf_data_captures.pt','validation_source':ART/'bilin18_fresh_early_geometry_panel_captures.pt','partition':ART/'bilin18_operator_state_prediction_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H851','hashes':hashes,'site':'raw preMLP4, after attention, before RMS normalization/MLP','training':'Original2048 H681 training docs x16positions=32768 states. H681 reserved test not evaluated.','validation':'Exact H832512 development document/position selections fromH824, reused for comparable unsupervised selection. Not a fresh validation claim.','fit':'Training globalmean, per-token means, residual/global PCA up to128. No task labels, task activations, or supervised relay directions loaded. Shape=rms normalized rawpreMLP4; common training residual scale.','planned_featurizers':'Same H836 signed scalar48/64, block8 Top8, fixed/learned shared unions,512atoms,2seeds,2400steps; selection by natural validation residualMSE only.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');old=torch.load(paths['training'],weights_only=False);trainids=old['all_token_ids'][:2048].clone();positions=old['positions'];del old;part=torch.load(paths['partition'],weights_only=False);docs=part['validation_documents'];vp=part['validation_positions'];del part;fresh=torch.load(paths['validation_source'],weights_only=False);validids=fresh['all_token_ids'][docs].clone();del fresh;model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 captures={};prefix=0;hookerr=0.
 for split,ids in [('train',trainids),('validation',validids)]:
  out=[]
  for st in range(0,len(ids),16):
   batch=ids[st:st+16].cuda();x0=F.rms_norm(model.embedding(batch),(1152,));x=x0;bus=None
   for l in range(5):
    block=model.blocks[l];raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
    if l==4:
     z=pre[:,positions] if split=='train' else pre[torch.arange(len(batch),device='cuda'),torch.tensor([positions[k] for k in vp[st:st+len(batch)]],device='cuda')];out.append(z.cpu())
    else:x=pre+block.mlp(F.rms_norm(pre,(1152,)))
   prefix+=len(batch)
  captures[split,4]=torch.cat(out).reshape(-1,1152);print('CAPTURED',split,flush=True)
 del model
 tids=trainids[:,positions].flatten();vids=validids[torch.arange(len(validids)),torch.tensor([positions[k] for k in vp])];x=captures['train',4].double().cuda();v=captures['validation',4].double().cuda();eps=torch.finfo(torch.float32).eps;u=x/(x.square().mean(1,keepdim=True)+eps).sqrt();vu=v/(v.square().mean(1,keepdim=True)+eps).sqrt();mu=u.mean(0);unique,inv=torch.unique(tids,return_inverse=True);cnt=torch.bincount(inv).double().cuda();tm=torch.zeros(len(unique),1152,dtype=torch.float64,device='cuda').index_add_(0,inv.cuda(),u)/cnt[:,None];res=u-tm[inv.cuda()];lookup={int(t):k for k,t in enumerate(unique)};vm=torch.stack([tm[lookup[int(t)]] if int(t) in lookup else mu for t in vids]);vr=vu-vm;baselines=[];bases={};orth=0.
 for kind,centered,valid in [('global',u-mu,vu-mu),('residual',res,vr)]:
  ev,B=torch.linalg.eigh(centered.T@centered/len(centered));B=B[:,-128:].flip(1);bases[kind]=B.cpu();orth=max(orth,float((B.T@B-torch.eye(128,device='cuda')).abs().max()))
  for rank in [1,8,48,64,128]:
   q=B[:,:rank];error=(valid-valid@q@q.T).square().sum()/valid.square().sum();baselines.append({'kind':kind,'rank':rank,'validation_relative_MSE':float(error)})
 gen=torch.Generator().manual_seed(836);directions=torch.linalg.qr(torch.randn(1152,8,dtype=torch.float64,generator=gen)).Q.T;fits={4:{'shape_mean':mu.cpu(),'raw_mean':x.mean(0).cpu(),'token_ids':unique,'token_shape_mean':tm.cpu(),'token_counts':cnt.cpu(),'basis':bases['global'],'residual_pca':bases['residual'],'directions':directions,'unseen_validation_tokens':sum(int(t) not in lookup for t in vids)}};audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'PCA_orthogonality_max_abs':orth,'prefix_forwards':prefix,'whole_model_forwards':0,'test_evaluations':0,'task_sources_loaded':False,'finite':bool(torch.isfinite(x).all() and torch.isfinite(v).all())};assert audits['hashes_unchanged'] and orth<1e-10 and audits['finite'];torch.save({'captures':captures,'fits':fits,'train_token_ids':tids,'validation_token_ids':vids,'validation_documents':docs,'validation_positions':vp},ART/f'{S}_captures.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'baselines':baselines,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
