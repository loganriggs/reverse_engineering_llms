"""H735: separately trained scalar sparsity capacities with a fixed width."""
import json,time
from pathlib import Path
import torch,numpy as np
from scipy.optimize import linear_sum_assignment
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_coactivation_group_swaps import sha
from run_bilin18_verified_head_sparse import SignedBSF
STEM='bilin18_scalar_capacity_train'
class SignedTopK(SignedBSF):
 def __init__(self,k):
  if not 1<=k<=128:raise ValueError('active count must be 1..128')
  super().__init__(1);self.k=k
 def encode(self,x):
  z=self.encoder(x);ids=z.square().topk(self.k,dim=1).indices;mask=torch.zeros_like(z).scatter_(1,ids,1);return z*mask,mask
@torch.no_grad()
def evaluate(m,x):
 z,mask=m.encode(x);pred=z@m.decoder+m.bias;return dict(validation_write_relative_mse=float((pred-x).square().sum()/x.square().sum()),active_coordinates=float(mask.sum(-1).mean()),dead_atoms=int((mask.sum(0)==0).sum()),decoder_norm_max_error=float((m.decoder.norm(dim=1)-1).abs().max())),mask.cpu()
def main():
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');start=time.time();paths={'data':ART/'bilin18_verified_head_pca_data.pt','fits':ART/'bilin18_verified_head_write_pca_fits.pt','reference_states':ART/'bilin18_verified_head_sparse_captures.pt','reference_results':ART/'bilin18_verified_head_sparse.json','base_implementation':ROOT/'scripts/run_bilin18_verified_head_sparse.py','runner':Path(__file__)};hashes={k:sha(p) for k,p in paths.items()};data=torch.load(paths['data'],weights_only=False);fits=torch.load(paths['fits'],weights_only=False)['fits'];ref=torch.load(paths['reference_states'],weights_only=False)['states'];refj=json.loads(paths['reference_results'].read_text());cfg=dict(experiment='H735',hashes=hashes,seeds=[7241,7242,7243],budgets=[16,32,64],steps=3000,batch=512,learning_rate=.001,design='Same128D writePCAcoordinates/globaltrainRMS scale and128atomdictionary asH724scalar. SignedTopK isSignedBSF1 with configurableactivecount. Separatelytrain K16/32/64 frommatchedseedinitialization andbatchorder, sameAdam/MSE,3000steps,decoderrowrenormalization,validationevery250steps choosebestMSE. ReuseoriginalK8 checkpoints asreference; exactK8implementationreplay checked. Same33024parameters atallK, differentactivecompute. No tasklabels orfunctionalcheckpointselection.',limitations='Reusedvalidationforcheckpointselection; equalsteps notproofconvergence/equaloptimizationdifficulty. HigherK moreactive dimensions, notfairfixedsparsitycomparison. Dictionarywidth128notvaried. Atomstabilitydescriptive, no semantics ormanifoldclaims. Originalreservedtestuntouched.')
 (ROOT/f'configs/{STEM}.json').write_text(json.dumps(cfg,indent=2));dest=ART/'scalar_capacity_train';dest.mkdir(exist_ok=True);records=[];states={};reps={};replay=[]
 for site,fit in fits.items():
  p=(data['activations'][site].reshape(-1,128).cuda()-fit['mean'].cuda().float())@fit['write_pca']['encode'].cuda().float();train=p[:32768];val=p[32768:];scale=train.square().mean().sqrt();x=train/scale;v=val/scale
  for seed in cfg['seeds']:
   state=ref[site,'bsf1',seed];m=SignedTopK(8).cuda();m.load_state_dict(state['model']);old=SignedBSF(1).cuda();old.load_state_dict(state['model'])
   with torch.no_grad():replay.append(float((m(v)-old(v)).abs().max()));metrics,mask=evaluate(m,v)
   rr=next(r for r in refj['records'] if r['site']==site and r['kind']=='bsf1' and r['seed']==seed);assert abs(metrics['validation_write_relative_mse']-rr['validation_write_relative_mse'])<1e-6;records.append(dict(site=site,seed=seed,k=8,reused_reference=True,selected_step=rr['selected_step'],**metrics));states[site,8,seed]=dict(model=state['model'],k=8,input_scale=state['input_scale']);reps[site,8,seed]=(m.decoder.detach().cpu(),mask)
   for k in cfg['budgets']:
    path=dest/f'{site}_k{k}_{seed}.pt'
    if path.exists():
     saved=torch.load(path,weights_only=False);assert saved['input_hashes']==hashes;records.append(saved['record']);states[site,k,seed]=saved['state'];reps[site,k,seed]=saved['representation'];print('REUSED',site,k,seed,flush=True);continue
    torch.manual_seed(seed);gen=torch.Generator(device='cuda').manual_seed(seed+100);m=SignedTopK(k).cuda();opt=torch.optim.Adam(m.parameters(),lr=.001);best=float('inf');history=[]
    for step in range(3001):
     if step%250==0:
      metrics,_=evaluate(m,v);score=metrics['validation_write_relative_mse'];history.append(dict(step=step,mse=score))
      if score<best:best=score;beststep=step;beststate={n:t.detach().cpu().clone() for n,t in m.state_dict().items()}
     if step==3000:break
     ids=torch.randint(len(x),(512,),generator=gen,device='cuda');loss=(m(x[ids])-x[ids]).square().mean();opt.zero_grad();loss.backward();opt.step()
     with torch.no_grad():m.decoder.div_(m.decoder.norm(dim=1,keepdim=True).clamp_min(1e-10))
    m.load_state_dict(beststate);metrics,mask=evaluate(m,v);assert metrics['active_coordinates']==k and metrics['decoder_norm_max_error']<1e-5;record=dict(site=site,seed=seed,k=k,reused_reference=False,selected_step=beststep,history=history,parameters=sum(t.numel() for t in m.parameters()),**metrics);state=dict(model=beststate,k=k,input_scale=float(scale));representation=(m.decoder.detach().cpu(),mask);torch.save(dict(input_hashes=hashes,record=record,state=state,representation=representation),path);records.append(record);states[site,k,seed]=state;reps[site,k,seed]=representation;print('FIT',site,k,seed,'MSE',metrics['validation_write_relative_mse'],'step',beststep,flush=True)
 stability=[]
 for site in fits:
  for k in [8,16,32,64]:
   for sa,sb in [(7241,7242),(7241,7243),(7242,7243)]:
    Da,ma=reps[site,k,sa];Db,mb=reps[site,k,sb];sim=(Da.double()@Db.double().T).abs();ia,ib=linear_sum_assignment(-sim.numpy());aa=ma[:,ia].bool();bb=mb[:,ib].bool();jac=(aa&bb).sum(0)/(aa|bb).sum(0).clamp_min(1);stability.append(dict(site=site,k=k,seeds=[sa,sb],matched_atom_cosine=float(sim[ia,ib].mean()),matched_support_jaccard=float(jac.mean())))
 audits=dict(hashes_unchanged={k:sha(p)==hashes[k] for k,p in paths.items()},finite=all(np.isfinite(r['validation_write_relative_mse']) for r in records),k8_implementation_replay_max_abs=max(replay),active_budgets_correct=all(r['active_coordinates']==r['k'] for r in records),equal_parameter_count=all(r.get('parameters',33024)==33024 for r in records),whole_model_forwards=0,test_evaluations=0);assert all(audits['hashes_unchanged'].values()) and audits['finite'] and max(replay)==0 and audits['active_budgets_correct'] and audits['equal_parameter_count'];torch.save(dict(states=states),ART/f'{STEM}_captures.pt');(ART/f'{STEM}.json').write_text(json.dumps(dict(status='complete',config=cfg,records=records,stability=stability,audits=audits,elapsed_seconds=time.time()-start),indent=2));print(audits,flush=True)
if __name__=='__main__':main()
