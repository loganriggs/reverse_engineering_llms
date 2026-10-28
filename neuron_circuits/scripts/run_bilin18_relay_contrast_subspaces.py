"""H848 discovery-only relay contrast compression and cross-format causal evaluation."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_contrast_subspaces'
def run(model,ids,repl,positions):
 x0=F.rms_norm(model.embedding(ids),(1152,));x=x0;bus=None
 for l,block in enumerate(model.blocks):
  raw=block.lambdas[0]*x+block.lambdas[1]*x0;att,bus=block.attn(F.rms_norm(raw,(1152,)),bus);pre=raw+att
  if l==4:pre=pre.clone();pre[torch.arange(len(ids),device=ids.device),positions]=repl.to(pre)
  x=pre+block.mlp(F.rms_norm(pre,(1152,)))
 return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,))[:,-1]).float()/30)
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'data':ART/'bilin18_same_token_context_transfer_captures.pt','positions':ART/'bilin18_context_relay_positions.json','examples':ART/'bilin18_candidate_task_screen_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H848','hashes':hashes,'fit':'Uncentered SVD of8 discovery donor-minus-recipient rawpreMLP4 contrasts per task, one orientation/pair. All8 ranks possible. No transfer differences or effects used in fitting. Rank1/2/4/8 frozen before evaluation.','scope':'Supervised task-contrast compression, not unsupervised featurizer/manifold discovery. H847 selected layer/position using both formats, so transfer split is held out from basis fit but not fresh research confirmation.','controls':'Full rawdonor, identity, own-task projection, own orthogonal complement, other-task projection, three Gaussian random subspace projections, three random directions scaled to own-projection norm (each both signs). Fixed independent random seed848.','evaluation':'All31 aligned pairs, both directions. Patch recipient rawstate + projected donor-minus-recipient difference at single H847 relay position; keep rest native. Record individual margin fidelity and KL(full swap||approx), not just positive answer transfer. Different tasks use same1152-dimensional preMLP4 coordinate system.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');data=torch.load(paths['data'],weights_only=False);caps,native=data['premlp'],data['native'];ex=torch.load(paths['examples'],weights_only=False)['examples'];old=json.loads(paths['positions'].read_text());offset={'agreement':-3,'sentiment':-4};jobs=[r for r in old['records'] if r['from_end']==offset[r['family']]];bases={};singular={};ortho=0.
 for fam in offset:
  fit=[r for r in jobs if r['family']==fam and r['split']=='discovery' and r['recipient']==0];assert len(fit)==8;delta=torch.stack([caps[r['j']][4][r['position']]-caps[r['i']][4][r['position']] for r in fit]).double();u,s,v=torch.linalg.svd(delta,full_matrices=False);bases[fam]=v.T;singular[fam]=s.tolist();ortho=max(ortho,float((v@v.T-torch.eye(8)).abs().max()))
 gen=torch.Generator().manual_seed(848);random={rep:torch.linalg.qr(torch.randn(1152,8,dtype=torch.float64,generator=gen)).Q for rep in range(3)};cross=(bases['agreement'].T@bases['sentiment']).square();torch.save({'bases':bases,'random':random,'singular_values':singular,'fit_ids':{f:[r['id'] for r in jobs if r['family']==f and r['split']=='discovery' and r['recipient']==0] for f in offset}},ART/f'{S}_fits.pt');model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 full={};ident=0.;replay=0.;actual=0
 for length in sorted({len(ex[r['i']]['ids']) for r in jobs}):
  jj=[r for r in jobs if len(ex[r['i']]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[r['i']]['ids'] for r in batch],device='cuda');pos=torch.tensor([r['position'] for r in batch],device='cuda');repl=torch.stack([caps[r['j']][4][r['position']] for r in batch]).cuda();y=run(model,ids,repl,pos);own=torch.stack([caps[r['i']][4][r['position']] for r in batch]).cuda();yy=run(model,ids,own,pos);actual+=2*len(batch)
   for k,r in enumerate(batch):
    full[r['i']]=y[k].double().cpu();ident=max(ident,float((yy[k].double().cpu()-native[r['i']]).abs().max()));sgn=1 if r['recipient']==1 else -1;effect=sgn*(measure(y[k],ex[r['i']]['groups'])['margin']-measure(native[r['i']],ex[r['i']]['groups'])['margin']);replay=max(replay,abs(effect-r['donor_directed_delta']))
 planned=[];geom=[]
 for r in jobs:
  i,j,p,f=r['i'],r['j'],r['position'],r['family'];delta=(caps[j][4][p]-caps[i][4][p]).double();other=next(fam for fam in offset if fam!=f)
  for rank in [1,2,4,8]:
   B=bases[f][:,:rank];projected=delta@B@B.T;geom.append({'id':r['id'],'recipient':r['recipient'],'family':f,'split':r['split'],'rank':rank,'relative_error':float((delta-projected).norm()/delta.norm()),'energy_fraction':float(projected.square().sum()/delta.square().sum())});arms=[('own',0,projected),('complement',0,delta-projected),('other',0,delta@bases[other][:,:rank]@bases[other][:,:rank].T)]
   for rep,Q in random.items():
    arms.append(('random_projection',rep,delta@Q[:,:rank]@Q[:,:rank].T))
    for sign in [-1,1]:arms.append(('random_matched',2*rep+(sign==1),sign*Q[:,0]*projected.norm()))
   for kind,rep,edit in arms:planned.append((r,rank,kind,rep,edit))
 records=[]
 for length in sorted({len(ex[r['i']]['ids']) for r in jobs}):
  jj=[v for v in planned if len(ex[v[0]['i']]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[r['i']]['ids'] for r,rank,kind,rep,edit in batch],device='cuda');pos=torch.tensor([r['position'] for r,rank,kind,rep,edit in batch],device='cuda');repl=torch.stack([caps[r['i']][4][r['position']].double()+edit for r,rank,kind,rep,edit in batch]).cuda();y=run(model,ids,repl,pos);actual+=len(batch)
   for k,(r,rank,kind,rep,edit) in enumerate(batch):
    i=r['i'];b=measure(native[i],ex[i]['groups'])['margin'];v=measure(y[k],ex[i]['groups'])['margin'];fm=measure(full[i],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;lp=full[i].log_softmax(0);records.append({key:r[key] for key in ['id','family','split','recipient','donor','i','j','position','native_answer_gap']}|{'rank':rank,'kind':kind,'rep':rep,'edit_norm':float(edit.norm()),'donor_directed_delta':sgn*(v-b),'full_donor_delta':sgn*(fm-b),'KL_full_to_approx':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
  print('DONE',length,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'basis_orthogonality_max_abs':ortho,'identity_logits_max_abs':ident,'H847_repeat_margin_max_abs':replay,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL_full_to_approx']) for r in records)};assert audits['hashes_unchanged'] and max(ident,replay)<1e-4 and ortho<1e-10 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'geometry':geom,'cross_basis_squared_overlaps':cross.tolist(),'singular_values':singular,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
