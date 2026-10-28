"""H862 frozen component selection, geometric capacity versus encoded causal transfer."""
import json,time
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_component_unions'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'fits':ART/'bilin18_relay_natural_featurizers_captures.pt','means':ART/'bilin18_relay_natural_data_captures.pt','tasks':ART/'bilin18_relay_fresh_distance_captures.pt','task_results':ART/'bilin18_relay_fresh_distance.json','axes':ART/'bilin18_relay_contrast_subspaces_fits.pt','matches':ART/'bilin18_relay_seed_manifolds_captures.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H862','hashes':hashes,'selection':'All5 architectures and2 seeds. Nested groups ranked by discovery-axis overlap, nominal atom-slot budgets8/32/64. A fixed random group ordering per fit and task gives nested controls. Shared unions have fewer unique atoms; actual rank/atomcounts recorded, not claimed exactly dimension matched. Rank1 supervised axis controls included.','interventions':'Projection onto union span; encoded active selected-group union decoded once per atom, scaled by native rawRMS per input. Random projection also rescaled to own ranked-projection edit norm, explicitly pairwise oracle control. No norm calibration of learned encodings. Supervised rank1 axis uses projection only.','evaluation':'All78 aligned H849 pairs both directions, agreement/sentiment formats and gaps. Full raw donor and identity controls. Task selection is discovery-supervised and tasks broadly reused; not unsupervised semantic recovery. Random groups rank matched but not norm matched; record edit norms.','question':'Does a geometrically/behaviorally stable component carry task-selective causal effects, or is signal distributed/encoder-limited?'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 states=torch.load(paths['fits'],weights_only=False)['states'];f=torch.load(paths['means'],weights_only=False)['fits'][4];task=torch.load(paths['tasks'],weights_only=False);ex=task['examples'];native=task['native'];b=torch.stack([task['premlp_relay'][i] for i in range(len(ex))]).double();rad=(b.square().mean(1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();lookup={int(t):i for i,t in enumerate(f['token_ids'])};tids=[e['ids'][e['position']] for e in ex];tm=torch.stack([f['token_shape_mean'][lookup[t]] if t in lookup else f['shape_mean'] for t in tids]);res=b/rad-tm;axes=torch.load(paths['axes'],weights_only=False)['bases'];matches=torch.load(paths['matches'],weights_only=False);rng=np.random.default_rng(862);candidates=[];ortho=0.;unionerr=0.;monotonicerr=0.
 for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
  for seed in [8351,8352]:
   ck=states[4,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);outs=[m(x) for x in (res/ck['scale']).float().cuda().split(128)];code=torch.cat([o['code'] for o in outs]).double().cpu();D=(m.dictionary_matrix().T if kind.startswith('shared') else m.decoder).double().cpu()
   if kind.startswith('shared'):idx=torch.stack([r.bool().nonzero().flatten().cpu() for r in outs[0]['membership']]);gates=torch.cat([o['gates'] for o in outs]).double().cpu()
   else:idx=torch.arange(512).reshape(-1,8 if kind=='block8' else 1);gates=torch.ones(len(b),len(idx))
   groups=D[idx];_,sv,V=torch.linalg.svd(groups,full_matrices=False);assert bool((sv[:,-1]>sv[:,0]*1e-10).all());Q=V.transpose(1,2);ortho=max(ortho,float((Q.transpose(1,2)@Q-torch.eye(Q.shape[-1])).abs().max()));selections=[]
   for fam in ['agreement','sentiment']:
    energy=(Q.transpose(1,2)@axes[fam][:,0].double()).square().sum(-1);order=torch.argsort(energy,descending=True);random_order=torch.tensor(rng.permutation(len(Q)));previous_energy=-1.
    for budget in [8,32,64]:
     count=budget//idx.shape[1];targetQ=None
     for selection,ordering in [('ranked',order),('random',random_order)]:
      chosen=ordering[:count];atoms=torch.unique(idx[chosen].flatten());_,svv,vv=torch.linalg.svd(D[atoms],full_matrices=False);rank=int((svv>svv[0]*1e-10).sum());qq=vv[:rank].T;ortho=max(ortho,float((qq.T@qq-torch.eye(rank)).abs().max()))
      if selection=='ranked':
       targetQ=qq;axis_energy=float((qq.T@axes[fam][:,0].double()).square().sum());monotonicerr=max(monotonicerr,previous_energy-axis_energy);previous_energy=axis_energy
      mask=torch.zeros(len(b),512,dtype=torch.bool)
      for g in chosen:mask[:,idx[g]] |= gates[:,g,None].bool().expand(-1,idx.shape[1])
      # Explicit union reference verifies no double counting when selected groups overlap.
      explicit=torch.stack([torch.any(torch.stack([(gates[n,g]>0)&torch.isin(torch.arange(512),idx[g]) for g in chosen]),dim=0) for n in range(len(b))]);unionerr=max(unionerr,float((mask!=explicit).sum()));comp=(code*mask)@D*ck['scale']*rad
      candidates.append({'kind':kind,'seed':seed,'selected_for':fam,'selection':selection,'group':count,'rank':rank,'budget':budget,'unique_atoms':len(atoms),'Q':qq,'targetQ':targetQ,'component':comp,'modes':['projection','encoded']+(['norm_random'] if selection=='random' else [])})
   del m,outs
 for fam in ['agreement','sentiment']:
  qq=axes[fam][:,:1].double();candidates.append({'kind':'supervised_axis','seed':0,'selected_for':fam,'selection':'axis','group':1,'rank':1,'budget':1,'unique_atoms':1,'Q':qq,'component':None,'modes':['projection']})
 # Freeze exact selections before observing downstream interventions.
 torch.save(candidates,ART/f'{S}_candidates.pt');index={(e['id'],e['variant']):i for i,e in enumerate(ex)};jobs=[(r,index[r['id'],r['recipient']],index[r['id'],r['donor']]) for r in json.loads(paths['task_results'].read_text())['records'] if r['kind']=='full'];assert len(jobs)==156
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];actual=0;ident=0.;replay=0.;full={}
 for length in sorted({len(e['ids']) for e in ex}):
  jj=[z for z in jobs if len(ex[z[1]]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[i]['ids'] for r,i,j in batch],device='cuda');pos=torch.tensor([ex[i]['position'] for r,i,j in batch],device='cuda')
   for control in ['identity','full']:
    y=run(model,ids,torch.stack([b[i if control=='identity' else j] for r,i,j in batch]).cuda(),pos);actual+=len(batch)
    for k,(r,i,j) in enumerate(batch):
     if control=='identity':ident=max(ident,float((y[k].double().cpu()-native[i]).abs().max()))
     else:
      full[i]=y[k].double().cpu();sgn=1 if r['recipient']==1 else -1;effect=sgn*(measure(y[k],ex[i]['groups'])['margin']-measure(native[i],ex[i]['groups'])['margin']);replay=max(replay,abs(effect-r['donor_directed_delta']))
   for ci,c in enumerate(candidates):
    for mode in c['modes']:
     edits=torch.stack([(b[j]-b[i])@c['Q']@c['Q'].T if mode!='encoded' else c['component'][j]-c['component'][i] for r,i,j in batch])
     if mode=='norm_random':
      targetnorm=torch.stack([((b[j]-b[i])@c['targetQ']).norm() for r,i,j in batch]);assert bool((edits.norm(dim=1)>1e-12).all());edits=edits*targetnorm[:,None]/edits.norm(dim=1,keepdim=True)
     y=run(model,ids,torch.stack([b[i] for r,i,j in batch]).cuda()+edits.cuda(),pos);actual+=len(batch)
     for k,(r,i,j) in enumerate(batch):
      sgn=1 if r['recipient']==1 else -1;effect=sgn*(measure(y[k],ex[i]['groups'])['margin']-measure(native[i],ex[i]['groups'])['margin']);lp=full[i].log_softmax(0);records.append({z:r[z] for z in ['id','family','format','gap','source_pair','recipient','native_answer_gap']}|{z:c[z] for z in ['kind','seed','selected_for','selection','group','rank','budget','unique_atoms']}|{'mode':mode,'candidate':ci,'donor_directed_delta':effect,'full_donor_delta':r['donor_directed_delta'],'edit_norm':float(edits[k].norm()),'raw_difference_norm':float((b[j]-b[i]).norm()),'KL_full_to_component':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
  print('DONE',length,actual,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':ident,'full_reference_error':replay,'orthogonality_max_abs':ortho,'candidates':len(candidates),'union_error':unionerr,'nested_axis_energy_decrease':monotonicerr,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['donor_directed_delta']) and np.isfinite(r['KL_full_to_component']) for r in records)};assert audits['hashes_unchanged'] and max(ident,replay)<1e-4 and ortho<1e-10 and audits['finite'] and actual==156*(2+sum(len(c['modes']) for c in candidates)) and unionerr==0 and monotonicerr<1e-10;(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
