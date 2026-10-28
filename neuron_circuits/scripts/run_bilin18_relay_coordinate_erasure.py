"""H864 frozen component selection, geometric capacity versus encoded causal transfer."""
import json,time
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_coordinate_erasure'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'unions':ART/'bilin18_relay_component_unions_candidates.pt','prior':ART/'bilin18_relay_component_unions.json','tasks':ART/'bilin18_relay_fresh_distance_captures.pt','task_results':ART/'bilin18_relay_fresh_distance.json','axes':ART/'bilin18_relay_contrast_subspaces_fits.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H864','hashes':hashes,'design':'All20 ranked H862 unions at64slots, all5 architectures andbothseeds/discoveryaxes. Native pair-midpoint coordinate erasure in bothrecipient contexts. Add2 supervised discoveryaxes,2 fixed Gaussian rank1 directions and fullraw midpoint controls.','interventions':'Add half of donor-minus-recipient projection, making bothpairedcontexts identical along the targeted coordinate/span. Eraseunion, within-union rank1, complement. Fullraw midpoint equalizes whole relaystate. No normrestoration. Native residual pathways elsewhere remain intact: test localcontrast dependence, not globalnecessity.','metrics':'Both tasks, all156 recipients. Reference full rawrelay effect plus rerun fullunion effect, rank1/complement interaction residual and KL(union||rank1). Reused research tasks; supervised selection, not manifold discovery. No reservedtest.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');candidates=[c for c in torch.load(paths['unions'],weights_only=False) if c['selection']=='ranked' and c['budget']==64];axes=torch.load(paths['axes'],weights_only=False)['bases'];ortho=0.;decomperr=0.;orth_edit=0.;normerr=0.;erasureerr=0.
 for c in candidates:
  q=c['Q'];v=q@(q.T@axes[c['selected_for']][:,0].double());assert float(v.norm())>1e-10;c['axis']=v/v.norm();c['modes']=['union','rank1','complement'];ortho=max(ortho,float((q.T@q-torch.eye(q.shape[1])).abs().max()))
 for fam in ['agreement','sentiment']:
  v=axes[fam][:,0].double();candidates.append({'kind':'supervised_axis','seed':0,'selected_for':fam,'selection':'axis','group':1,'rank':1,'budget':1,'unique_atoms':1,'Q':v[:,None],'axis':v,'modes':['rank1']})
 gen=torch.Generator().manual_seed(864)
 for rep in [0,1]:
  v=torch.randn(1152,dtype=torch.double,generator=gen);v=v/v.norm();candidates.append({'kind':'random_axis','seed':rep,'selected_for':'random','selection':'random','group':1,'rank':1,'budget':1,'unique_atoms':1,'Q':v[:,None],'axis':v,'modes':['rank1']})
 candidates.append({'kind':'full_midpoint','seed':0,'selected_for':'all','selection':'raw','group':1,'rank':1152,'budget':1152,'unique_atoms':1152,'Q':None,'axis':None,'modes':['raw']})
 task=torch.load(paths['tasks'],weights_only=False);ex=task['examples'];native=task['native'];b=torch.stack([task['premlp_relay'][i] for i in range(len(ex))]).double();index={(e['id'],e['variant']):i for i,e in enumerate(ex)};jobs=[(r,index[r['id'],r['recipient']],index[r['id'],r['donor']]) for r in json.loads(paths['task_results'].read_text())['records'] if r['kind']=='full'];assert len(jobs)==156 and len(candidates)==25
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
    delta=torch.stack([b[j]-b[i] for r,i,j in batch]);union=delta if c['Q'] is None else delta@c['Q']@c['Q'].T;axial=torch.zeros_like(delta) if c['axis'] is None else (delta@c['axis'])[:,None]*c['axis'];complement=union-axial;decomperr=max(decomperr,float((union-axial-complement).abs().max()));union_logits=None
    for mode in c['modes']:
     edits=.5*{'union':union,'rank1':axial,'complement':complement,'raw':delta}[mode];remaining=delta-2*edits
     if mode=='raw':err=remaining
     elif mode=='union':err=remaining@c['Q']
     elif mode=='rank1':err=remaining@c['axis']
     else:err=remaining@c['Q']@c['Q'].T-(remaining@c['axis'])[:,None]*c['axis']
     erasureerr=max(erasureerr,float(err.norm()/delta.norm()))
     y=run(model,ids,torch.stack([b[i] for r,i,j in batch]).cuda()+edits.cuda(),pos);actual+=len(batch)
     if union_logits is None:union_logits=y.double().cpu()
     for k,(r,i,j) in enumerate(batch):
      sgn=1 if r['recipient']==1 else -1;effect=sgn*(measure(y[k],ex[i]['groups'])['margin']-measure(native[i],ex[i]['groups'])['margin']);lp=full[i].log_softmax(0);records.append({z:r[z] for z in ['id','family','format','gap','source_pair','recipient','native_answer_gap']}|{z:c[z] for z in ['kind','seed','selected_for','selection','group','rank','budget','unique_atoms']}|{'mode':mode,'candidate':ci,'native_margin':measure(native[i],ex[i]['groups'])['margin'],'erased_margin':measure(y[k],ex[i]['groups'])['margin'],'native_top1_correct':measure(native[i],ex[i]['groups'])['top1'][r['recipient']],'erased_top1_correct':measure(y[k],ex[i]['groups'])['top1'][r['recipient']],'KL_native_to_erased':float((native[i].softmax(0)*(native[i].log_softmax(0)-y[k].double().cpu().log_softmax(0))).sum()),'donor_directed_delta':effect,'full_donor_delta':r['donor_directed_delta'],'edit_norm':float(edits[k].norm()),'raw_difference_norm':float((b[j]-b[i]).norm()),'KL_full_to_component':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum()),'KL_union_to_component':float((union_logits[k].softmax(0)*(union_logits[k].log_softmax(0)-y[k].double().cpu().log_softmax(0))).sum())})
  print('DONE',length,actual,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':ident,'full_reference_error':replay,'orthogonality_max_abs':ortho,'candidates':len(candidates),'decomposition_max_abs':decomperr,'orthogonal_edit_max_abs':orth_edit,'erased_coordinate_relative_error':erasureerr,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['donor_directed_delta']) and np.isfinite(r['KL_full_to_component']) for r in records)};assert audits['hashes_unchanged'] and max(ident,replay)<1e-4 and ortho<1e-10 and audits['finite'] and actual==156*(2+sum(len(c['modes']) for c in candidates)) and max(decomperr,erasureerr)<1e-8;(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
