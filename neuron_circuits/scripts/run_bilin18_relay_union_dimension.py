"""H863 frozen component selection, geometric capacity versus encoded causal transfer."""
import json,time
from pathlib import Path
import numpy as np
import torch
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_union_dimension'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'unions':ART/'bilin18_relay_component_unions_candidates.pt','prior':ART/'bilin18_relay_component_unions.json','tasks':ART/'bilin18_relay_fresh_distance_captures.pt','task_results':ART/'bilin18_relay_fresh_distance.json','axes':ART/'bilin18_relay_contrast_subspaces_fits.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H863','hashes':hashes,'design':'All60 ranked H862 unions, all5 architectures, bothseeds, both discoveryaxes,8/32/64 budgets. Frozen taskaxis projected into union and normalized; no downstream fit.','interventions':'Full union projection, within-union rank1 projection, orthogonal union complement, and rank1 rescaled to full union edit norm. Norm-matched rank1 is pairwise oracle, not a single-input encoding. All rawstate difference transfers preserve recipient remainder.','metrics':'Both tasks, all156 recipients. Reference full rawrelay effect plus rerun fullunion effect, rank1/complement interaction residual and KL(union||rank1). Reused research tasks; supervised selection, not manifold discovery. No reservedtest.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');candidates=[c for c in torch.load(paths['unions'],weights_only=False) if c['selection']=='ranked'];axes=torch.load(paths['axes'],weights_only=False)['bases'];ortho=0.;decomperr=0.;orth_edit=0.;normerr=0.
 for c in candidates:
  q=c['Q'];v=q@(q.T@axes[c['selected_for']][:,0].double());assert float(v.norm())>1e-10;c['axis']=v/v.norm();c['modes']=['union','rank1','complement','rank1_norm'];ortho=max(ortho,float((q.T@q-torch.eye(q.shape[1])).abs().max()))
 task=torch.load(paths['tasks'],weights_only=False);ex=task['examples'];native=task['native'];b=torch.stack([task['premlp_relay'][i] for i in range(len(ex))]).double();index={(e['id'],e['variant']):i for i,e in enumerate(ex)};jobs=[(r,index[r['id'],r['recipient']],index[r['id'],r['donor']]) for r in json.loads(paths['task_results'].read_text())['records'] if r['kind']=='full'];assert len(jobs)==156 and len(candidates)==60
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
    delta=torch.stack([b[j]-b[i] for r,i,j in batch]);union=delta@c['Q']@c['Q'].T;axial=(delta@c['axis'])[:,None]*c['axis'];complement=union-axial;decomperr=max(decomperr,float((union-axial-complement).abs().max()));orth_edit=max(orth_edit,float((complement@c['axis']).abs().max()));union_logits=None
    for mode in c['modes']:
     edits={'union':union,'rank1':axial,'complement':complement}.get(mode)
     if mode=='rank1_norm':
      assert bool((axial.norm(dim=1)>1e-12).all());edits=axial*union.norm(dim=1,keepdim=True)/axial.norm(dim=1,keepdim=True);normerr=max(normerr,float((edits.norm(dim=1)-union.norm(dim=1)).abs().max()))
     y=run(model,ids,torch.stack([b[i] for r,i,j in batch]).cuda()+edits.cuda(),pos);actual+=len(batch)
     if mode=='union':union_logits=y.double().cpu()
     for k,(r,i,j) in enumerate(batch):
      sgn=1 if r['recipient']==1 else -1;effect=sgn*(measure(y[k],ex[i]['groups'])['margin']-measure(native[i],ex[i]['groups'])['margin']);lp=full[i].log_softmax(0);records.append({z:r[z] for z in ['id','family','format','gap','source_pair','recipient','native_answer_gap']}|{z:c[z] for z in ['kind','seed','selected_for','selection','group','rank','budget','unique_atoms']}|{'mode':mode,'candidate':ci,'donor_directed_delta':effect,'full_donor_delta':r['donor_directed_delta'],'edit_norm':float(edits[k].norm()),'raw_difference_norm':float((b[j]-b[i]).norm()),'KL_full_to_component':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum()),'KL_union_to_component':float((union_logits[k].softmax(0)*(union_logits[k].log_softmax(0)-y[k].double().cpu().log_softmax(0))).sum())})
  print('DONE',length,actual,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':ident,'full_reference_error':replay,'orthogonality_max_abs':ortho,'candidates':len(candidates),'decomposition_max_abs':decomperr,'orthogonal_edit_max_abs':orth_edit,'norm_control_max_abs':normerr,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['donor_directed_delta']) and np.isfinite(r['KL_full_to_component']) for r in records)};assert audits['hashes_unchanged'] and max(ident,replay)<1e-4 and ortho<1e-10 and audits['finite'] and actual==156*(2+sum(len(c['modes']) for c in candidates)) and max(decomperr,orth_edit,normerr)<1e-8;(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
