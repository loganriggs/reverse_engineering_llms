"""H856 active-union raw contrast calibration without task directions."""
import json,time
from pathlib import Path
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_norm_calibration'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'fits':ART/'bilin18_relay_natural_featurizers_captures.pt','data':ART/'bilin18_relay_natural_data_captures.pt','tasks':ART/'bilin18_relay_fresh_distance_captures.pt','task_results':ART/'bilin18_relay_fresh_distance.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H856','hashes':hashes,'design':'All10 frozen fits and78 H849 pairs. Active-union Euclidean projection p of true rawpreMLP4 donor contrast delta. Compare p, p*norm(delta)/norm(p), and fixed2*p. No task-coordinate file loaded.','scope':'Task-direction-free pairwise oracle calibration, not a single-state encoder or new learned featurizer. Requires true donor contrast and both native supports. Fixedgain2 chosen prospectively for this followup, not optimized per task.','controls':'Identity/full rawdonor. Preserve gates, subspace and direction across arms; only magnitude changes. No effect-based filtering. Reused exploratory taskpanel.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');states=torch.load(paths['fits'],weights_only=False)['states'];natural=torch.load(paths['data'],weights_only=False)['fits'][4];task=torch.load(paths['tasks'],weights_only=False);ex=task['examples'];native=task['native'];b=torch.stack([task['premlp_relay'][i] for i in range(len(ex))]).double();eps=torch.finfo(torch.float32).eps;u=b/(b.square().mean(1,keepdim=True)+eps).sqrt();lookup={int(t):i for i,t in enumerate(natural['token_ids'])};tids=[e['ids'][e['position']] for e in ex];tm=torch.stack([natural['token_shape_mean'][lookup[t]] if t in lookup else natural['shape_mean'] for t in tids]);res=u-tm;index={(e['id'],e['variant']):i for i,e in enumerate(ex)};source=json.loads(paths['task_results'].read_text());jobs=[(r,index[r['id'],r['recipient']],index[r['id'],r['donor']]) for r in source['records'] if r['kind']=='full'];one=[z for z in jobs if z[0]['recipient']==0];edits={};geom=[];ortherr=0.;coeferr=0.;spanerr=0.;minenergy=1.;conditionmax=0.
 for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
  for seed in [8351,8352]:
   name=f'{kind}_{seed}';ck=states[4,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);outs=[m(x) for x in (res/ck['scale']).float().cuda().split(128)];support=torch.cat([o['support'] for o in outs]).bool().cpu();D=(m.dictionary_matrix().T if kind.startswith('shared') else m.decoder).double().cpu();cache={}
   for r,i,j in one:
    mask=support[i]|support[j];key=tuple(mask.nonzero().flatten().tolist())
    if key not in cache:
     _,sv,v=torch.linalg.svd(D[list(key)],full_matrices=False);rank=int((sv>sv[0]*1e-10).sum());Q=v[:rank].T;cache[key]=Q;ortherr=max(ortherr,float((Q.T@Q-torch.eye(rank)).abs().max()));conditionmax=max(conditionmax,float(sv[0]/sv[rank-1]))
    Q=cache[key];delta=b[j]-b[i];p=delta@Q@Q.T;variants={'euclidean':p,'norm_restored':p*(delta.norm()/p.norm()),'fixed_gain2':2*p}
    for arm,edit in variants.items():
     spanerr=max(spanerr,float((edit-edit@Q@Q.T).norm()/edit.norm().clamp_min(1e-30)));edits[name,arm,i,j]=edit;edits[name,arm,j,i]=-edit;geom.append({'model':name,'arm':arm,'id':r['id'],'family':r['family'],'format':r['format'],'gap':r['gap'],'rank':Q.shape[1],'edit_norm_ratio':float(edit.norm()/delta.norm()),'geometry_relative_error':float((edit-delta).norm()/delta.norm())})
   del m,outs
   print('PROJECTED',name,flush=True)
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 names=sorted({k[0] for k in edits});arms=['euclidean','norm_restored','fixed_gain2'];records=[];actual=0;ident=0.;replay=0.;fullref={}
 for length in sorted({len(e['ids']) for e in ex}):
  jj=[z for z in jobs if len(ex[z[1]]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[i]['ids'] for r,i,j in batch],device='cuda');pos=torch.tensor([ex[i]['position'] for r,i,j in batch],device='cuda');settings=[('reference','identity'),('reference','full')]+[(n,a) for n in names for a in arms]
   for name,arm in settings:
    replacement=[]
    for r,i,j in batch:
     delta=b[j]-b[i]
     edit=torch.zeros_like(delta) if arm=='identity' else delta if arm=='full' else edits[name,arm,i,j];replacement.append(b[i]+edit)
    y=run(model,ids,torch.stack(replacement).cuda(),pos);actual+=len(batch)
    for k,(r,i,j) in enumerate(batch):
     if arm=='identity':ident=max(ident,float((y[k].double().cpu()-native[i]).abs().max()));continue
     baseline=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;effect=sgn*(value-baseline)
     if arm=='full':fullref[i]=y[k].double().cpu();replay=max(replay,abs(effect-r['donor_directed_delta']))
     lp=fullref[i].log_softmax(0);records.append({key:r[key] for key in ['id','family','format','gap','source_pair','recipient','donor','native_answer_gap']}|{'model':name,'arm':arm,'donor_directed_delta':effect,'KL_full_to_edit':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
  print('DONE',length,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'basis_orthogonality_max_abs':ortherr,'span_relative_error':spanerr,'maximum_active_span_condition':conditionmax,'identity_logits_max_abs':ident,'H849_full_margin_replay_max_abs':replay,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL_full_to_edit']) for r in records)};assert audits['hashes_unchanged'] and max(ident,replay)<1e-4 and max(ortherr,coeferr,spanerr)<1e-10 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'geometry':geom,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
