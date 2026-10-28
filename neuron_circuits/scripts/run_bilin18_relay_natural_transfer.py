"""H853 frozen exact-site natural featurizers on previously frozen relay tasks."""
import json,time
from pathlib import Path
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_natural_transfer'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'fits':ART/'bilin18_relay_natural_featurizers_captures.pt','data':ART/'bilin18_relay_natural_data_captures.pt','tasks':ART/'bilin18_relay_fresh_distance_captures.pt','task_results':ART/'bilin18_relay_fresh_distance.json','directions':ART/'bilin18_relay_contrast_subspaces_fits.pt','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H853','hashes':hashes,'design':'All10 frozen H852 preMLP4 fits, plus global/residual PCA48/64, on all78 H849 pairs both directions. Model selection used only natural-text developmentMSE. Task activations/directions used only now for evaluation.','intervention':'Native recipient normalizedshape + decoded donor-minus-recipient residual; renormalize and restore native recipient rawRMS. Preserve recipient reconstructionerror, mean cancels at same relaytoken. Compare exact donor shape/recipient radius and full rawdonor.','metrics':'Full task-effect fidelity to exactshape and rawdonor, output KL(shape-reference||model), contrastgeometry and postfit decoder-direction overlap. Overlap alone is not recovery; no best-feature cherry-picking for causal swaps.','scope':'H849 reused tasks, not a fresh confirmation after architecture training. Training and hyperparameters taskblind but research design task-informed. No reserved-test.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');state=torch.load(paths['fits'],weights_only=False)['states'];natural=torch.load(paths['data'],weights_only=False)['fits'][4];task=torch.load(paths['tasks'],weights_only=False);results=json.loads(paths['task_results'].read_text());ex=task['examples'];native=task['native'];caps=task['premlp_relay'];tids=[e['ids'][e['position']] for e in ex];lookup={int(t):i for i,t in enumerate(natural['token_ids'])};b=torch.stack([caps[i] for i in range(len(ex))]).double();eps=torch.finfo(torch.float32).eps;rad=(b.square().mean(1,keepdim=True)+eps).sqrt();u=b/rad;tm=torch.stack([natural['token_shape_mean'][lookup[t]] if t in lookup else natural['shape_mean'] for t in tids]);res=u-tm;recons={};recovery=[];directions=torch.load(paths['directions'],weights_only=False)['bases']
 for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
  for seed in [8351,8352]:
   ck=state[4,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);outs=[m(x) for x in (res/ck['scale']).float().cuda().split(128)];recons[f'{kind}_{seed}']=torch.cat([o['reconstruction'] for o in outs]).double().cpu()*ck['scale']
   if kind.startswith('shared'):
    dictionary=m.dictionary_matrix().T.double();membership=outs[0]['membership'];groups=torch.stack([dictionary[row.bool()] for row in membership])
   else:dictionary=m.decoder.double();groups=dictionary.reshape(-1,8 if kind=='block8' else 1,1152)
   qr=torch.linalg.qr(groups.transpose(1,2),mode='reduced').Q
   for fam in ['agreement','sentiment']:
    q=directions[fam][:,0].cuda();atom=(dictionary@q).square()/dictionary.square().sum(1);span=(qr.transpose(1,2)@q).square().sum(1);recovery.append({'kind':kind,'seed':seed,'family':fam,'max_atom_squared_cosine':float(atom.max()),'max_group_direction_energy':float(span.max()),'best_atom':int(atom.argmax()),'best_group':int(span.argmax())})
   del m,outs
 for typ in ['global','residual']:
  B=natural['basis' if typ=='global' else 'residual_pca']
  for rank in [48,64]:q=B[:,:rank];recons[f'{typ}_pca{rank}']=res@q@q.T
 index={(e['id'],e['variant']):i for i,e in enumerate(ex)};jobs=[]
 for r in results['records']:
  if r['kind']!='full':continue
  i,j=index[r['id'],r['recipient']],index[r['id'],r['donor']];assert tids[i]==tids[j];jobs.append((r,i,j))
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 records=[];geom=[];references={};actual=0;ident=0.;fullerr=0.
 for length in sorted({len(e['ids']) for e in ex}):
  jj=[z for z in jobs if len(ex[z[1]]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[i]['ids'] for r,i,j in batch],device='cuda');pos=torch.tensor([ex[i]['position'] for r,i,j in batch],device='cuda')
   for name in ['identity','raw','shape']+list(recons):
    repl=[]
    for r,i,j in batch:
     if name=='identity':p=b[i]
     elif name=='raw':p=b[j]
     else:
      sh=u[j] if name=='shape' else u[i]+recons[name][j]-recons[name][i];p=sh/sh.square().mean().sqrt().clamp_min(1e-30)*(rad[i].square()-eps).sqrt()
      if name!='shape':
       truth=res[j]-res[i];pred=recons[name][j]-recons[name][i];geom.append({'id':r['id'],'recipient':r['recipient'],'family':r['family'],'gap':r['gap'],'model':name,'error_squared':float((truth-pred).square().sum()),'truth_squared':float(truth.square().sum())})
     repl.append(p)
    y=run(model,ids,torch.stack(repl).cuda(),pos);actual+=len(batch)
    for k,(r,i,j) in enumerate(batch):
     base=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1
     if name=='identity':ident=max(ident,float((y[k].double().cpu()-native[i]).abs().max()));continue
     if name=='raw':fullerr=max(fullerr,abs(sgn*(value-base)-r['donor_directed_delta']))
     if name in ['raw','shape']:references[i,name]=y[k].double().cpu()
     shape_ref=references.get((i,'shape'));kl=None
     if shape_ref is not None:lp=shape_ref.log_softmax(0);kl=float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())
     records.append({k:r[k] for k in ['id','family','format','gap','source_pair','recipient','donor','native_answer_gap']}|{'model':name,'donor_directed_delta':sgn*(value-base),'KL_shape_to_model':kl})
  print('DONE',length,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'identity_logits_max_abs':ident,'raw_reference_margin_max_abs':fullerr,'unseen_relay_token_occurrences':sum(t not in lookup for t in tids),'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['donor_directed_delta']) for r in records)};assert audits['hashes_unchanged'] and max(ident,fullerr)<1e-4 and audits['finite'];(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'geometry':geom,'decoder_recovery':recovery,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
