"""H854 distinguish frozen dictionary, support, and coefficient inference limits."""
import json,time
from pathlib import Path
import torch
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_relay_dictionary_oracles'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'fits':ART/'bilin18_relay_natural_featurizers_captures.pt','data':ART/'bilin18_relay_natural_data_captures.pt','tasks':ART/'bilin18_relay_fresh_distance_captures.pt','task_results':ART/'bilin18_relay_fresh_distance.json','previous':ART/'bilin18_relay_natural_transfer.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H854','hashes':hashes,'design':'All frozen10 H852 fits and78H849 pairs. Diagnostic coefficient oracles, no training. Pairwise contrast projected into full512 decoder span, covered-membership span, or union of native donor/recipient active atoms. Also independent per-state least-squares reconstruction on each frozen native support, retaining decoder bias.','fairness':'Full dictionary rank up to512, active-union up to128, per-state support48/64/variable. These are capacity diagnostics, not equal-cost architecture rankings. Covered span excludes inaccessible untrained atoms in learnedshared. Independent per-state LS uses only input/known decoder/support, not task contrast or label; gates remain frozen.','intervention':'Same H853 native normalizedshape + predicted donor-minus-recipient contrast; normalize/restore recipientradius. Exactshape and identity controls. No effect-based filtering.','numerics':'Float64 SVD numerical rank threshold1e-10 times largest singularvalue. Audit projection orthogonality, nesting/reconstruction errors, per-state LS <= learned reconstruction error. Record effective ranks/condition numbers.','scope':'Reused task panel. Oracles diagnose representability, not learned semantic units. Same-token means cancel but encoding/support may be out-of-distribution.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');states=torch.load(paths['fits'],weights_only=False)['states'];natural=torch.load(paths['data'],weights_only=False)['fits'][4];task=torch.load(paths['tasks'],weights_only=False);ex=task['examples'];native=task['native'];b=torch.stack([task['premlp_relay'][i] for i in range(len(ex))]).double();eps=torch.finfo(torch.float32).eps;rad=(b.square().mean(1,keepdim=True)+eps).sqrt();u=b/rad;lookup={int(t):i for i,t in enumerate(natural['token_ids'])};tids=[e['ids'][e['position']] for e in ex];tm=torch.stack([natural['token_shape_mean'][lookup[t]] if t in lookup else natural['shape_mean'] for t in tids]);res=u-tm;index={(e['id'],e['variant']):i for i,e in enumerate(ex)};source=json.loads(paths['task_results'].read_text());jobs=[(r,index[r['id'],r['recipient']],index[r['id'],r['donor']]) for r in source['records'] if r['kind']=='full'];uniquejobs=[z for z in jobs if z[0]['recipient']==0];edits={};geometry=[];rankaudit=[];orth=0.;lsworse=0.;nested=0.
 def basis(D):
  nonlocal orth
  _,sv,v=torch.linalg.svd(D,full_matrices=False);rank=int((sv>sv[0]*1e-10).sum());Q=v[:rank].T;orth=max(orth,float((Q.T@Q-torch.eye(rank)).abs().max()));return Q,rank,float(sv[0]/sv[rank-1])
 for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
  for seed in [8351,8352]:
   name=f'{kind}_{seed}';ck=states[4,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);outs=[m(x) for x in (res/ck['scale']).float().cuda().split(128)];learned=torch.cat([o['reconstruction'] for o in outs]).double().cpu()*ck['scale'];support=torch.cat([o['support'] for o in outs]).bool().cpu();D=(m.dictionary_matrix().T if kind.startswith('shared') else m.decoder).double().cpu();bias=m.bias.double().cpu()*ck['scale'];covered=outs[0]['membership'].bool().any(0).cpu() if kind.startswith('shared') else torch.ones(512,dtype=torch.bool);full,fullrank,fullcond=basis(D);cover,coverrank,covercond=basis(D[covered]);ls=[];cache={}
   for i in range(len(ex)):
    key=tuple(support[i].nonzero().flatten().tolist())
    if key not in cache:cache[key]=basis(D[list(key)])
    Q,rank,cond=cache[key];pred=(res[i]-bias)@Q@Q.T+bias;ls.append(pred);lsworse=max(lsworse,float((res[i]-pred).square().sum()-(res[i]-learned[i]).square().sum()));rankaudit.append({'model':name,'type':'state_support','index':i,'rank':rank,'condition':cond})
   ls=torch.stack(ls);rankaudit.extend([{'model':name,'type':'full','rank':fullrank,'condition':fullcond},{'model':name,'type':'covered','rank':coverrank,'condition':covercond}])
   for r,i,j in uniquejobs:
    delta=res[j]-res[i];mask=support[i]|support[j];key=tuple(mask.nonzero().flatten().tolist())
    if key not in cache:cache[key]=basis(D[list(key)])
    Q,rank,cond=cache[key];variants={'learned':learned[j]-learned[i],'state_ls':ls[j]-ls[i],'union_oracle':delta@Q@Q.T,'covered_oracle':delta@cover@cover.T,'full_oracle':delta@full@full.T};errs={arm:float((delta-pred).square().sum()) for arm,pred in variants.items()};nested=max(nested,errs['full_oracle']-errs['covered_oracle'],errs['covered_oracle']-errs['union_oracle'],errs['union_oracle']-errs['learned']);rankaudit.append({'model':name,'type':'union','id':r['id'],'rank':rank,'condition':cond})
    for arm,pred in variants.items():
     edits[name,arm,i,j]=pred;edits[name,arm,j,i]=-pred;geometry.append({'model':name,'arm':arm,'id':r['id'],'family':r['family'],'gap':r['gap'],'error_squared':errs[arm],'truth_squared':float(delta.square().sum())})
   del m,outs
   print('PROJECTED',name,flush=True)
 assert orth<1e-10 and lsworse<1e-4 and nested<1e-4
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 previous=json.loads(paths['previous'].read_text());old={(r['id'],r['recipient'],r['model']):r for r in previous['records']};records=[];actual=0;ident=0.;replay=0.;shape_ref={};names=sorted({key[0] for key in edits});arms=['learned','state_ls','union_oracle','full_oracle']
 for length in sorted({len(e['ids']) for e in ex}):
  jj=[z for z in jobs if len(ex[z[1]]['ids'])==length]
  settings=[('reference','identity'),('reference','shape')]+[(name,arm) for name in names for arm in arms+(['covered_oracle'] if name.startswith('shared_learned') else [])]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[i]['ids'] for r,i,j in batch],device='cuda');pos=torch.tensor([ex[i]['position'] for r,i,j in batch],device='cuda')
   for name,arm in settings:
    replacement=[]
    for r,i,j in batch:
     if arm=='identity':p=b[i]
     else:shape=u[j] if arm=='shape' else u[i]+edits[name,arm,i,j];p=shape/shape.square().mean().sqrt().clamp_min(1e-30)*(rad[i].square()-eps).sqrt()
     replacement.append(p)
    y=run(model,ids,torch.stack(replacement).cuda(),pos);actual+=len(batch)
    for k,(r,i,j) in enumerate(batch):
     if arm=='identity':ident=max(ident,float((y[k].double().cpu()-native[i]).abs().max()));continue
     base=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;effect=sgn*(value-base)
     if arm=='shape':shape_ref[i]=y[k].double().cpu()
     if arm=='learned':replay=max(replay,abs(effect-old[r['id'],r['recipient'],name]['donor_directed_delta']))
     lp=shape_ref[i].log_softmax(0);records.append({key:r[key] for key in ['id','family','format','gap','source_pair','recipient','donor','native_answer_gap']}|{'model':name,'arm':arm,'donor_directed_delta':effect,'KL_shape_to_model':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
  print('DONE',length,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'basis_orthogonality_max_abs':orth,'per_state_LS_error_increase_max':lsworse,'nested_projection_error_violation_max':nested,'identity_logits_max_abs':ident,'H853_learned_effect_replay_max_abs':replay,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL_shape_to_model']) for r in records)};assert audits['hashes_unchanged'] and max(ident,replay)<1e-4 and audits['finite'];torch.save({'edits':edits},ART/f'{S}_edits.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'geometry':geometry,'ranks':rankaudit,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
