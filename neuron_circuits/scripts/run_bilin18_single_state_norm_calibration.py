"""H857 consistent single-state residual-norm encodings, natural and causal evaluation."""
import json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from run_bilin18_diverse_causal_tasks import ROOT,ART,measure
from run_bilin18_qkov_causal_edges import load_full_model,SNAPSHOT
from run_bilin18_balanced_shared_featurizers import Sparse,BalancedShared
from run_bilin18_relay_contrast_subspaces import run
from run_bilin18_early_local_geometry import sha
S='bilin18_single_state_norm_calibration'
@torch.inference_mode()
def main():
 start=time.time();torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');paths={'fits':ART/'bilin18_relay_natural_featurizers_captures.pt','data':ART/'bilin18_relay_natural_data_captures.pt','tasks':ART/'bilin18_relay_fresh_distance_captures.pt','task_results':ART/'bilin18_relay_fresh_distance.json','runner':Path(__file__),'model':SNAPSHOT/'pytorch_model.bin'};hashes={k:sha(p) for k,p in paths.items()};cfg={'experiment':'H857','hashes':hashes,'design':'All10 frozenH852 models; independent single-state learned and fixed-support LS coordinates, each with/without residual-norm calibration. No task direction or gain fitting.','encoding':'C=normalizedstate-tokenmean, decoder bias beta. Calibrated C_hat=beta+(C_hat-beta)*norm(C-beta)/norm(C_hat-beta). Preserve mean and restore each states own rawRMS after fullshape normalization. One additional native residual-norm scalar, beyond tokenidentity and rawRMS already accessed. Gates/support unchanged.','causal':'Define one raw reconstruction function F(input), then edit native recipient rawstate by F(donor)-F(recipient). This is cycle-consistent before model execution. UnlikeH853/H854, normalize each reconstruction before differencing; compare variants within this experiment. Exact rawdonor is reference.','natural':'Same512 H851 development states, reconstruction and four fixed Gaussian localMLP derivative probes fromH851. Reused validation, no fitting.','scope':'Reused task panel and naturaldevelopment. Test inference rule, not fresh confirmation or semantic-unit recovery. Single-state mapping preserves recipient reconstruction error under interventions.'};(ROOT/f'configs/{S}.json').write_text(json.dumps(cfg,indent=2)+'\n');states=torch.load(paths['fits'],weights_only=False)['states'];data=torch.load(paths['data'],weights_only=False);f=data['fits'][4];task=torch.load(paths['tasks'],weights_only=False);ex=task['examples'];native=task['native'];tb=torch.stack([task['premlp_relay'][i] for i in range(len(ex))]).double();vb=data['captures']['validation',4].double();b=torch.cat([tb,vb]);tids=[e['ids'][e['position']] for e in ex]+list(map(int,data['validation_token_ids']));lookup={int(t):i for i,t in enumerate(f['token_ids'])};eps=torch.finfo(torch.float32).eps;rad=(b.square().mean(1,keepdim=True)+eps).sqrt();u=b/rad;tm=torch.stack([f['token_shape_mean'][lookup[t]] if t in lookup else f['shape_mean'] for t in tids]);res=u-tm;N=len(ex);predictions={};natural=[];gains=[];normerr=0.;lsworse=0.;cycleerr=0.
 for kind in ['scalar48','scalar64','block8','shared_fixed','shared_learned']:
  for seed in [8351,8352]:
   name=f'{kind}_{seed}';ck=states[4,kind,seed];m=BalancedShared(seed,kind=='shared_fixed').cuda() if kind.startswith('shared') else Sparse(8 if kind=='block8' else 1,48 if kind=='scalar48' else 64).cuda();m.load_state_dict(ck['state_dict']);outs=[m(x) for x in (res/ck['scale']).float().cuda().split(128)];learned=torch.cat([o['reconstruction'] for o in outs]).double().cpu()*ck['scale'];support=torch.cat([o['support'] for o in outs]).bool().cpu();D=(m.dictionary_matrix().T if kind.startswith('shared') else m.decoder).double().cpu();beta=m.bias.double().cpu()*ck['scale'];cache={};ls=[]
   for i in range(len(b)):
    key=tuple(support[i].nonzero().flatten().tolist())
    if key not in cache:
     _,sv,v=torch.linalg.svd(D[list(key)],full_matrices=False);rank=int((sv>sv[0]*1e-10).sum());cache[key]=v[:rank].T
    Q=cache[key];ls.append((res[i]-beta)@Q@Q.T+beta)
   ls=torch.stack(ls);lsworse=max(lsworse,float(((ls-res).square().sum(1)-(learned-res).square().sum(1)).max()));variants={'learned':learned,'ls':ls}
   for base,p in list(variants.items()):
    den=(p-beta).norm(dim=1,keepdim=True);assert float(den.min())>1e-10;gain=(res-beta).norm(dim=1,keepdim=True)/den;pn=beta+(p-beta)*gain;normerr=max(normerr,float(((pn-beta).norm(dim=1)-(res-beta).norm(dim=1)).abs().max()));variants[base+'_norm']=pn;gains.append({'model':name,'base':base,'task_gain_mean':float(gain[:N].mean()),'task_gain_max':float(gain[:N].max()),'natural_gain_mean':float(gain[N:].mean()),'natural_gain_max':float(gain[N:].max())})
   for variant,p in variants.items():
    shape=tm+p;raw=shape/shape.square().mean(1,keepdim=True).sqrt().clamp_min(1e-30)*(rad.square()-eps).sqrt();predictions[name,variant]=raw;natural.append({'model':name,'variant':variant,'residual_relative_MSE':float((res[N:]-p[N:]).square().sum()/res[N:].square().sum()),'raw_relative_MSE':float((b[N:]-raw[N:]).square().sum()/b[N:].square().sum())});cycle=(raw[1]-raw[0])+(raw[2]-raw[1])+(raw[0]-raw[2]);cycleerr=max(cycleerr,float(cycle.abs().max()))
   del m,outs
   print('ENCODED',name,flush=True)
 model=load_full_model('cuda')
 for mod in [model.embedding,model.blocks,model.lm_head]:mod.float().eval()
 mlp=model.blocks[4].mlp;L,R,D=[getattr(mlp,k).weight.double() for k in ['Left','Right','Down']]
 def response(base,v):
  radius=(base.square().mean(1,keepdim=True)+eps).sqrt();sh=base/radius;q=v/radius-base*(base*v).mean(1,keepdim=True)/radius.pow(3);return F.linear(F.linear(sh,L)*F.linear(q,R)+F.linear(q,L)*F.linear(sh,R),D)
 errors={key:[0.,0.] for key in predictions}
 for v in f['directions'][:4].cuda():
  truth=response(vb.cuda(),v);den=float(truth.square().sum())
  for key,p in predictions.items():y=response(p[N:].cuda(),v);errors[key][0]+=float((y-truth).square().sum());errors[key][1]+=den
 for r in natural:r['local_response_relative_RMSE']=(errors[r['model'],r['variant']][0]/errors[r['model'],r['variant']][1])**.5
 source=json.loads(paths['task_results'].read_text());index={(e['id'],e['variant']):i for i,e in enumerate(ex)};jobs=[(r,index[r['id'],r['recipient']],index[r['id'],r['donor']]) for r in source['records'] if r['kind']=='full'];records=[];actual=0;identity=0.;replay=0.;full={}
 for length in sorted({len(e['ids']) for e in ex}):
  jj=[z for z in jobs if len(ex[z[1]]['ids'])==length]
  for st in range(0,len(jj),24):
   batch=jj[st:st+24];ids=torch.tensor([ex[i]['ids'] for r,i,j in batch],device='cuda');pos=torch.tensor([ex[i]['position'] for r,i,j in batch],device='cuda');settings=[('reference','identity'),('reference','full')]+list(predictions)
   for name,variant in settings:
    replacement=torch.stack([b[i] if variant=='identity' else b[j] if variant=='full' else b[i]+predictions[name,variant][j]-predictions[name,variant][i] for r,i,j in batch]).cuda();y=run(model,ids,replacement,pos);actual+=len(batch)
    for k,(r,i,j) in enumerate(batch):
     if variant=='identity':identity=max(identity,float((y[k].double().cpu()-native[i]).abs().max()));continue
     baseline=measure(native[i],ex[i]['groups'])['margin'];value=measure(y[k],ex[i]['groups'])['margin'];sgn=1 if r['recipient']==1 else -1;effect=sgn*(value-baseline)
     if variant=='full':full[i]=y[k].double().cpu();replay=max(replay,abs(effect-r['donor_directed_delta']))
     lp=full[i].log_softmax(0);records.append({key:r[key] for key in ['id','family','format','gap','source_pair','recipient','donor','native_answer_gap']}|{'model':name,'variant':variant,'donor_directed_delta':effect,'KL_full_to_edit':float((lp.exp()*(lp-y[k].double().cpu().log_softmax(0))).sum())})
  print('DONE',length,flush=True)
 audits={'hashes_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'residual_norm_retention_max_abs':normerr,'per_state_LS_error_increase_max':lsworse,'raw_reconstruction_cycle_error_max_abs':cycleerr,'identity_logits_max_abs':identity,'H849_full_margin_replay_max_abs':replay,'whole_model_forwards':actual,'test_evaluations':0,'finite':all(np.isfinite(r['KL_full_to_edit']) for r in records)};assert audits['hashes_unchanged'] and max(identity,replay)<1e-4 and max(normerr,cycleerr)<1e-9 and lsworse<1e-4 and audits['finite'];torch.save({'predictions':predictions},ART/f'{S}_predictions.pt');(ART/f'{S}.json').write_text(json.dumps({'config':cfg,'records':records,'natural_evaluation':natural,'gains':gains,'audits':audits,'elapsed_seconds':time.time()-start},indent=2)+'\n');print(audits,flush=True)
if __name__=='__main__':main()
