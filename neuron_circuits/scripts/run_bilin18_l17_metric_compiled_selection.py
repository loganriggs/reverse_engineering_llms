"""H442 average baseline Fisher metric, regularized oblique compression."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_metric_compiled_selection'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 data=np.load(ART/'bilin18_l17_calibration_coverage_frozen_bases.npz');pre=torch.tensor(data['calibration_pre'],device='cuda');train=data['training_ids'];val=data['validation_ids'];model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(pre,(1152,));st=pre+((xf@L.T)*(xf@R.T))@D.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(st,(1152,));scale=(pre.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=pre.double()/scale;maps=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');act=torch.einsum('roi,ni->nro',maps,x);aa=act[train].reshape(-1,1152);weight=torch.einsum('roi,rpi->op',maps,maps);G=.5*(aa.T@aa/len(train))+.5*x[train].square().mean()*weight;sketch=[];generator=torch.Generator(device='cuda');generator.manual_seed(cfg['metric_seed'])
 for si in range(512):
  th=torch.tanh(base[si:si+1].double()@W.T/30);prob=(30*th).softmax(-1)[0];cap=1-th.square()[0];center=(prob*cap)@W;ids=torch.multinomial(prob,cfg['tokens_per_context'],replacement=True,generator=generator);grad=cap[ids,None]*W[ids]-center;grad=native_drms(st[si:si+1].double(),grad);sketch.append(grad)
  if si%64==63:print('metric context',si,'complete',flush=True)
 sketch=torch.stack(sketch);gt=sketch[train].reshape(-1,1152);Fbar=.5*(gt.T@gt)/len(gt);ridge_scale=Fbar.trace()/1152;rows=[];encoders=[];decoders=[]
 for ridge in cfg['ridge_multipliers']:
  if ridge is None:
   _,b=torch.linalg.eigh(G);decoder=b[:,-64:];encoder=decoder.T;label='unweighted'
  else:
   metric=Fbar+ridge*ridge_scale*torch.eye(1152,device='cuda');ev,U=torch.linalg.eigh(metric);assert ev.min()>0;sqrt=(U*ev.sqrt())@U.T;inv=(U*ev.rsqrt())@U.T;_,v=torch.linalg.eigh(sqrt@G@sqrt);v=v[:,-64:];decoder=inv@v;encoder=v.T@sqrt;label=f'metric_ridge_{ridge}'
  prediction=(act@encoder.T)@decoder.T;res=(act-prediction)/scale[:,:,None];proj=torch.einsum('nrd,nkd->nrk',res,sketch);truth=torch.einsum('nrd,nkd->nrk',act/scale[:,:,None],sketch);loss=proj.square().sum((1,2));den=truth.square().sum((1,2));rows.append({'label':label,'ridge':ridge,'training_metric_error_fraction':float(loss[train].sum()/den[train].sum()),'validation_metric_error_fraction':float(loss[val].sum()/den[val].sum()),'validation_raw_activity_error_fraction':float(res[val].square().sum()/(act[val]/scale[val,:,None]).square().sum()),'projector_idempotence_relative':float((encoder@decoder-torch.eye(64,device='cuda')).norm()/8)});encoders.append(encoder.cpu().numpy());decoders.append(decoder.cpu().numpy());print(rows[-1],flush=True)
 selected=min(range(len(rows)),key=lambda i:(rows[i]['validation_metric_error_fraction'],i));np.savez_compressed(STEM.with_name(STEM.name+'_frozen_projection.npz'),encoders=np.stack(encoders),decoders=np.stack(decoders),train=train,val=val);j={'experiment':'H442','config':cfg,'candidates':rows,'selected':selected,'selection_scope':'Reserved baseline sampled-Fisher predicted-map error, no finite targets. Average metric estimated with128 probability-sampled exact centered token gradients per context. Conditional shared source maps/calibration split.','unique_completed_documents':12480};STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps(j,indent=2)+'\n');STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('SELECTION FROZEN',selected,flush=True)
if __name__=='__main__':main()
