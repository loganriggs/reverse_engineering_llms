import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_canonical_objective_transfer'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 pp=np.empty((512,1152),np.float32)
 for p in ['compact_operation_confirmation','unprojected_finite_confirmation','covariance_source_confirmation','covariance_fresh_confirmation']:
  p='bilin18_l17_'+p;ids=json.loads((ART/f'{p}_selection.json').read_text())['sites'];pp[ids]=np.load(ART/f'{p}_rows.npz')['pre']
 te=json.loads((ART/'bilin18_l17_data_canonical_pilot_selection.json').read_text())['test'];p=torch.tensor(pp[te],device='cuda');cal=np.load(ART/'bilin18_l17_data_canonical_pilot_canonical_maps.npz');dr=np.load(ART/'bilin18_l17_canonical_gradient_diagnosis_directions.npz');selected=json.loads((ART/'bilin18_l17_canonical_gradient_diagnosis_selected_steps.json').read_text())['fractions'];t=lambda x:torch.tensor(x,device='cuda');A=t(dr['A0']);B=t(dr['B0']);maps={'write':t(cal['write_effective_map'])}
 for name in ['kl','ce']:
  alpha=selected[name];maps[name]=(A+alpha*t(dr[name+'_delta_A']))@(B+alpha*t(dr[name+'_delta_B']))
 tx=t(cal['training_normalized_inputs']);km=t(cal['kernel_mean']);grand=float(cal['kernel_grand_mean']);enc=t(cal['kernel_encoder']);mean=t(cal['write_mean']);scale=float(cal['write_scale'])
 def features(x):
  k=(x.double()@tx.T/1152).square();return ((k-km[None]-k.mean(1,keepdim=True)+grand)@enc).float()
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();labels=t(np.load(ART/'bilin18_compact_operation_confirmation_documents.npy')[te,64].astype(np.int64))
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 x0=F.rms_norm(p,(1152,));y0=native(x0);z0=features(x0);pred0={n:mean+scale*(z0@c) for n,c in maps.items()};h=t(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0]).float()[np.arange(128)%64];rows={};metrics={}
 for mode,alpha in [('intact',0),('negative',-1),('positive',1)]:
  x=F.rms_norm(p-alpha*h,(1152,));y=native(x);zz=features(x);full={n:mean+scale*(zz@c) for n,c in maps.items()};acc={n:[] for n in maps};delta={n:[] for n in maps};effects=[]
  for start in range(0,128,16):
   sl=slice(start,start+16);truth=lp(p[sl]+y[sl]);prob=truth.exp();labs=labels[sl];ar=torch.arange(len(labs),device='cuda');base=lp(p[sl]+y0[sl]);ce=-truth[ar,labs];effects.append(torch.stack([(prob*(truth-base)).sum(-1),(y[sl]-y0[sl]).double().square().sum(-1)],-1).cpu().numpy())
   for name in maps:
    q=lp(p[sl]+full[name][sl]);vals=torch.stack([((full[name][sl]-y[sl])/scale).square().mean(-1),(prob*(truth-q)).sum(-1),-q[ar,labs]-ce,(q.argmax(-1)==truth.argmax(-1)).float()],-1);acc[name].append(vals.cpu().numpy());change=full[name][sl]-pred0[name][sl];qd=lp(p[sl]+y0[sl]+change);delta[name].append(torch.stack([(prob*(truth-qd)).sum(-1),(change-(y[sl]-y0[sl])).double().square().sum(-1)],-1).cpu().numpy())
  rows[mode+'_effects']=np.concatenate(effects)
  for name in maps:
   v=np.concatenate(acc[name]);dd=np.concatenate(delta[name]);rows[mode+'_'+name]=v;rows[mode+'_'+name+'_delta']=dd;metrics[mode+'_'+name]=dict(zip(['write_MSE','KL','CE_delta','top1_agreement'],v.mean(0).tolist()));print(mode,name,metrics[mode+'_'+name],flush=True)
 old=np.load(ART/'bilin18_l17_data_canonical_pilot_rows.npz');replay=max(float(abs(rows[m+'_write']-old[m+'_write']).max()) for m in ['intact','negative','positive']);zero=max(float(abs(rows['intact_'+n+'_delta']).max()) for n in maps);assert replay<1e-4 and zero<1e-8;assert all(np.isfinite(v).all() for v in rows.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);np.savez_compressed(STEM.with_name(STEM.name+'_maps.npz'),**{n:c.cpu().numpy() for n,c in maps.items()});j={'experiment':'H468','config':cfg,'metrics':metrics,'numerical_gates':{'baseline_replay':replay<1e-4,'zero_delta':zero<1e-8},'baseline_replay_max_abs':replay,'zero_delta_max_abs':zero,'unique_completed_documents':12992};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
