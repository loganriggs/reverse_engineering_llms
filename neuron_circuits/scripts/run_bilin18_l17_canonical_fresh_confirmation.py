import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_canonical_fresh_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();docs=np.load(ROOT/cfg['corpus']);p,_,_=collect_sources(model,docs,cfg);L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();h=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float()[np.arange(512)%64];labels=torch.tensor(docs[:,64].astype(np.int64),device='cuda');models={};unfused={};maxerr=0.
 for rank in [64,128]:
  stem=f'bilin18_l17_canonical_width_budget_rank{rank}';z=np.load(ART/f'{stem}_compiled.npz');models[f'r{rank}']={k:torch.tensor(z[k],device='cuda') for k in z.files};u=np.load(ART/f'{stem}_models.npz');unfused[f'r{rank}']={k:torch.tensor(u[k],device='cuda') for k in u.files}
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 def predict(x,n):
  nonlocal maxerr
  z=models[n];k=(x.double()@z['support'].T/1152).square();pred=z['scale']*((k@z['budget32_read']-k.mean(1,keepdim=True)*z['budget32_rowmean_read'][None])@z['budget32_write']+z['budget32_intercept']);u=unfused[n];coords=((k-u['kernel_mean'][None]-k.mean(1,keepdim=True)+u['kernel_grand'])@u['kernel_encoder']).float();ref=u['scale'].float()*(coords@u['budget32_map']+u['budget32_intercept']);maxerr=max(maxerr,float((pred-ref.double()).norm()/ref.double().norm()));return pred.float()
 x0=F.rms_norm(p,(1152,));y0=native(x0);pred0={n:predict(x0,n) for n in models};rows={};metrics={}
 for mode,alpha in [('intact',0),('negative',-1),('positive',1)]:
  xx=F.rms_norm(p-alpha*h,(1152,));yy=native(xx);pred={n:predict(xx,n) for n in models};acc={n:[] for n in models};de={n:[] for n in models};effects=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);truth=lp(p[sl]+yy[sl]);prob=truth.exp();base=lp(p[sl]+y0[sl]);labs=labels[sl];ar=torch.arange(len(labs),device='cuda');ce=-truth[ar,labs];effects.append(torch.stack([(prob*(truth-base)).sum(-1),(yy[sl]-y0[sl]).double().square().sum(-1)],-1).cpu().numpy())
   for n in models:
    q=lp(p[sl]+pred[n][sl]);acc[n].append(torch.stack([((pred[n][sl]-yy[sl])/models[n]['scale'].float()).square().mean(-1),(prob*(truth-q)).sum(-1),-q[ar,labs]-ce,(q.argmax(-1)==truth.argmax(-1)).float()],-1).cpu().numpy());delta=pred[n][sl]-pred0[n][sl];qd=lp(p[sl]+y0[sl]+delta);de[n].append(torch.stack([(prob*(truth-qd)).sum(-1),(delta-(yy[sl]-y0[sl])).double().square().sum(-1)],-1).cpu().numpy())
  rows[mode+'_effects']=np.concatenate(effects)
  for n in models:
   rows[mode+'_'+n]=np.concatenate(acc[n]);rows[mode+'_'+n+'_delta']=np.concatenate(de[n]);metrics[mode+'_'+n]=dict(zip(['write_MSE','KL','CE_delta','top1_agreement'],rows[mode+'_'+n].mean(0).tolist()));print(mode,n,metrics[mode+'_'+n],flush=True)
 assert maxerr<2e-6 and all(np.isfinite(v).all() for v in rows.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,pre=p.cpu().numpy(),state=(p+y0).cpu().numpy());j={'experiment':'H473','config':cfg,'metrics':metrics,'compiled_float32_reference_relative_max':maxerr,'numerical_gates':{'compiled_fidelity':maxerr<2e-6,'finite':True},'unique_completed_documents':13504};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',j['numerical_gates'],flush=True)
if __name__=='__main__':main()
