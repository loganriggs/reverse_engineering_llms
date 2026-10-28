import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_canonical_input_expansion'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Wd=W.double();docs=np.load(ROOT/cfg['corpus']);pa=np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][:1280];h0=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').float();z=np.load(ART/'bilin18_l17_canonical_width_budget_rank128_models.npz');z={k:torch.tensor(z[k],device='cuda').double() for k in z.files};B=torch.tensor(np.load(ART/'bilin18_l17_canonical_span_oracle_rows.npz')['output_basis'],device='cuda');scale=z['scale'];C0=z['budget32_map']@B.T;audit={'map_projection_relative':float((C0@B-z['budget32_map']).norm()/z['budget32_map'].norm())};assert audit['map_projection_relative']<1e-6
 K=(z['support']@z['support'].T/1152).square();Kc=K-z['kernel_mean'][None]-z['kernel_mean'][:,None]+z['kernel_grand'];ev,U=torch.linalg.eigh(Kc);ev=ev.flip(0);U=U.flip(1);assert float(ev[255])>1e-10;extra=U[:,128:256]/ev[128:256][None]*len(K)**.5;E=torch.cat([z['kernel_encoder'],extra],1);Z=Kc@E;audit['whitened_gram_error']=float((Z.T@Z/len(K)-torch.eye(256,device='cuda')).abs().max());assert audit['whitened_gram_error']<1e-6;z['kernel_encoder']=E;C0=torch.cat([C0,torch.zeros_like(C0)],0)
 def native(x):return ((x@L.T)*(x@R.T))@D.T+bias
 def feat(x):
  k=(x.double()@z['support'].T/1152).square();return (k-z['kernel_mean'][None]-k.mean(1,keepdim=True)+z['kernel_grand'])@z['kernel_encoder']
 def lp(s):return (30*torch.tanh(F.rms_norm(s.float(),(1152,))@W.T/30)).double().log_softmax(-1)
 def panel(pa,donors,metric=False):
  p=torch.tensor(pa,device='cuda');h=h0[np.arange(len(p))%donors];xx=F.rms_norm(p,(1152,));y0=native(xx);state=p+y0;z0=feat(xx);Xs=[];Us=[];Ys=[]
  for alpha in [-1,1]:
   xx=F.rms_norm(p-alpha*h,(1152,));yy=native(xx);Xs.append(feat(xx)-z0);Us.append((yy-y0).double()/scale);Ys.append(p+yy)
  out={'X':torch.cat(Xs),'U':torch.cat(Us),'target':torch.cat(Ys),'state':state.repeat(2,1),'n':len(p)}
  if metric:
   gs=[];bs=[]
   for start in range(0,2*len(p),16):
    ss=out['state'][start:start+16];u=out['U'][start:start+16];th=torch.tanh(F.rms_norm(ss,(1152,)).double()@Wd.T/30);prob=(30*th).softmax(-1)
    def tangent(v):
     t=(native_drms(ss.double(),v)@Wd.T)*(1-th.square());return t-(prob*t).sum(-1,keepdim=True)
    tb=torch.stack([tangent(b.expand_as(u)) for b in B],1);tu=tangent(u);gs.append(torch.einsum('biv,bv,bjv->bij',tb,prob,tb));bs.append(torch.einsum('biv,bv,bv->bi',tb,prob,tu))
   out['G']=torch.cat(gs);out['rhs']=torch.cat(bs)
  return out
 train=panel(pa[:1024],32,True);val=panel(pa[1024:],64);saved={'basis':B.cpu().numpy(),'original_coefficients':C0.cpu().numpy(),'expanded_kernel_encoder':E.cpu().numpy()};candidates={'original':C0};solver={}
 for width in cfg['widths']:
  X=train['X'][:,:width];G=train['G'];rhs=train['rhs'];N=len(X);normalizer=G.diagonal(dim1=-2,dim2=-1).mean();G=G/normalizer;rhs=rhs/normalizer;dim=width*16;H=torch.einsum('ni,nj,nab->iajb',X,X,G).reshape(dim,dim)/N;H=(H+H.T)/2;target=(X.T@rhs/N).flatten();ridgeunit=H.diagonal().mean();I=torch.eye(dim,device='cuda',dtype=torch.float64)
  for lam in cfg['ridge_grid']:
   A=H+lam*ridgeunit*I;b=target+lam*ridgeunit*C0[:width].flatten();c=torch.linalg.solve(A,b);res=float((A@c-b).norm()/b.norm());assert res<1e-7;name='r'+str(width)+'_'+str(lam);cc=torch.zeros_like(C0);cc[:width]=c.reshape(width,16);candidates[name]=cc;solver[name]=res
  print(width,'fits complete',flush=True)
 def evaluate(pan,models):
  errors={n:[] for n in models};effects=[]
  for start in range(0,len(pan['X']),16):
   sl=slice(start,start+16);truth=lp(pan['target'][sl]);pr=truth.exp();base=lp(pan['state'][sl]);u=pan['U'][sl]*scale;effects.append(torch.stack([(pr*(truth-base)).sum(-1),u.square().sum(-1)],-1).cpu().numpy())
   for name,C in models.items():
    delta=(scale*((pan['X'][sl]@C)@B)).float();q=lp(pan['state'][sl]+delta);errors[name].append(torch.stack([(pr*(truth-q)).sum(-1),(delta.double()-u).square().sum(-1)],-1).cpu().numpy())
  return {n:np.concatenate(v) for n,v in errors.items()},np.concatenate(effects)
 ve,ef=evaluate(val,candidates);vl={n:float(v[:,0].mean()) for n,v in ve.items()};selected={f'r{width}':min(['original']+[n for n in candidates if n.startswith(f'r{width}_')],key=lambda n:vl[n]) for width in cfg['widths']};chosen={'original':C0,**{m:candidates[n] for m,n in selected.items()}};print('selected',selected,vl,flush=True)
 for n,c in candidates.items():saved[n+'_coefficients']=c.cpu().numpy()
 np.savez_compressed(STEM.with_name(STEM.name+'_models.npz'),**saved);STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps({'selected':selected,'validation_KL':vl},indent=2)+'\n');print('SELECTION SAVED BEFORE CONFIRMATION',flush=True)
 tc=dict(cfg,rows=512);pt,_,_=collect_sources(model,docs,tc);test=panel(pt.cpu().numpy(),64);rows={};metrics={}
 for label,pan in [('train',train),('validation',val),('confirmation',test)]:
  er,ef=evaluate(pan,chosen);rows[label+'_effects']=ef;metrics[label]={}
  for name,v in er.items():
   rows[label+'_'+name]=v;metrics[label][name]={}
   for sign,sl in [('negative',slice(0,pan['n'])),('positive',slice(pan['n'],None))]:metrics[label][name][sign]={'KL_recovery':float(1-v[sl,0].sum()/ef[sl,0].sum()),'raw_recovery':float(1-v[sl,1].sum()/ef[sl,1].sum())}
  print(label,metrics[label],flush=True)
 audit['normal_equation_relative']=solver;assert all(np.isfinite(v).all() for v in rows.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,pre=np.concatenate([pa,pt.cpu().numpy()]));STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H479','config':cfg,'selected':selected,'validation_KL':vl,'metrics':metrics,'audits':audit,'unique_completed_documents':15808},indent=2)+'\n')
if __name__=='__main__':main()
