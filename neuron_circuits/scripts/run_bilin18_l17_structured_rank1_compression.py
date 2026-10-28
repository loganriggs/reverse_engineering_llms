import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_structured_rank1_compression'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd=L.double(),R.double(),D.double();src=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');b=torch.tensor(src['basis'][:,0],device='cuda').double();h=torch.tensor(src['sources'][0],device='cuda').double();amps=h@b;bl= Ld@b;br=Rd@b;M=-Dd@(br[:,None]*Ld+bl[:,None]*Rd);N2=Dd@(bl*br);d2=b.square().mean();eps=torch.finfo(torch.float32).eps;pa=np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][:1280];pt=torch.tensor(pa[:1024],device='cuda').double();aa=amps[np.arange(1024)%32];d0=pt.square().mean(-1)+eps;d1=-2*(pt*b).mean(-1);cross=pt@M.T;Y=torch.cat([alpha*aa[:,None]*cross/(d0+alpha*aa*d1+aa.square()*d2)[:,None] for alpha in [-1,1]]);_,sv,Vh=torch.linalg.svd(Y,full_matrices=False);V=Vh[:128];read=M.T@V.T;audit={'cross_map_identity_relative':float((pt@M.T+((pt@Ld.T)*br+(pt@Rd.T)*bl)@Dd.T).norm()/(pt@M.T).norm())};assert audit['cross_map_identity_relative']<1e-10
 def native(p):
  x=F.rms_norm(p,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 def evaluate(pa,models):
  p=torch.tensor(pa,device='cuda');pd=p.double();y0=native(p);w0=y0.double()-bias.double();state=p+y0;hh=h[np.arange(len(p))%64];aa=amps[np.arange(len(p))%64];d0=pd.square().mean(-1)+eps;d1=-2*(pd*b).mean(-1);coords=pd@read;rows={};metrics={}
  for sign,alpha in [('negative',-1),('positive',1)]:
   a=alpha*aa;den=d0+a*d1+a.square()*d2;yy=native(p-alpha*((hh@b)[:,None]*b).float());yf=native(p-alpha*hh.float());u=(yy-y0).double();uf=(yf-y0).double();ds={}
   for name,(rank,norm) in models.items():
    cr=pd@M.T if rank==1152 else coords[:,:rank]@V[:rank];delta=a[:,None]*cr/den[:,None]+a.square()[:,None]*N2/den[:,None]
    if norm:delta=delta+(d0/den-1)[:,None]*w0
    ds[name]=delta
   acc={n:[] for n in models};effects=[]
   for start in range(0,len(p),16):
    sl=slice(start,start+16);truth=lp(p[sl]+yy[sl]);pr=truth.exp();full=lp(p[sl]+yf[sl]);pf=full.exp();base=lp(state[sl]);effects.append(torch.stack([(pr*(truth-base)).sum(-1),u[sl].square().sum(-1),(pf*(full-base)).sum(-1),uf[sl].square().sum(-1)],-1).cpu().numpy())
    for name,delta in ds.items():
     q=lp(state[sl]+delta[sl].float());acc[name].append(torch.stack([(pr*(truth-q)).sum(-1),(delta[sl]-u[sl]).square().sum(-1),(pf*(full-q)).sum(-1),(delta[sl]-uf[sl]).square().sum(-1)],-1).cpu().numpy())
   ef=np.concatenate(effects);rows[sign+'_effects']=ef;metrics[sign]={}
   for name,vv in acc.items():
    rr=np.concatenate(vv);rows[sign+'_'+name]=rr;metrics[sign][name]=dict(zip(['rank1_KL','rank1_raw','full_KL','full_raw'],(1-rr.sum(0)/ef.sum(0)).tolist()))
  return rows,metrics
 vr,vm=evaluate(pa[1024:],{f'r{rank}':(rank,True) for rank in cfg['ranks']});eligible=[rank for rank in cfg['ranks'] if all(vm[sign][f'r{rank}']['rank1_KL']>.95 for sign in ['negative','positive'])];rank=min(eligible) if eligible else max(cfg['ranks']);selection={'rank':rank,'validation_criterion_met':bool(eligible),'validation_metrics':vm};STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps(selection,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_compiled.npz'),read=read[:,:rank].cpu().numpy(),write=V[:rank].cpu().numpy(),source_direction=b.cpu().numpy(),quadratic_vector=N2.cpu().numpy(),d2=float(d2),eps=eps);print('SELECTION SAVED',selection,flush=True)
 docs=np.load(ROOT/cfg['corpus']);pre,_,_=collect_sources(model,docs,cfg);rows,metrics=evaluate(pre.cpu().numpy(),{'structured':(rank,True),'omit_norm_correction':(rank,False),'exact_rank1':(1152,True)});print('confirmation',metrics,flush=True);assert all(np.isfinite(v).all() for v in rows.values());assert all(metrics[sign]['exact_rank1']['rank1_raw']>1-1e-7 for sign in metrics);np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,pre=pre.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H487','config':cfg,'selection':selection,'metrics':metrics,'audits':audit,'stored_compiled_scalars':2304*rank+2306,'unique_completed_documents':17344},indent=2)+'\n')
if __name__=='__main__':main()
