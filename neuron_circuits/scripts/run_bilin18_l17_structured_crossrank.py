import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_structured_crossrank'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd=L.double(),R.double(),D.double();src=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');basis=torch.tensor(src['basis'],device='cuda').double();h=torch.tensor(src['sources'][0],device='cuda').double();pa=np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][:1280];pt=torch.tensor(pa[:1024],device='cuda').double();eps=torch.finfo(torch.float32).eps;models={};saved={};audit={}
 for width in cfg['cross_ranks']:
  k=4
  b=basis[:,:k];bl=Ld@b;br=Rd@b;M=torch.stack([-Dd@(br[:,i,None]*Ld+bl[:,i,None]*Rd) for i in range(k)]);Q=torch.stack([torch.stack([Dd@(bl[:,i]*br[:,j]) for j in range(k)]) for i in range(k)]);amps=(h@b)[np.arange(1024)%32];hp=amps@b.T;crosses=torch.einsum('nd,kod->nko',pt,M);cross=(crosses*amps[:,:,None]).sum(1);d0=pt.square().mean(-1)+eps;d1=-2*(pt*hp).mean(-1);d2=hp.square().mean(-1);Y=torch.cat([alpha*cross/(d0+alpha*d1+d2)[:,None] for alpha in [-1,1]]);_,sv,Vh=torch.linalg.svd(Y,full_matrices=False);V=Vh[:width];read=M.transpose(1,2)@V.T;models[width]={'basis':b,'M':M,'Q':Q,'read':read,'write':V};saved.update({f'r{width}_{n}':v.cpu().numpy() for n,v in models[width].items() if n!='M'});audit[f'r{width}_basis_orthogonality']=float((b.T@b-torch.eye(k,device='cuda')).abs().max());assert audit[f'r{width}_basis_orthogonality']<1e-10
 def native(p):
  x=F.rms_norm(p,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 def evaluate(pa):
  p=torch.tensor(pa,device='cuda');pd=p.double();y0=native(p);w0=y0.double()-bias.double();state=p+y0;hh=h[np.arange(len(p))%64];d0=pd.square().mean(-1)+eps;rows={};metrics={}
  for sign,alpha in [('negative',-1),('positive',1)]:
   yf=native(p-alpha*hh.float());uf=(yf-y0).double();pred={};targets={};errs={};effects=[]
   for k,m in models.items():
    amps=alpha*(hh@m['basis']);hp=amps@m['basis'].T;den=(pd-hp).square().mean(-1)+eps;coef=torch.einsum('nd,kdr->nkr',pd,m['read']);cr=(coef*amps[:,:,None]).sum(1)@m['write'];quad=torch.einsum('ni,nj,ijo->no',amps,amps,m['Q']);pred[f'r{k}']=(cr+quad)/den[:,None]+(d0/den-1)[:,None]*w0;crx=(torch.einsum('nd,kod->nko',pd,m['M'])*amps[:,:,None]).sum(1);pred[f'exact_r{k}']=(crx+quad)/den[:,None]+(d0/den-1)[:,None]*w0;targets[k]=native(p-hp.float());ref=(targets[k]-y0).double();rel=float((pred[f'exact_r{k}']-ref).norm()/ref.norm());audit[f'{sign}_k{k}_exact_delta_relative']=max(audit.get(f'{sign}_k{k}_exact_delta_relative',0),rel);assert rel<1e-4
   errs={n:[] for n in pred}
   for start in range(0,len(p),16):
    sl=slice(start,start+16);tf=lp(p[sl]+yf[sl]);pf=tf.exp();base=lp(state[sl]);effects.append(torch.stack([(pf*(tf-base)).sum(-1),uf[sl].square().sum(-1)],-1).cpu().numpy())
    for n,v in pred.items():
     q=lp(state[sl]+v[sl].float());errs[n].append(torch.stack([(pf*(tf-q)).sum(-1),(v[sl]-uf[sl]).square().sum(-1)],-1).cpu().numpy())
   ef=np.concatenate(effects);rows[sign+'_effects']=ef;metrics[sign]={}
   for n,er in errs.items():
    er=np.concatenate(er);rows[sign+'_'+n]=er;metrics[sign][n]={'full_KL':float(1-er[:,0].sum()/ef[:,0].sum()),'full_raw':float(1-er[:,1].sum()/ef[:,1].sum())}
  return rows,metrics
 vr,vm=evaluate(pa[1024:]);eligible=[k for k in cfg['cross_ranks'] if all(vm[sign][f'r{k}']['full_KL']>.95 for sign in ['negative','positive'])];selected=min(eligible) if eligible else 128;selection={'cross_rank':selected,'validation_criterion_met':bool(eligible),'validation_metrics':vm};STEM.with_name(STEM.name+'_selection.json').write_text(json.dumps(selection,indent=2)+'\n');np.savez_compressed(STEM.with_name(STEM.name+'_compiled.npz'),**saved);print('SELECTION SAVED',selection,flush=True)
 pre,_,_=collect_sources(model,np.load(ROOT/cfg['corpus']),cfg);rows,metrics=evaluate(pre.cpu().numpy());print('confirmation',metrics,flush=True);assert all(np.isfinite(v).all() for v in rows.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows,pre=pre.cpu().numpy());STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H490','config':cfg,'selection':selection,'metrics':metrics,'audits':audit,'unique_completed_documents':18368},indent=2)+'\n')
if __name__=='__main__':main()
