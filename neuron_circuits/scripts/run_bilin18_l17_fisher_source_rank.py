import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_fisher_source_rank'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();_,dh,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);src=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');calids=src['calibration_ids'];donors=src['donors'];L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd,bd,Wd=L.double(),R.double(),D.double(),bias.double(),W.double();p=torch.tensor(np.load(ART/'bilin18_l17_structured_metric_subspace_rows.npz')['pre'],device='cuda');train=torch.tensor(np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][cfg['metric_receiver_ids']],device='cuda').double();previous=np.load(ART/'bilin18_l17_allheads_source_rank_rows.npz');saved={};metrics={};geometry={};audits={};compiled={};eps=torch.finfo(torch.float32).eps
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 def double_logits(receiver,inp):
  x=inp/torch.sqrt(inp.square().mean()+eps);st=receiver+Dd@((Ld@x)*(Rd@x))+bd;n=st/torch.sqrt(st.square().mean()+eps);return 30*torch.tanh(Wd@n/30)
 y0=native(p)
 for head in cfg['heads']:
  cal=dh[calids,head].double();mu=cal.mean(0);center=cal-mu;_,sv,Vh=torch.linalg.svd(center,full_matrices=False);B=Vh[:128].T;c=sv[:128].square()/(len(calids)-1);assert float(c.min())>0;cs=c.sqrt();h=dh[donors,head].double()-mu;hc=h@B;hh=h[np.arange(512)%64];FM=torch.zeros((128,128),device='cuda',dtype=torch.float64);head_audit={}
  for ti,t in enumerate(train):
   scale=torch.sqrt(t.square().mean()+eps);x=t/scale;dx=B/scale-t[:,None]*(t@B)[None,:]/(1152*scale**3);dy=Dd@((Rd@x)[:,None]*(Ld@dx)+(Ld@x)[:,None]*(Rd@dx));st=t+Dd@((Ld@x)*(Rd@x))+bd;sc=torch.sqrt(st.square().mean()+eps);dn=dy/sc-st[:,None]*(st@dy)[None,:]/(1152*sc**3);logits=Wd@(st/sc);tanh=torch.tanh(logits/30);prob=(30*tanh).softmax(-1);J=(W@dn.float()).double()*(1-tanh.square())[:,None];J-= (prob[:,None]*J).sum(0);weighted=(prob.sqrt()[:,None]*J).float();FM+=(weighted.T@weighted).double()/len(train)
   if ti==0:
    analytic=(Wd@dn[:,0])*(1-tanh.square());step=float(t.norm())*1e-4;errs=[]
    for ss in [step,step/2]:
     numerical=(double_logits(t,t+ss*B[:,0])-double_logits(t,t-ss*B[:,0]))/(2*ss);errs.append(float((numerical-analytic).norm()/analytic.norm()))
    head_audit['central_derivative_relative_errors']=errs;assert max(errs)<1e-4
  FM=(FM+FM.T)/2;G=cs[:,None]*FM*cs[None,:];ev,U=torch.linalg.eigh(G);ev=ev.flip(0);U=U.flip(1);assert float(ev.min())>-1e-6*float(ev.max());compiled.update({f'h{head}_{n}':v.cpu().numpy() for n,v in {'basis':B,'covariance':c,'fisher':FM,'weighted_eigenvectors':U,'weighted_eigenvalues':ev,'mean':mu}.items()});metrics[str(head)]={};geometry[str(head)]={}
  for k in cfg['source_ranks']:
   u=U[:,:k];hp0=(((hc/cs)@u)@u.T*cs)@B.T;hp=hp0[np.arange(512)%64];A=cs[:,None]*(u@u.T)/cs[None,:];res=torch.eye(128,device='cuda',dtype=torch.float64)-A;weighted_error=torch.trace(FM@((res*c[None,:])@res.T));pca_error=torch.trace(G[k:,k:]);geometry[str(head)][str(k)]={'donor_raw_recovery':float(1-(h-hp0).square().sum()/h.square().sum()),'projected_source_energy_ratio':float(hp0.square().sum()/h.square().sum()),'operator_norm':float(torch.linalg.matrix_norm(A,ord=2)),'predicted_local_Fisher_recovery':float(1-weighted_error/torch.trace(G)),'PCA_predicted_local_Fisher_recovery':float(1-pca_error/torch.trace(G))};assert float(weighted_error-pca_error)<1e-6*float(torch.trace(G));metrics[str(head)][str(k)]={}
   if k==128:
    head_audit['rank128_source_relative_error']=float((h-hp0).norm()/h.norm());assert head_audit['rank128_source_relative_error']<1e-5
   for sign,alpha in [('negative',-1),('positive',1)]:
    yf=native(p-alpha*hh.float());yp=native(p-alpha*hp.float());uf=(yf-y0).double();up=(yp-y0).double();rows=[]
    for start in range(0,512,16):
     sl=slice(start,start+16);tf=lp(p[sl]+yf[sl]);tp=lp(p[sl]+yp[sl]);pb=lp(p[sl]+y0[sl]);pf=tf.exp();rows.append(torch.stack([(pf*(tf-pb)).sum(-1),(pf*(tf-tp)).sum(-1),uf[sl].square().sum(-1),(uf[sl]-up[sl]).square().sum(-1)],-1).cpu().numpy())
    rr=np.concatenate(rows);saved[f'h{head}_k{k}_{sign}']=rr;v=rr.sum(0);old=previous[f'h{head}_k{k}_{sign}'];assert np.allclose(rr[:,0],old[:,0],rtol=1e-5,atol=1e-9);metrics[str(head)][str(k)][sign]={'mean_effect_KL':float(v[0]/512),'KL_recovery':float(1-v[1]/v[0]),'raw_recovery':float(1-v[3]/v[2]),'PCA_KL_recovery':float(1-old[:,1].sum()/old[:,0].sum()),'gain_over_PCA':float((old[:,1].sum()-v[1])/v[0])}
    if k==128:assert abs(1-metrics[str(head)][str(k)][sign]['KL_recovery'])<1e-4
   print('head',head,'rank',k,'metrics',metrics[str(head)][str(k)],flush=True)
  head_audit['covariance_condition_number']=float(c.max()/c.min());head_audit['min_weighted_eigenvalue']=float(ev.min());audits[str(head)]=head_audit
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);np.savez_compressed(STEM.with_name(STEM.name+'_compiled.npz'),**compiled);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H498','config':cfg,'metrics':metrics,'source_geometry':geometry,'audits':audits,'columns':['effect_KL','projection_error_KL','effect_raw_energy','projection_raw_error'],'unique_documents':18880},indent=2)+'\n')
if __name__=='__main__':main()
