import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources
STEM=ART/'bilin18_l17_structured_allheads'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();_,dh,_=collect_sources(model,np.load(ROOT/cfg['donor_corpus']),cfg);src=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');calids=src['calibration_ids'];donors=src['donors'];L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();Ld,Rd,Dd=L.double(),R.double(),D.double();train=torch.tensor(np.load(ART/'bilin18_l17_canonical_learning_curve_rows.npz')['pre'][:1024],device='cuda').double();old=np.load(ART/'bilin18_l17_structured_metric_subspace_rows.npz');p=torch.tensor(old['pre'],device='cuda');pd=p.double();eps=torch.finfo(torch.float32).eps;metrics={};geometry={};saved={};compiled={};audit={}
 def native(p):
  x=F.rms_norm(p,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);s0=p+y0;w0=y0.double()-bias.double();d0=pd.square().mean(-1)+eps
 for head in cfg['heads']:
  cal=dh[calids,head].double();mu=cal.mean(0);center=cal-mu;_,sv,Vh=torch.linalg.svd(center,full_matrices=False);b=Vh[:4].T;h=dh[donors,head].double()-mu;geometry[str(head)]={'calibration_top1_energy':float(sv[0].square()/sv.square().sum()),'calibration_top4_energy':float(sv[:4].square().sum()/sv.square().sum()),'donor_top4_energy':float((h@b).square().sum()/h.square().sum()),'source_mean_energy':float(h.square().sum(-1).mean())};bl=Ld@b;br=Rd@b;M=torch.stack([-Dd@(br[:,i,None]*Ld+bl[:,i,None]*Rd) for i in range(4)]);Q=torch.stack([torch.stack([Dd@(bl[:,i]*br[:,j]) for j in range(4)]) for i in range(4)]);aa=(h@b)[np.arange(1024)%32];hp=aa@b.T;cr=(torch.einsum('nd,kod->nko',train,M)*aa[:,:,None]).sum(1);Y=torch.cat([alpha*cr/((train-alpha*hp).square().mean(-1)+eps)[:,None] for alpha in [-1,1]]);_,csv,Cvh=torch.linalg.svd(Y,full_matrices=False);V=Cvh[:64];read=M.transpose(1,2)@V.T;compiled.update({f'h{head}_{n}':v.cpu().numpy() for n,v in {'basis':b,'mean':mu,'sources':h,'read':read,'write':V,'Q':Q}.items()});hh=h[np.arange(512)%64];coords=torch.einsum('nd,kdr->nkr',pd,read);metrics[str(head)]={}
  if head==6:
   audit['head6_source_relative']=float((h-torch.tensor(src['sources'][0],device='cuda')).norm()/h.norm());assert audit['head6_source_relative']<1e-6
  for sign,alpha in [('negative',-1),('positive',1)]:
   a=alpha*(hh@b);hp=a@b.T;den=(pd-hp).square().mean(-1)+eps;cross=(coords*a[:,:,None]).sum(1)@V;quad=torch.einsum('ni,nj,ijo->no',a,a,Q);delta=(cross+quad)/den[:,None]+(d0/den-1)[:,None]*w0;yf=native(p-alpha*hh.float());yp=native(p-hp.float());uf=(yf-y0).double();up=(yp-y0).double();rows=[]
   for start in range(0,512,16):
    sl=slice(start,start+16);tf=lp(p[sl]+yf[sl]);tp=lp(p[sl]+yp[sl]);pf=tf.exp();pp=tp.exp();base=lp(s0[sl]);q=lp(s0[sl]+delta[sl].float());rows.append(torch.stack([(pf*(tf-base)).sum(-1),(pp*(tp-base)).sum(-1),(pf*(tf-tp)).sum(-1),(pf*(tf-q)).sum(-1),(pp*(tp-q)).sum(-1),uf[sl].square().sum(-1),(uf[sl]-up[sl]).square().sum(-1),(uf[sl]-delta[sl]).square().sum(-1)],-1).cpu().numpy())
   rr=np.concatenate(rows);saved[f'h{head}_{sign}']=rr;v=rr.sum(0);metrics[str(head)][sign]={'mean_full_effect_KL':float(v[0]/512),'exact_projection_KL':float(1-v[2]/v[0]),'compressed_full_KL':float(1-v[3]/v[0]),'compression_projected_KL':float(1-v[4]/v[1]),'exact_projection_raw':float(1-v[6]/v[5]),'compressed_full_raw':float(1-v[7]/v[5])}
   if head==6:
    sl=slice(0,512) if sign=='negative' else slice(512,1024);ref=old['confirmation_initial'][sl];diff=float(abs(rr[:,3].sum()-ref[:,0].sum())/v[0]);audit[sign+'_head6_recovery_drift']=diff;assert diff<1e-5
  print('head',head,'geometry',geometry[str(head)],'metrics',metrics[str(head)],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);np.savez_compressed(STEM.with_name(STEM.name+'_compiled.npz'),**compiled);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H496','config':cfg,'metrics':metrics,'source_geometry':geometry,'audits':audit,'columns':['full_effect_KL','projected_effect_KL','projection_error_KL','compressed_full_error_KL','compressed_projected_error_KL','full_raw_energy','projection_raw_error','compressed_raw_error'],'unique_completed_documents':18880},indent=2)+'\n')
if __name__=='__main__':main()
