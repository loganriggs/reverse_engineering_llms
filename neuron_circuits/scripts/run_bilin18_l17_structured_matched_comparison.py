import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_structured_matched_comparison'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 old=np.load(ART/'bilin18_l17_structured_rank1_compression_rows.npz');p=torch.tensor(old['pre'],device='cuda');pd=p.double();src=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');h=torch.tensor(src['sources'][0],device='cuda').double()[np.arange(512)%64];z=np.load(ART/'bilin18_l17_structured_rank1_compression_compiled.npz');z={k:torch.tensor(z[k],device='cuda') for k in z.files};b=z['source_direction'];aa=h@b;hr=aa[:,None]*b;model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();m=np.load(ART/'bilin18_l17_canonical_width_budget_rank128_models.npz');m={k:torch.tensor(m[k],device='cuda').double() for k in m.files};l=np.load(ART/'bilin18_l17_canonical_remainder_learning_models.npz');sel=json.loads((ART/'bilin18_l17_canonical_remainder_learning_selection.json').read_text())['selected']['pca'];C=torch.tensor(l[sel+'_coefficients'][:,:20],device='cuda');B=torch.tensor(l['basis'][:20],device='cuda');E=torch.tensor(l['expanded_kernel_encoder'],device='cuda')
 def native(q):
  x=F.rms_norm(q,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def feat(q):
  x=F.rms_norm(q,(1152,));k=(x.double()@m['support'].T/1152).square();return (k-m['kernel_mean'][None]-k.mean(1,keepdim=True)+m['kernel_grand'])@E
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);s0=p+y0;w0=y0.double()-bias.double();f0=feat(p);cross=(pd@z['read'])@z['write'];d0=pd.square().mean(-1)+z['eps'];d1=-2*(pd*b).mean(-1);saved={};metrics={};audit={}
 for sign,alpha in [('negative',-1),('positive',1)]:
  qf=p-alpha*h.float();qr=p-alpha*hr.float();yf=native(qf);yr=native(qr);uf=(yf-y0).double();ur=(yr-y0).double();gf=m['scale']*(((feat(qf)-f0)@C)@B);gr=m['scale']*(((feat(qr)-f0)@C)@B);a=alpha*aa;den=d0+a*d1+a.square()*z['d2'];st=a[:,None]*cross/den[:,None]+a.square()[:,None]*z['quadratic_vector']/den[:,None]+(d0/den-1)[:,None]*w0;ds={'generic_full':gf,'generic_rank1':gr,'structured_rank1':st,'hybrid':st+gf-gr};rows={n:[] for n in ds};effects=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);tr=lp(p[sl]+yr[sl]);tf=lp(p[sl]+yf[sl]);pr=tr.exp();pf=tf.exp();base=lp(s0[sl]);effects.append(torch.stack([(pr*(tr-base)).sum(-1),ur[sl].square().sum(-1),(pf*(tf-base)).sum(-1),uf[sl].square().sum(-1)],-1).cpu().numpy())
   for n,v in ds.items():
    q=lp(s0[sl]+v[sl].float());rows[n].append(torch.stack([(pr*(tr-q)).sum(-1),(v[sl]-ur[sl]).square().sum(-1),(pf*(tf-q)).sum(-1),(v[sl]-uf[sl]).square().sum(-1)],-1).cpu().numpy())
  ef=np.concatenate(effects);saved[sign+'_effects']=ef;metrics[sign]={}
  for n,rr in rows.items():
   rr=np.concatenate(rr);saved[sign+'_'+n]=rr;metrics[sign][n]=dict(zip(['rank1_KL','rank1_raw','full_KL','full_raw'],(1-rr.sum(0)/ef.sum(0)).tolist()))
  assert np.allclose(ef,old[sign+'_effects'],rtol=3e-5,atol=1e-8);assert np.allclose(saved[sign+'_structured_rank1'],old[sign+'_structured'],rtol=3e-5,atol=1e-8);audit[sign+'_structured_KL_replay_maxabs']=float(abs(saved[sign+'_structured_rank1'][:,[0,2]]-old[sign+'_structured'][:,[0,2]]).max());print(sign,metrics[sign],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H488','config':cfg,'metrics':metrics,'audits':audit,'unique_completed_documents':17344},indent=2)+'\n')
if __name__=='__main__':main()
