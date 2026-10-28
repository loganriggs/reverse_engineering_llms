import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_common_differential_error'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 z=np.load(ART/'bilin18_l17_source_split_stability_models.npz');B=T(z['basis']);mu=T(z['mean']);old=np.load(ART/'bilin18_l17_source_split_stability_rows.npz');pool=np.load(ART/'bilin18_l17_source_sample_curve_rows.npz')
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();eps=torch.finfo(torch.float32).eps
 def native(x):
  v=F.rms_norm(x,(1152,));return ((v@L.T)*(v@R.T))@D.T+bias
 def lp(x):return (30*torch.tanh(F.rms_norm(x,(1152,))@W.T/30)).double().log_softmax(-1)
 def dn(x,v):
  ss=(x.square().mean(-1)+eps).sqrt();return v/ss[:,None]-x*(x*v).mean(-1)[:,None]/ss[:,None]**3
 saved={};metrics={};audits={}
 for panel,file in cfg['evaluation_panels'].items():
  data=np.load(ART/file);p=torch.tensor(data['pre'],device='cuda');h=T(data['head_writes']).sum(1)-mu;hc=h@B;y=native(p);metrics[panel]={}
  for rep in range(4):
   for k in cfg['ranks']:
    for method in ['PCA','Fisher']:
     name=f'r{rep}_{method}{k}';aa=T(z[f'r{rep}s0_{method}{k}']);bb=T(z[f'r{rep}s1_{method}{k}']);ea=h-hc@aa.T@B.T;eb=h-hc@bb.T@B.T;ec=(ea+eb)/2;ed=(ea-eb)/2;inp=p-ec.float();state=inp+native(inp);rows=[]
     for start in range(0,512,16):
      sl=slice(start,start+16);pp=p[sl];x=F.rms_norm(pp,(1152,));lx=x@L.T;rx=x@R.T;st=pp+y[sl];soft=torch.tanh(F.rms_norm(st,(1152,))@W.T/30);prob=(30*soft).double().softmax(-1)
      def score(e):
       dx=dn(pp,e.float());ds=e.float()+((dx@L.T)*rx+lx*(dx@R.T))@D.T;j=((dn(st,ds)@W.T)*(1-soft.square())).double();return j-(prob*j).sum(-1)[:,None]
      jc,jd,ja,jb=[score(e[sl]) for e in [ec,ed,ea,eb]]
      c=.5*(prob*jc.square()).sum(-1);d=.5*(prob*jd.square()).sum(-1);cross=(prob*jc*jd).sum(-1);al=.5*(prob*ja.square()).sum(-1);bl=.5*(prob*jb.square()).sum(-1)
      base=lp(st);q=lp(state[sl]);mid=(base.exp()*(base-q)).sum(-1)
      rows.append(torch.stack([c,d,cross,al,bl,mid],-1).cpu().numpy())
     rr=np.concatenate(rows);assert np.isfinite(rr).all();scale=rr[:,3]+rr[:,4]+1e-20;identity=max(np.max(abs(rr[:,3]-(rr[:,0]+rr[:,1]+rr[:,2]))/scale),np.max(abs(rr[:,4]-(rr[:,0]+rr[:,1]-rr[:,2]))/scale));assert identity<1e-5
     saved[panel+'_'+name]=rr;prev=old[panel+'_'+name];pooled=pool[panel+f'_n3072_{method}{k}'][:,0].mean();m=rr.mean(0);half=prev[:,:2].mean();pair=prev[:,2].mean()
     metrics[panel][name]={'common_local':float(m[0]),'differential_local':float(m[1]),'cross_local':float(m[2]),'average_half_local':float(m[3:5].mean()),'differential_fraction':float(m[1]/m[3:5].mean()),'average_half_finite_KL':float(half),'midpoint_finite_KL':float(m[5]),'midpoint_relative_improvement':float(1-m[5]/half),'pooled_fixed_rank_KL':float(pooled),'pooled_over_midpoint_KL':float(pooled/m[5]),'finite_midpoint_gain_over_local_differential':float((half-m[5])/m[1]),'finite_pair_KL_over_four_local_differential':float(pair/(4*m[1])),'midpoint_rank_upper_bound':min(2*k,1023)}
     audits[panel+'_'+name]={'max_local_identity_relative':float(identity)}
   print('evaluated',panel,'split',rep,flush=True)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H524','config':cfg,'columns':['common_local','differential_local','cross_local','left_local','right_local','midpoint_finite_KL'],'metrics':metrics,'audits':audits,'unique_documents':24064},indent=2)+'\n')
if __name__=='__main__':main()
