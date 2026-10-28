import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
STEM=ART/'bilin18_l17_canonical_error_geometry'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 oldp=np.empty((512,1152),np.float32);olds=np.empty_like(oldp)
 for n in ['compact_operation_confirmation','unprojected_finite_confirmation','covariance_source_confirmation','covariance_fresh_confirmation']:
  n='bilin18_l17_'+n;ids=json.loads((ART/f'{n}_selection.json').read_text())['sites'];z=np.load(ART/f'{n}_rows.npz');oldp[ids]=z['pre'];olds[ids]=z['state']
 ids=json.loads((ART/'bilin18_l17_data_canonical_pilot_selection.json').read_text())['test'];new=np.load(ART/'bilin18_l17_canonical_fresh_confirmation_rows.npz');panels={'old':(oldp[ids],olds[ids]),'new':(new['pre'],new['state'])};model=load_full_model('cuda');W=model.lm_head.weight.detach().float().double();h0=torch.tensor(np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz')['sources'][0],device='cuda').double();models={};audit={};saved={}
 for rank in [64,128]:
  z=np.load(ART/f'bilin18_l17_canonical_width_budget_rank{rank}_models.npz');u={k:torch.tensor(z[k],device='cuda').double() for k in z.files};E=u['kernel_encoder'];N=len(u['support']);lam=(N/E.square().sum(0)).sqrt();coef=E*(lam/N).sqrt()[None];K=(u['support']@u['support'].T/1152).square();Kc=K-K.mean(0)[None]-K.mean(1)[:,None]+K.mean();err=float((coef.T@Kc@coef-torch.eye(rank,device='cuda')).abs().max());assert err<1e-6;u['orthogonal_feature_coefficients']=coef;models[rank]=u;audit['orthogonal_rank'+str(rank)]=err
 def norm(x):return x/(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
 names=['log_source_norm','log_receiver_norm','log_relative_source_norm','signed_source_receiver_cosine','log_feature_change_energy','uncovered_fraction64','uncovered_fraction128','log_uncovered_energy128','baseline_entropy','baseline_top1_probability','log_baseline_RMS','log_predicted_delta_energy','log_predicted_delta_Fisher']
 audit['lifted_identity_relative_max']=0.;audit['projection_excess_max']=0.
 for label,(pa,sa) in panels.items():
  p=torch.tensor(pa,device='cuda').double();state=torch.tensor(sa,device='cuda').double();N=len(p);h=h0[np.arange(N)%64];x0=norm(p);entropy=[];top=[]
  for start in range(0,N,16):
   ss=state[start:start+16];logits=30*torch.tanh(F.rms_norm(ss.float(),(1152,)).double()@W.T/30);lp=logits.log_softmax(-1);prob=lp.exp();entropy.append(-(prob*lp).sum(-1));top.append(prob.max(-1).values)
  entropy=torch.cat(entropy);top=torch.cat(top);rows=[]
  for alpha in [-1,1]:
   xx=norm(p-alpha*h);energy=(xx.square().mean(-1)).square()+(x0.square().mean(-1)).square()-2*((xx*x0).mean(-1)).square();assert float(energy.min())>0;uncovered={};pred=None
   for rank,u in models.items():
    kdelta=(xx@u['support'].T/1152).square()-(x0@u['support'].T/1152).square();kc=kdelta-kdelta.mean(1,keepdim=True);proj=(kc@u['orthogonal_feature_coefficients']).square().sum(-1);audit['projection_excess_max']=max(audit['projection_excess_max'],float((proj-energy).max()));assert float((proj-energy).max())<1e-9;uncovered[rank]=(energy-proj).clamp_min(0)
    if rank==128:pred=u['scale']*((kc@u['kernel_encoder'])@u['budget32_map'])
   # Independent explicit lifted matrices on a few rows verify the degree-two norm identity.
   for i in range(min(4,N)):
    lift=(torch.outer(xx[i],xx[i])-torch.outer(x0[i],x0[i]))/1152;actual=lift.square().sum();audit['lifted_identity_relative_max']=max(audit['lifted_identity_relative_max'],float(abs(actual-energy[i])/actual))
   fisher=[]
   for start in range(0,N,16):
    ss=state[start:start+16];v=pred[start:start+16];th=torch.tanh(F.rms_norm(ss.float(),(1152,)).double()@W.T/30);prob=(30*th).softmax(-1);t=(native_drms(ss,v)@W.T)*(1-th.square());t=t-(prob*t).sum(-1,keepdim=True);fisher.append(.5*(prob*t.square()).sum(-1))
   fisher=torch.cat(fisher);log=lambda v:torch.log(v.clamp_min(1e-30));hn=h.norm(dim=-1);pn=p.norm(dim=-1);cos=alpha*(p*h).sum(-1)/(pn*hn);row=torch.stack([log(hn),log(pn),log(hn/pn),cos,log(energy),uncovered[64]/energy,uncovered[128]/energy,log(uncovered[128]),entropy,top,log(state.square().mean(-1).sqrt()),log(pred.square().sum(-1)),log(fisher)],-1);rows.append(row.cpu().numpy())
  saved[label+'_features']=np.stack(rows);print(label,'features complete',np.stack(rows).shape,flush=True)
 assert audit['lifted_identity_relative_max']<1e-7;assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_features.npz'),**saved);j={'experiment':'H474','config':cfg,'feature_names':names,'audits':audit,'scope':'Post-hoc diagnostics; all features computed without native intervention targets/errors','unique_completed_documents':13504};STEM.with_suffix('.json').write_text(json.dumps(j,indent=2)+'\n');print('FINISHED',audit,flush=True)
if __name__=='__main__':main()
