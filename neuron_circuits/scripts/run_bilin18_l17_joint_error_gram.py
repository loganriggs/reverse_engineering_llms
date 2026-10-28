import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_joint_error_gram'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();data=np.load(ART/'bilin18_l17_natural_source_replacement_rows.npz');p=torch.tensor(data['pre'],device='cuda');dh=torch.tensor(data['head_writes'],device='cuda').double();compiled=np.load(ART/'bilin18_l17_fisher_source_rank_compiled.npz');pooled=np.load(ART/'bilin18_l17_fisher_pooled_metric_models.npz');full=np.load(ART/'bilin18_l17_full_path_metric_models.npz');finite=np.load(ART/'bilin18_l17_full_path_metric_rows.npz');L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();errors={n:[] for n in cfg['methods']};saved={};metrics={};audits={};eps=torch.finfo(torch.float32).eps
 for head in cfg['heads']:
  B=torch.tensor(compiled[f'h{head}_basis'],device='cuda');h=dh[:,head]-torch.tensor(compiled[f'h{head}_mean'],device='cuda');hc=h@B;Ap=torch.tensor(pooled[f'h{head}_A'],device='cuda');Af=torch.tensor(full[f'h{head}_A'],device='cuda');maps={'mean':torch.zeros_like(h),'PCA':hc[:,:4]@B[:,:4].T,'pooled':hc@Ap.T@B.T,'full_metric':hc@Af.T@B.T}
  for n in errors:errors[n].append((h-maps[n]).float())
 def rms_derivative(x,v):
  s=(x.square().mean(-1)+eps).sqrt();return v/s[:,None]-x*(x*v).mean(-1)[:,None]/s[:,None]**3
 for name in errors:
  es=torch.stack(errors[name],1);grams=[];checks=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);pp=p[sl];ee=es[sl];x=F.rms_norm(pp,(1152,));lx=x@L.T;rx=x@R.T;y=(lx*rx)@D.T+bias;state=pp+y;logits=F.rms_norm(state,(1152,))@W.T;soft=torch.tanh(logits/30);prob=(30*soft).double().softmax(-1)
   def score(e):
    dx=rms_derivative(pp,e);ds=e+((dx@L.T)*rx+lx*(dx@R.T))@D.T;J=((rms_derivative(state,ds)@W.T)*(1-soft.square())).double();return J-(prob*J).sum(-1)[:,None]
   scores=torch.stack([score(ee[:,i]) for i in range(9)],1);weighted=scores*prob.sqrt()[:,None,:];G=.5*(weighted@weighted.transpose(1,2));direct=score(ee.sum(1));rel=float((direct-scores.sum(1)).norm()/direct.norm());checks.append(rel);assert rel<1e-4;directenergy=.5*(prob*direct.square()).sum(-1);drift=float((G.sum((1,2))-directenergy).abs().sum()/directenergy.sum());checks.append(drift);assert drift<1e-4;grams.append(G.cpu().numpy())
  gg=np.concatenate(grams);saved[name+'_gram']=gg;agg=gg.mean(0);diag=float(np.trace(agg));joint=float(agg.sum());cross=joint-diag;fk=finite['full_pre_MLP_'+name][:,0];saved[name+'_finite_KL']=fk;off=agg-np.diag(np.diag(agg));metrics[name]={'mean_local_joint_KL':joint,'mean_diagonal_KL':diag,'mean_cross_KL':cross,'cross_fraction_of_joint':cross/joint,'positive_offdiagonal_sum':float(np.maximum(off,0).sum()),'negative_offdiagonal_sum':float(np.minimum(off,0).sum()),'mean_finite_KL':float(fk.mean()),'finite_over_local':float(fk.mean()/joint),'head_diagonal_fractions':(np.diag(agg)/diag).tolist(),'mean_gram':agg.tolist()};audits[name]={'max_tangent_linearity_or_energy_relative':max(checks),'minimum_per_case_diagonal':float(np.diagonal(gg,axis1=1,axis2=2).min())};print(name,{k:v for k,v in metrics[name].items() if k not in ['mean_gram','head_diagonal_fractions']},flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H508','config':cfg,'metrics':metrics,'audits':audits,'unique_documents':19456},indent=2)+'\n')
if __name__=='__main__':main()
