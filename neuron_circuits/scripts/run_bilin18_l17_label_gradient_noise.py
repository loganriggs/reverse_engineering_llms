import json,hashlib,numpy as np,torch
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_label_gradient_noise'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,d in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==d
 old=np.load(ART/'bilin18_l17_objective_gain_alignment_rows.npz');curv=json.loads((ART/'bilin18_l17_objective_gain_alignment_curvature_audit.json').read_text());rng=np.random.default_rng(cfg['seed']);stats={};boots={};gradsets={};saved={}
 for panel in cfg['evaluation_panels']:
  gg=old[panel+'_gradients'];gradsets[panel]={'KL':gg[:,1],'CE_minus_KL':gg[:,2]-gg[:,1]};stats[panel]={};boots[panel]={};weights=rng.multinomial(512,np.ones(512)/512,size=cfg['bootstrap_draws'])/512
  for name,x in gradsets[panel].items():
   n=len(x);mean=x.mean(0);energy=(x*x).sum(1);naive=float(mean@mean);noise=float(x.var(0,ddof=1).sum()/n);unbiased=float(((x.sum(0)**2).sum()-energy.sum())/(n*(n-1)));assert np.isclose(unbiased,naive-noise,rtol=1e-10,atol=1e-20)
   bm=weights@x;bu=(n*(bm*bm).sum(1)-weights@energy)/(n-1);boots[panel][name]=bm;saved[panel+'_'+name+'_bootstrap_unbiased']=bu
   stats[panel][name]={'naive_squared_mean':naive,'estimated_mean_sampling_variance':noise,'unbiased_squared_mean':unbiased,'bootstrap_unbiased95':np.quantile(bu,[.025,.975]).tolist(),'top1percent_gradient_energy_fraction':float(np.sort(energy)[-6:].sum()/energy.sum()),'top10percent_gradient_energy_fraction':float(np.sort(energy)[-52:].sum()/energy.sum()),'max_leave_one_out_mean_change_over_mean_norm':float(np.linalg.norm((mean-x)/(n-1),axis=1).max()/np.linalg.norm(mean))}
 cross={}
 for name in ['KL','CE_minus_KL']:
  x=gradsets['H512'][name].mean(0);y=gradsets['H521'][name].mean(0);bd=(boots['H512'][name]*boots['H521'][name]).sum(1);cross[name]={'independent_panel_dot':float(x@y),'bootstrap95':np.quantile(bd,[.025,.975]).tolist()};saved[name+'_cross_dot_bootstrap']=bd
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float()
 def T(x):return torch.tensor(x,device='cuda',dtype=torch.float64)
 z=np.load(ART/'bilin18_l17_source_sample_curve_models.npz');B=T(z['basis']);mu=T(z['mean']);read=T(old['read']);write=T(old['write'])
 L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach();eps=torch.finfo(torch.float32).eps
 def norm(x):return x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
 def lp(p):
  x=norm(p);st=p+((x@L.T)*(x@R.T))@D.T+bias;return (30*torch.tanh(norm(st)@W.T/30)).double().log_softmax(-1)
 steps={};audits={}
 for source in cfg['evaluation_panels']:
  grad=old[source+'_gradients'][:,0].mean(0);direction=-6*grad/np.linalg.norm(grad);alpha=curv[source]['exact_raw_MSE_optimal_alpha'];g=T(1+alpha*direction);steps[source]={'alpha':alpha,'panels':{}}
  for panel,file in cfg['evaluation_panels'].items():
   data=np.load(ART/file);p=torch.tensor(data['pre'],device='cuda');h=T(data['head_writes']).sum(1)-mu;coord=h@B@read;targets=torch.tensor(data['targets'],device='cuda');err=h-(coord*g)@write.T;inp=p-err.float();rows=[]
   for start in range(0,512,16):
    sl=slice(start,start+16);base=lp(p[sl]);q=lp(inp[sl]);rows.append(torch.stack([err[sl].square().mean(-1),(base.exp()*(base-q)).sum(-1),(base-q).gather(1,targets[sl,None]).squeeze(1)],-1).cpu().numpy())
   rr=np.concatenate(rows);delta=rr-old[panel+'_baseline'];saved[source+'_MSEopt_'+panel]=rr;steps[source]['panels'][panel]={'mean_change':delta.mean(0).tolist(),'paired_SE':(delta.std(0,ddof=1)/np.sqrt(512)).tolist()}
   if source==panel:
    predicted=curv[source]['linear_coefficient']*alpha+curv[source]['quadratic_coefficient']*alpha**2;actual=delta[:,0].mean();audits[source+'_quadratic_optimum_replay']=float(abs(actual-predicted)/abs(predicted));assert audits[source+'_quadratic_optimum_replay']<1e-9
 assert all(np.isfinite(x).all() for x in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);out={'experiment':'H526','config':cfg,'gradient_statistics':stats,'cross_panel':cross,'MSE_optimal_steps':steps,'audits':audits,'unique_documents':24064};STEM.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
