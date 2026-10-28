"""H294 exact remaining-response polynomial terms for frozen conditional96 surrogate."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program,forward_normalized,B
from run_bilin18_l17_residual_term_accounting import residual_bilinear,mean_linear
from bilin18_residual_cross_program import intact_forward,change
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_conditional_term_accounting'
def base_bilinear(a,b,p):
    u=a@p['core'];v=b@p['core'];ij=torch.triu_indices(u.shape[1],u.shape[1],device=a.device);pairs=.5*(u[:,ij[0]]*v[:,ij[1]]+v[:,ij[0]]*u[:,ij[1]])
    return pairs@p['within']+.5*(torch.einsum('ni,nj,ijo->no',u,b@p['partner'],p['cross'])+torch.einsum('ni,nj,ijo->no',v,a@p['partner'],p['cross']))
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text());model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight.double() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda',torch.float64);pr=load_program(ART/'bilin18_l17_residual_cross_actual_refined_program.npz','cuda',torch.float64);f=load_program(ART/'bilin18_l17_response_conditional_frozen.npz','cuda',torch.float64);sj=json.loads((ART/'bilin18_l17_response_conditional.json').read_text());key=sj['best_informed'];m=f['map_'+sj['records'][key]['family']];w=f[key];q=torch.linalg.qr(m).Q;mu=f['state_mean'];pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);rows={};audit=0.
 def error(z):return B(z,z,L,R,D)-forward_normalized(z,p)-intact_forward(z,pr)-(z-mu)@m@w
 def bil(a,b):return residual_bilinear(a,b,L,R,D,p)-base_bilinear(a,b,pr)
 def lin(dx):return mean_linear(dx,mu,L,R,D,p)-change(mu[None]-dx/2,mu[None]+dx/2,pr)-(dx@m)@w
 e0=error(x);models={'linear':[0,1],'linear_within':[0,1,2],'linear_cross':[0,1,3],'linear_within_cross':[0,1,2,3],'linear_outside':[0,1,4],'full':[0,1,2,3,4]}
 def save(k,v):rows[k]=v.reshape(128,4,*v.shape[1:]).sum(1).cpu().numpy()
 for h in range(9):
  actual=heads[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
  for label,pay in [('actual',actual),('angle',angle)]:
   z=F.rms_norm(pre-pay,(1152,)).double();dx=z-x;a=x-mu;b=z-mu;ca=(a@q)@q.T;cb=(b@q)@q.T;oa=a-ca;ob=b-cb;inside=(dx@q)@q.T;terms=torch.stack([lin(inside),lin(dx-inside),bil(ca+cb,cb-ca),bil(ca+cb,ob-oa)+bil(cb-ca,oa+ob),bil(oa+ob,ob-oa)],1);target=error(z)-e0;prefix=f'{label}_h{h}';audit=max(audit,float((terms.sum(1)-target).norm()/target.norm()));save(prefix+'_energy',target.square().sum(-1));save(prefix+'_gram',torch.einsum('ntd,nud->ntu',terms,terms));save(prefix+'_alignment',(terms*target[:,None]).sum(-1))
   for name,ids in models.items():save(prefix+'_'+name,(terms[:,ids].sum(1)-target).square().sum(-1))
  print('head',h,'complete',flush=True)
 results={};ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128))
 for label in ['actual','angle']:
  total=lambda k:sum(rows[f'{label}_h{h}_'+k] for h in range(9));energy=total('energy');gram=total('gram').sum(0)/energy.sum();res={'normalized_gram':gram.tolist(),'normalized_alignment':(total('alignment').sum(0)/energy.sum()).tolist(),'sum_term_energy_ratio':float(np.trace(gram)),'sum_gram':float(gram.sum()),'models':{}}
  for name in models:
   err=total(name);res['models'][name]={'fidelity':float(1-err.sum()/energy.sum()),'paired95':np.quantile(1-err[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist()}
  results[label]=res
 gates={'within_sufficient':all(v['models']['linear_within']['fidelity']>=.8 for v in results.values()),'cross_gain':all(v['models']['linear_within_cross']['fidelity']-v['models']['linear_within']['fidelity']>=.1 for v in results.values()),'outside_needed':all(1-v['models']['linear_within_cross']['fidelity']>=.2 for v in results.values()),'replay':audit<1e-8};np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'gates':gates,'float64_relative_replay':audit},indent=2)+'\n');print('GATES',gates,'REPLAY',audit,'RESULTS',results,flush=True)
if __name__=='__main__':main()
