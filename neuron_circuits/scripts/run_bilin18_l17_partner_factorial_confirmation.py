"""H301 frozen rank/refinement factorial on new document panel."""
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model,collect_sources,load_program
from run_bilin18_l17_partner_compression import cross_response
from run_bilin18_l17_core_interaction_factorial import quadratic_terms
from run_bilin18_l8_execution_alignment import angle_matched_payload
STEM=ART/'bilin18_l17_partner_factorial_confirmation'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for name,digest in cfg['source_sha256'].items():assert hashlib.sha256((ART/name).read_bytes()).hexdigest()==digest
 model=load_full_model('cuda');model.embedding.float();model.blocks.float();block=model.blocks[17];L,R,D=[m.weight.double() for m in [block.mlp.Left,block.mlp.Right,block.mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda',torch.float64);partners={};audits={}
 for rank,suffix in [(32,''),(64,'_rank64')]:
  f=load_program(ART/f'bilin18_l17_partner_common_descent{suffix}_frozen.npz','cuda',torch.float64)
  if rank==32:core=torch.linalg.qr(f['core']).Q;mu=f['mean']
  else:assert torch.equal(f['mean'],mu)
  for name,key in [('proxy','weighted'),('refined','optimized')]:
   q=torch.linalg.qr(f[key]-core@(core.T@f[key])).Q;partners[f'{name}{rank}']=q;audits[f'{name}{rank}']={'overlap':float((core.T@q).abs().max()),'orthogonality':float((q.T@q-torch.eye(rank,device='cuda')).abs().max())}
 pre,heads,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),cfg);x=F.rms_norm(pre,(1152,)).double();gen=torch.Generator(device='cuda').manual_seed(cfg['seed']);rows={}
 def save(k,v):rows[k]=v.reshape(128,4).sum(1).cpu().numpy()
 for h in range(9):
  source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;actual=heads[:,h];angle=angle_matched_payload(pre,actual,source,torch.randn(len(pre),128,generator=gen,device='cuda'))
  for label,pay in [('actual',actual),('angle',angle)]:
   z=F.rms_norm(pre-pay,(1152,)).double();cross=quadratic_terms(x,z,mu,core,L,R,D,p)[:,1];prefix=f'{label}_h{h}';save(prefix+'_energy',cross.square().sum(-1))
   for name,q in partners.items():save(f'{prefix}_{name}_error',(cross_response(x,z,mu,core,q,L,R,D,p)-cross).square().sum(-1))
  print('head',h,'complete',flush=True)
 ix=np.random.default_rng(cfg['seed']+1).integers(128,size=(2000,128));results={};contrasts={};concentration={};per_head={}
 pairs={'rank_gain':('refined32','refined64'),'refinement32':('proxy32','refined32'),'refinement64':('proxy64','refined64')}
 for label in ['actual','angle']:
  energy=sum(rows[f'{label}_h{h}_energy'] for h in range(9));assert np.all(energy>0);errors={name:sum(rows[f'{label}_h{h}_{name}_error'] for h in range(9)) for name in partners};results[label]={};contrasts[label]={};per_head[label]={}
  concentration[label]={'max_document_energy_share':float(energy.max()/energy.sum()),'top5_document_energy_share':float(np.sort(energy)[-5:].sum()/energy.sum())}
  for name,err in errors.items():results[label][name]={'fidelity':float(1-err.sum()/energy.sum()),'paired95':np.quantile(1-err[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist(),'equal_document_fidelity':float(np.mean(1-err/energy))}
  for name,(old,new) in pairs.items():
   gain=errors[old]-errors[new];eq=gain/energy;contrasts[label][name]={'gain':float(gain.sum()/energy.sum()),'paired95':np.quantile(gain[ix].sum(1)/energy[ix].sum(1),[.025,.975]).tolist(),'equal_document_gain':float(eq.mean()),'equal_document95':np.quantile(eq[ix].mean(1),[.025,.975]).tolist(),'positive_document_fraction':float(np.mean(gain>0)),'leave_one_out_range':[float(v) for v in [np.min((gain.sum()-gain)/(energy.sum()-energy)),np.max((gain.sum()-gain)/(energy.sum()-energy))]]}
   per_head[label][name]=[float((rows[f'{label}_h{h}_{old}_error']-rows[f'{label}_h{h}_{new}_error']).sum()/rows[f'{label}_h{h}_energy'].sum()) for h in range(9)]
 gates={name:all(contrasts[label][name]['gain']>=(.05 if name=='rank_gain' else .02) and contrasts[label][name]['paired95'][0]>0 for label in results) for name in pairs};gates['compact80']=all(results[label]['refined64']['fidelity']>=.8 for label in results)
 np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**rows);STEM.with_suffix('.json').write_text(json.dumps({'config':cfg,'results':results,'contrasts':contrasts,'per_head_gains':per_head,'concentration':concentration,'basis_audits':audits,'gates':gates},indent=2)+'\n');print('GATES',gates,flush=True)
if __name__=='__main__':main()
