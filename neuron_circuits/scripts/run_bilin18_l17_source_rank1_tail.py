import json,hashlib,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
STEM=ART/'bilin18_l17_source_rank1_tail'
@torch.inference_mode()
def main():
 torch.set_num_threads(6);cfg=json.loads((ROOT/'configs'/f'{STEM.name}.json').read_text())
 for n,h in cfg['hashes'].items():assert hashlib.sha256((ART/n).read_bytes()).hexdigest()==h
 assert hashlib.sha256((ART/'BILIN18_SOURCE_RANK1_TAIL_HYPOTHESES.md').read_bytes()).hexdigest()==cfg['hypothesis_sha256']
 old=np.load(ART/'bilin18_l17_canonical_finite_refinement_rows.npz');p=torch.tensor(old['pre'],device='cuda');src=np.load(ART/'bilin18_l17_covariance_source_confirmation_source_audit.npz');b=torch.tensor(src['basis'][:,0],device='cuda').double();h=torch.tensor(src['sources'][0],device='cuda').double();h1=(h@b)[:,None]*b[None];ht=h-h1;cal=torch.tensor(src['calibration_center'],device='cuda').double();audit={'rank1_calibration_energy':float((cal@b).square().sum()/cal.square().sum()),'rank1_donor_energy':float(h1.square().sum()/h.square().sum()),'source_orthogonality_relative':float(abs((h1*ht).sum())/h.square().sum()),'source_sum_relative':float((h-h1-ht).norm()/h.norm())};assert audit['source_orthogonality_relative']<1e-10;assign=np.arange(512)%64;sources={'full':h[assign].float(),'rank1':h1[assign].float(),'tail':ht[assign].float()};model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];bias=model.blocks[17].mlp.Down_bias.detach();W=model.lm_head.weight.detach()
 def native(q):
  x=F.rms_norm(q,(1152,));return ((x@L.T)*(x@R.T))@D.T+bias
 def lp(s):return (30*torch.tanh(F.rms_norm(s,(1152,))@W.T/30)).double().log_softmax(-1)
 y0=native(p);s0=p+y0;saved={'source_full':h.cpu().numpy(),'source_rank1':h1.cpu().numpy(),'source_tail':ht.cpu().numpy(),'basis':b.cpu().numpy()};summary={}
 for sign,alpha in [('negative',-1),('positive',1)]:
  ys={m:native(p-alpha*source) for m,source in sources.items()};deltas={m:(v-y0).double() for m,v in ys.items()};deltas['additive_response']=deltas['rank1']+deltas['tail'];rows={m:[] for m in cfg['methods']};cross=[]
  for start in range(0,512,16):
   sl=slice(start,start+16);truth=lp(p[sl]+ys['full'][sl]);prob=truth.exp();base=lp(s0[sl]);uf=deltas['full'][sl]
   for m in cfg['methods']:
    # Use direct native states for separate interventions, preserving reference arithmetic.
    state=p[sl]+ys[m][sl] if m in ys else s0[sl]+deltas[m][sl].float();q=lp(state);pq=q.exp();error=(prob*(truth-q)).sum(-1);effect=(pq*(q-base)).sum(-1);rawerr=(uf-deltas[m][sl]).square().sum(-1);raweffect=deltas[m][sl].square().sum(-1);rows[m].append(torch.stack([error,effect,rawerr,raweffect],-1).cpu().numpy())
   ua=deltas['rank1'][sl];ut=deltas['tail'][sl];cross.append(torch.stack([(ua*ut).sum(-1),(uf-ua-ut).square().sum(-1)],-1).cpu().numpy())
  rows={m:np.concatenate(v) for m,v in rows.items()};den=rows['full'];summary[sign]={m:{'KL_recovery_of_full':float(1-v[:,0].sum()/den[:,1].sum()),'raw_recovery_of_full':float(1-v[:,2].sum()/den[:,3].sum()),'own_KL_effect_over_full':float(v[:,1].sum()/den[:,1].sum()),'own_raw_energy_over_full':float(v[:,3].sum()/den[:,3].sum())} for m,v in rows.items()};sl=slice(0,512) if sign=='negative' else slice(512,1024);ref=old['confirmation_effects'][sl];assert np.allclose(den[:,[1,3]],ref,rtol=3e-5,atol=1e-8);audit[sign+'_effect_KL_replay_maxabs']=float(np.abs(den[:,1]-ref[:,0]).max());saved.update({sign+'_'+m:v for m,v in rows.items()});saved[sign+'_cross']=np.concatenate(cross);print(sign,summary[sign],flush=True)
 assert all(np.isfinite(v).all() for v in saved.values());np.savez_compressed(STEM.with_name(STEM.name+'_rows.npz'),**saved);STEM.with_suffix('.json').write_text(json.dumps({'experiment':'H485','config':cfg,'summary':summary,'audits':audit,'columns':['KL_to_full','own_effect_KL','raw_error_to_full','own_raw_energy'],'cross_columns':['raw_rank1_tail_inner_product','raw_nonadditivity_energy'],'unique_completed_documents':16832},indent=2)+'\n')
if __name__=='__main__':main()
