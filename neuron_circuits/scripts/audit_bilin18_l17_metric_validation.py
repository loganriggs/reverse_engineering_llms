import json,numpy as np,torch
import torch.nn.functional as F
from run_bilin18_l17_local_residual_charts import ROOT,ART,load_full_model
from run_bilin18_l17_fisher_ranked_factors import native_drms
@torch.inference_mode()
def main():
 torch.set_num_threads(6);j=json.loads((ART/'bilin18_l17_metric_compiled_selection_selection.json').read_text());f=np.load(ART/'bilin18_l17_metric_compiled_selection_frozen_projection.npz');c=np.load(ART/'bilin18_l17_calibration_coverage_frozen_bases.npz');p=torch.tensor(c['calibration_pre'][f['val']],device='cuda');model=load_full_model('cuda');model.embedding.float();model.blocks.float();model.lm_head.float();L,R,D=[m.weight.detach() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];W=model.lm_head.weight.double();xf=F.rms_norm(p,(1152,));st=p+((xf@L.T)*(xf@R.T))@D.T+model.blocks[17].mlp.Down_bias;base=F.rms_norm(st,(1152,));s=(p.double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x=p.double()/s;T=torch.tensor(np.load(ART/'bilin18_l17_compiled_omitted_maps_weight_compilation.npz')['maps'],device='cuda');act=torch.einsum('roi,ni->nro',T,x)/s[:,:,None];rows=[]
 for si in range(len(p)):
  th=torch.tanh(base[si:si+1].double()@W.T/30);prob=(30*th).softmax(-1);cap=1-th.square()
  def energy(v):
   dt=(native_drms(st[si:si+1].double(),v)@W.T)*cap;dt-=(prob*dt).sum(-1,keepdim=True);return float(.5*(prob*dt.square()).sum())
  row=[energy(act[si])]
  for idx in [0,j['selected']]:
   enc=torch.tensor(f['encoders'][idx],device='cuda');dec=torch.tensor(f['decoders'][idx],device='cuda');res=act[si]-(act[si]@enc.T)@dec.T;row.append(energy(res))
  rows.append(row)
 rows=np.array(rows);result={'scope':'Full-vocabulary baseline Fisher audit of frozen selected projection on reserved validation states; no reselection and no finite targets.','full_metric_validation_error_fractions':(rows[:,1:].sum(0)/rows[:,0].sum()).tolist(),'methods':['unweighted','selected'],'sampled_validation_error_fractions':[j['candidates'][i]['validation_metric_error_fraction'] for i in [0,j['selected']]]};np.savez_compressed(ART/'bilin18_l17_metric_validation_audit_rows.npz',rows=rows);(ART/'bilin18_l17_metric_validation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
