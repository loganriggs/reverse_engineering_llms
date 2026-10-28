import json
import numpy as np
import torch
from run_bilin18_l17_local_residual_charts import ART,load_full_model,load_program,B
from run_bilin18_l17_residual_term_accounting import residual_bilinear
with torch.inference_mode():
 torch.set_num_threads(6);model=load_full_model('cuda');model.blocks.float();L,R,D=[m.weight.double() for m in [model.blocks[17].mlp.Left,model.blocks[17].mlp.Right,model.blocks[17].mlp.Down]];p=load_program(ART/'bilin18_l17_compiled_partners_m16_factorized.npz','cuda',torch.float64)
 f=load_program(ART/'bilin18_l17_shared_kernel_coverage_6144_frozen.npz','cuda',torch.float64);n=load_program(ART/'bilin18_l17_native_factor_readout_native_frozen.npz','cuda',torch.float64);c=n['core'];Y=torch.tensor(np.load(ART/'bilin18_l17_document_repartition_responses.npy'),device='cuda').reshape(18,4096,1152).double();ids=torch.arange(0,6144,12,device='cuda');aa=f['ux'][ids]@c.T;bb=f['vx'][ids];cc=f['uz'][ids]@c.T;dd=f['vz'][ids]
 raw=(cc@L.T)*(dd@R.T)+(cc@R.T)*(dd@L.T)-(aa@L.T)*(bb@R.T)-(aa@R.T)*(bb@L.T)
 native=2*(B(cc,dd,L,R,D)-B(aa,bb,L,R,D));exact=2*(residual_bilinear(cc,dd,L,R,D,p)-residual_bilinear(aa,bb,L,R,D,p));labels=Y[f['assignment'][ids].long(),ids%3072]
 result={'sampled_references':len(ids),'explicit_factor_algebra_relative':float((raw@D.T-native).norm()/native.norm()),'cached_residual_label_relative':float((exact-labels).norm()/labels.norm()),'native_factor_file_relative':float((L-n['left']).norm()/L.norm()),'scope':'Post-result audit,512 regularly spaced reference examples; checks target algebra, not fresh generalization.'}
 assert result['explicit_factor_algebra_relative']<1e-10;assert result['cached_residual_label_relative']<1e-4;assert result['native_factor_file_relative']==0
 (ART/'bilin18_l17_native_factor_readout_algebra_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
