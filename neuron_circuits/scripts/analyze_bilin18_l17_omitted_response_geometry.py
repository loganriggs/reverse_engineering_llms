from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_omitted_response_geometry';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');en=z['energies'];raw=z['raw'];old=np.load(A/'bilin18_l17_baseline_context_confirmation_rows.npz');cells={}
for mode in ['matched','crossed']:
 for oi,os in enumerate([np.arange(16),np.arange(16,32)]):
  for di,ds in enumerate([np.arange(16),np.arange(16,32)]):
   if mode=='matched' and oi!=di:continue
   if mode=='matched':ee=en[:,os,os].sum((0,1));rr=raw[:,os].sum((0,1));klerr=old['errors'][0][:,os,os,2,1].sum();kleff=old['effects'][0][:,os,os,1].sum()
   else:ee=en[:,os][:,:,ds].sum((0,1,2));rr=raw[:,os].sum((0,1))*len(ds);klerr=old['errors'][0][:,os][:,:,ds,2,1].sum();kleff=old['effects'][0][:,os][:,:,ds,1].sum()
   v={'Fisher_error_fraction':float(ee[2]/ee[0]),'raw_error_fraction':float(rr[2]/rr[0]),'omitted_vs_full_Fisher_gain_ratio':float((ee[2]/ee[0])/(rr[2]/rr[0])),'retained_omitted_cross_over_full':float(ee[3]/ee[0]),'retained_energy_over_full':float(ee[1]/ee[0]),'Fisher_parallel_orthogonal_cross_over_omitted':(ee[[4,5,6]]/ee[2]).tolist(),'raw_parallel_orthogonal_cross_over_omitted':(rr[[4,5,6]]/rr[2]).tolist(),'finite_KL_error_fraction':float(klerr/kleff),'finite_to_Fisher_error_fraction_ratio':float((klerr/kleff)/(ee[2]/ee[0]))};cells[f'{mode}_origin{oi}_destination{di}']=v
j['diagnostic_cells']=cells;(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(cells,indent=2))
