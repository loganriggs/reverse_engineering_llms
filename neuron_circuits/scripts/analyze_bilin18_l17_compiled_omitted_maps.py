from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/bilin18_tensor_decomposition';stem='bilin18_l17_compiled_omitted_maps';j=json.loads((A/f'{stem}.json').read_text());z=np.load(A/f'{stem}_rows.npz');old=np.load(A/'bilin18_l17_operator_transfer_confirmation_rows.npz');e=z['errors'];d=old['effects'];cells={}
for name,ix in [('low',np.arange(16)),('median',np.arange(16,32))]:
 ee=e[:,:,ix].sum((1,2));dd=d[:,ix].sum((0,1));bb=old['errors'][6,:,ix].sum((0,1));cells[name]={'full_recovery':(1-ee/dd).tolist(),'omitted_recovery':(1-ee/bb).tolist()}
j['strata']=cells;j['storage_scalar_counts']={'omitted_native_three_matrices':3*4480*1152,'dense_16_maps':16*1152*1152,'compressed_maps_and_output_basis':{str(k):16*k*1152+1152*k for k in j['config']['ranks']},'shared_source_directions':1152*16,'source_coefficient_read_in_head_coordinates':128*16};j['exact_score_replay_max_abs']=float(np.max(abs(e[0]-old['errors'][3])));(A/f'{stem}.json').write_text(json.dumps(j,indent=2)+'\n');print('captures',j['weight_energy_capture']);print('scores replay',j['exact_score_replay_max_abs']);
for i,name in enumerate(j['methods']):print(name,'KL',[cells[s]['full_recovery'][i][1] for s in ['low','median']])
