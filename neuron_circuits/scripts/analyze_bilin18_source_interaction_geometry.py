from pathlib import Path
import json,math
import numpy as np
A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition');S='bilin18_source_identity_composition';j=json.loads((A/f'{S}.json').read_text());idx={(r['id'],r['a'],r['b']):r for r in j['records']};rows=[]
for f in [0,1]:
 for kind in ['coherent','mixed']:
  energy=axisenergy=residualenergy=actualresidual=scalarerr=scalaractual=0.;n=0
  for r in j['records']:
   a,b=r['a'],r['b']
   if r['format']!=f or a==0 or b==0 or ((a==b)!=(kind=='coherent')):continue
   rid=r['id'];y=np.array(r['candidate_logit_changes']);inter=y-np.array(idx[rid,a,0]['candidate_logit_changes'])-np.array(idx[rid,0,b]['candidate_logit_changes'])+np.array(idx[rid,0,0]['candidate_logit_changes']);center=inter-inter.mean();receiver=r['receiver_index'];other=[z for z in range(len(y)) if z!=receiver];direction=np.full(len(y),-1/(len(y)-1));direction[receiver]=1;direction/=np.linalg.norm(direction);axis=center@direction;res=center-axis*direction;yc=y-y.mean();yr=yc-(yc@direction)*direction;targets=sorted({r['variant_indices'][a],r['variant_indices'][b]});rest=[z for z in other if z not in targets];err=inter[targets].mean()-inter[rest].mean();target=y[targets].mean()-y[rest].mean();energy+=center@center;axisenergy+=axis*axis;residualenergy+=res@res;actualresidual+=yr@yr;scalarerr+=err*err;scalaractual+=target*target;n+=1
  rows.append({'format':f,'kind':kind,'n':n,'receiver_axis_fraction_of_centered_interaction_squared_norm':axisenergy/energy,'nonreceiver_contrast_additive_relative_RMS_error':math.sqrt(residualenergy/actualresidual),'target_specificity_additive_RMS_error':math.sqrt(scalarerr/n),'target_specificity_RMS':math.sqrt(scalaractual/n),'target_specificity_additive_relative_RMS_error':math.sqrt(scalarerr/scalaractual)})
(A/f'{S}_interaction_geometry.json').write_text(json.dumps({'summary':rows,'note':'Orthogonal receiver-vs-rest axis is descriptive output geometry, not unique causal mediation. Relative errors describe prediction; squared-energy share is not a causal share.'},indent=2)+'\n');print(json.dumps(rows,indent=2))
