"""Training-only native-product subset inventory; no transformer execution."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;V=P/'validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load():
 protocol=json.loads((V/'product_subset_protocol.json').read_text())
 assert sha(V/'bilinear_input_evaluation_models.json')==protocol['manifest_sha256']
 assert sha(V/'native_input_reference.json')==protocol['reference_receipt_sha256']
 ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256']
 r=torch.load(ref['path'],weights_only=True);z=r['z'][r['indices']].double().numpy()
 H=(z@r['L'].double().numpy().T)*(z@r['R'].double().numpy().T)
 return protocol,H,r['D'].double().numpy()
def spaces(W):
 yield 'native',W,0.
 for e in json.loads((V/'bilinear_input_evaluation_models.json').read_text())['models']:
  if e['variant']!='input':continue
  assert sha(e['path'])==e['sha256'];s=torch.load(e['path'],weights_only=True)['state'];D=s['features.dictionary'].double().numpy();Q,_=np.linalg.qr(D,mode='reduced')
  c=s['output_mean'].double().numpy()+float(s['output_scale'])*s['features.bias'].double().numpy();offset=c-(c@Q)@Q.T
  yield f"{e['kind']}_{e['seed']}_{e['base_arm']}",Q.T@W,float(offset@offset)
def run():
 torch.set_num_threads(1);p,H,W=load();rows=[];inventories=[];orders={};n=len(H)
 for name,C,offset2 in spaces(W):
  Y=H@C.T;E=float(np.sum(Y*Y));score=np.mean(H*H,axis=0)*np.sum(C*C,axis=0);diag=float(score.sum()*n)
  inventory={'space':name,'dynamic_energy':E,'individual_energy':diag,'net_cross_energy':E-diag,'affine_offset_energy':n*offset2,'diagonal_over_total':diag/E};inventories.append(inventory)
  rankings={'energy':np.argsort(-score,kind='stable'),'random9001':np.random.default_rng(9001).permutation(H.shape[1]),'random9002':np.random.default_rng(9002).permutation(H.shape[1])}
  orders[name]={k:v.tolist() for k,v in rankings.items()}
  for method,order in rankings.items():
   for size in p['subsets']:
    ix=order[:size];S=Y.copy() if size==H.shape[1] else H[:,ix]@C[:,ix].T;D=Y-S;se=float(np.sum(S*S));dot=float(np.sum(S*Y));alpha=dot/se;err=float(np.sum(D*D));comp=float(np.sum((Y-alpha*S)**2))
    row={'space':name,'method':method,'size':size,'raw_relative_squared':err/E,'raw_affine_relative_squared':err/(E+n*offset2),'retained_individual_fraction':float(score[ix].sum()/score.sum()),'cosine':dot/np.sqrt(se*E),'scalar':alpha,'compensated_relative_squared':comp/E,'compensated_affine_relative_squared':comp/(E+n*offset2),'selected_energy':se,'selected_omitted_dot':float(np.sum(S*D))}
    if size==4608:assert row['raw_relative_squared']<1e-20 and abs(alpha-1)<1e-10
    rows.append(row)
  print(name,'diagonal/total',diag/E,'half energy-selected',next(r['raw_relative_squared'] for r in rows if r['space']==name and r['method']=='energy' and r['size']==2304),flush=True)
 out={'status':'completed_pending_audit','protocol_sha256':sha(V/'product_subset_protocol.json'),'source_sha256':sha(__file__),'training_positions':n,'inventories':inventories,'rankings':orders,'rows':rows};assert len(rows)==255
 (V/'product_subset_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run()
