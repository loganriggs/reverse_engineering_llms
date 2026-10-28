import json,hashlib
from pathlib import Path
import torch,numpy as np
from run_bilin18_diverse_causal_tasks import ART
from run_bilin18_qkov_causal_edges import load_full_model
@torch.no_grad()
def main():
 torch.set_num_threads(6);paths={'coeff':ART/'bilin18_shift_interaction_orders_captures.pt','panel':ART/'bilin18_fresh_shift_panel_captures.pt','manifest':ART/'bilin18_fresh_shift_panel_manifest.json','locked':ART/'bilin18_mean_context_shift_locked.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};coef=torch.load(paths['coeff'],weights_only=False)['coefficients'];panel=torch.load(paths['panel'],weights_only=False);rows=json.loads(paths['manifest'].read_text());means=torch.load(paths['locked'],weights_only=False);lookup={(e['family'],e['form'],e['item']):i for i,e in enumerate(rows)};jj=[lookup[e['family'],e['form']^1,e['item']] for e in rows];sizes={(f,p):float(torch.stack([means[l][f,2*p] for l in range(10,18)]).norm()) for f in ['copy','fact'] for p in [0,1]};scales=torch.tensor([sizes[e['family'],1-e['form']//2]/sizes[e['family'],e['form']//2] for e in rows],dtype=torch.float64);model=load_full_model('cuda');W=model.lm_head.weight.double();X=panel['native_captures'][17].double().cuda();rr=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();sat=1-torch.tanh((X/rr)@W.T/30).square();P=panel['native_logits'].double().cuda().softmax(-1);records=[]
 def norm(z):return (((z-(z*P).sum(-1,keepdim=True)).square()*P).sum(-1)).sqrt()
 for name,cc in coef.items():
  for incoming in ['receiver','donor']:
   for readout in ['receiver','donor']:
    xx=X if readout=='receiver' else X[jj];rs=rr if readout=='receiver' else rr[jj];ss=sat if readout=='receiver' else sat[jj]
    def read(d):
     dd=d.cuda();return ss*((dd/rs-xx*(xx*dd).mean(-1,keepdim=True)/rs.pow(3))@W.T)
    cs=[c*scales[:,None]**k for k,c in enumerate(cc[incoming])];full=read(sum(cs));native=read(cs[0]);den=norm(full-native)
    for k in [1,2,3,4]:
     pred=read(sum(cs[:k+1]));err=norm(pred-full)/den
     for i,e in enumerate(rows):records.append(dict(id=e['id'],family=e['family'],form=e['form'],probe=name,incoming=incoming,readout=readout,order=k,scale=float(scales[i]),error=float(err[i])))
 result=dict(records=records,shift_frobenius_norm={f'{f}_{p}':v for (f,p),v in sizes.items()},hashes=hashes,hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},scope='Post-result amplitude diagnostic. Each matched shift upscaled to broad Frobenius norm, each broad downscaled to matched norm within family. Changes intervention, not actual prompt. No refitting; full polynomial reference at same amplitude.');(ART/'bilin18_shift_order_amplitude_audit.json').write_text(json.dumps(result,indent=2))
 print('SIZES',sizes)
 for f in ['copy','fact']:
  for pair in [0,1]:
   print(f,pair,[(k,round(np.mean([r['error'] for r in records if r['family']==f and r['form']//2==pair and r['incoming']=='donor' and r['readout']=='donor' and r['order']==k]),4)) for k in [1,2,3,4]])
if __name__=='__main__':main()
