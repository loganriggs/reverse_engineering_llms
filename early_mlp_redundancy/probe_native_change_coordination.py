"""M66 exact native channel-change atlas; no downstream intervention or fitting."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent;V=P/'validation';A=Path('/home/loganriggs/Coding/neuron_circuits/artifacts/bilin18_tensor_decomposition')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def energy(x):return float(x.square().sum())
def main():
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);pp=V/'native_change_coordination_protocol.json';q=json.loads(pp.read_text());op=V/'native_change_coordination.json';assert not op.exists();epath=V/'native_change_coordination_execution.json'
 execution=dict(protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),mc_seed=6601,mc_draws=32,mc_design='One fixed bank of32 uniform exactly-half channel masks, reused for all conditions/layers; both raw-half and doubled-half errors. Report mean/SE and analytic expectations, no outcome selection.',numeric='Metrics use float64 differences of actual saved native float32 products. Expansion uses float64 normalized inputs and weights; numerical disagreement reported against actual native product/write differences. Null/identity denominators explicit.')
 if epath.exists():assert json.loads(epath.read_text())==execution
 else:epath.write_text(json.dumps(execution,indent=2)+'\n')
 banks=[]
 for split,pname,rname,key in [('train','training_causal_coverage_protocol.json','training_causal_coverage.json','training_protocol_sha256'),('source_eval','source_holdout_native_protocol.json','source_holdout_native.json','source_holdout_protocol_sha256')]:
  assert sha(V/pname)==q[key];pr=json.loads((V/pname).read_text());rr=json.loads((V/rname).read_text());sp=Path(rr['samples_path']);assert sha(sp)==rr['samples_sha256'];banks.append((split,pr,torch.load(sp,weights_only=True)))
 assert not set(p['id'] for p in banks[0][1]['pairs'])&set(p['id'] for p in banks[1][1]['pairs'])
 model,receipt=load_model();weights={};gpu={}
 for l in [0,1,2]:
  m=model.transformer.h[l].mlp;assert m.Left.bias is None and m.Right.bias is None and m.Down.bias is None;weights[l]={n:getattr(m,n).weight.detach().cpu() for n in ['Left','Right','Down']};gpu[l]={n:w.cuda().double() for n,w in weights[l].items()}
 dest=A/'emlp_m66';dest.mkdir(exist_ok=True);wp=dest/'weights.pt';torch.save(weights,wp);N=weights[0]['Down'].shape[1];assert N%2==0;gen=torch.Generator().manual_seed(execution['mc_seed']);masks=torch.zeros((32,N),dtype=torch.float64)
 for i in range(32):masks[i,torch.randperm(N,generator=gen)[:N//2]]=1
 maskpath=dest/'masks.pt';torch.save(masks,maskpath);masks=masks.cuda();rows=[];files=[];calls=0
 for split,pr,saved in banks:
  for pair in pr['pairs']:
   blob=dict(pair_id=pair['id'],split=split,captures=[],logits=[],conditions=[])
   for ep in [0,1]:
    ids=torch.tensor([pair['token_ids'][ep]],device='cuda');out,cap=trace(model,ids);calls+=1;assert torch.equal(out[0,-1].cpu(),saved['baselines'][pair['id']+f'|{ep}']['logits']);blob['logits'].append(out[0,-1].cpu());cs={}
    for l in [0,1,2]:cs[l]={k:cap[l][k].cpu() for k in ['normalized_input','left','right','products']};cs[l]['write']=model.transformer.h[l].mlp.Down(cap[l]['products']).cpu()
    blob['captures'].append(cs)
   for ep in [0,1]:
    for l in [0,1,2]:
     W=gpu[l];D=W['Down'];col=D.square().sum(0)
     for region in ['changed_input_positions','final_position','identity']:
      ident=dict(pair_id=pair['id'],split=split,family=pair['family'],endpoint=ep,layer=l,region=region)
      if region=='changed_input_positions' and not pair['source_region_eligible']:
       rows.append(ident|dict(status='ineligible',reason=pair['source_ineligibility_reason']));continue
      pos=pair['changed_positions'] if region=='changed_input_positions' else ([len(pair['token_ids'][ep])-1] if region=='final_position' else []);dp=pair['changed_positions'] if region=='changed_input_positions' else ([len(pair['token_ids'][1-ep])-1] if region=='final_position' else []);cr=blob['captures'][ep][l];cd=blob['captures'][1-ep][l];zr=cr['normalized_input'][:,pos].cuda().double();zd=cd['normalized_input'][:,dp].cuda().double();c=cd['products'][:,dp].cuda().double()-cr['products'][:,pos].cuda().double();native=cd['write'][:,dp].cuda().double()-cr['write'][:,pos].cuda().double();delta=c@D.T;S=energy(delta);ce=c.square().sum((0,1))*col;C=float(ce.sum());ne=energy(native);dz=zd-zr;lr=zr@W['Left'].T;rr=zr@W['Right'].T;dl=dz@W['Left'].T;dr=dz@W['Right'].T;lin=(lr*dr+dl*rr)@D.T;quad=(dl*dr)@D.T;donorlin=((zd@W['Left'].T)*dr+dl*(zd@W['Right'].T))@D.T
      keep=(c.unsqueeze(0)*masks[:,None,None,:])@D.T;raw=(keep-delta).square().sum((1,2,3));dbl=(2*keep-delta).square().sum((1,2,3));extra=(N*C-S)/(N-1);row=ident|dict(status='evaluated',recipient_positions=pos,donor_positions=dp,positions=len(pos),all_addresses_same_token=all(pair['token_ids'][ep][p]==pair['token_ids'][1-ep][d] for p,d in zip(pos,dp)) if pos else None,S=S,C=C,cross_sum=S-C,channel_participation=C*C/float(ce.square().sum()) if C else None,C_over_S=C/S if S else None,native_delta_energy=ne,sum_relative_squared_error=energy(delta-native)/ne if ne else None,expansion_relative_squared_error=energy(lin+quad-delta)/S if S else None,donor_expansion_relative_squared_error=energy(donorlin-quad-delta)/S if S else None,linear_energy=energy(lin),quadratic_energy=energy(quad),linear_only_error=energy(lin-delta),donor_linear_energy=energy(donorlin),donor_linear_only_error=energy(donorlin-delta),predicted_raw_half_error=.25*S+.25*extra,predicted_doubled_half_error=extra,mc_raw_mean=float(raw.mean()),mc_raw_se=float(raw.std()/32**.5),mc_doubled_mean=float(dbl.mean()),mc_doubled_se=float(dbl.std()/32**.5))
      if S:assert row['sum_relative_squared_error']<1e-6 and row['expansion_relative_squared_error']<1e-6 and row['donor_expansion_relative_squared_error']<1e-6
      sk=pair['id']+f'|{ep}|{l}|'+region
      if sk in saved['deltas']:assert torch.equal((cd['write'][:,dp]-cr['write'][:,pos]),saved['deltas'][sk]['delta'])
      rows.append(row);blob['conditions'].append(ident|dict(recipient_positions=pos,donor_positions=dp,mc_raw_errors=raw.cpu(),mc_doubled_errors=dbl.cpu()))
   bp=dest/(hashlib.sha256(pair['id'].encode()).hexdigest()+'.pt');torch.save(blob,bp);files.append(dict(pair_id=pair['id'],split=split,path=str(bp),sha256=sha(bp)));print('PAIRS',len(files),'FORWARDS',calls,flush=True)
 assert len(rows)==1512 and calls==168
 op.write_text(json.dumps(dict(status='completed',protocol_sha256=sha(pp),execution_sha256=sha(epath),source_sha256=sha(Path(__file__)),checkpoint=asdict(receipt),weights_path=str(wp),weights_sha256=sha(wp),masks_path=str(maskpath),masks_sha256=sha(maskpath),files=files,rows=rows,forward_count=calls),indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()
