"""M1: matched native-channel deletion and signed interference for MLP0/1/2."""
import argparse,hashlib,json
from pathlib import Path
import torch
from inspect_model import load_model,trace
from probe_mlp1_group_deletion import metrics
from probe_training_task_responses import competence
P=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 inp=P/'validation/training_source_context_audit.json';panel=json.loads(inp.read_text());dpath=P/'validation/mlp1_group_deletion.json';d1=json.loads(dpath.read_text())
 perm=torch.randperm(4608,generator=torch.Generator().manual_seed(2026102901))
 sets={k:v for k,v in d1['removed_sets'].items() if k.startswith(('quarter','half'))};sets['remove_87_5']=perm[:4032].tolist();sets['all_products']=list(range(4608))
 torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);model,receipt=load_model();device=next(model.parameters()).device
 rows=[];geometry=[];samples=[];gsamples=[];calls=0;replay=[]
 prior={(r['id'],r['endpoint'],r['kind']):r for r in d1['rows']}
 for pi,pair in enumerate(panel['rows']):
  for ep in (0,1):
   ids=torch.tensor([pair['token_ids'][ep]],device=device);base,c=trace(model,ids);calls+=1
   for layer in (0,1,2):
    mlp=model.transformer.h[layer].mlp;prod=c[layer]['products'];D=mlp.Down.weight;y=c[layer]['native_write']
    net=mlp.Down(prod);netnorm=net.double().norm(dim=-1);colnorm=D.double().norm(dim=0)
    magnitudes=prod.double().abs()*colnorm;mass=magnitudes.sum(-1);channel_energy=magnitudes.square().sum(-1);netenergy=netnorm.square();near=netnorm<1e-12
    geometry.append(dict(id=pair['id'],family=pair['family'],endpoint=ep,layer=layer,positions=netnorm.numel(),sum_channel_norms=float(mass.sum()),sum_net_norms=float(netnorm.sum()),sum_channel_energy=float(channel_energy.sum()),sum_net_energy=float(netenergy.sum()),sum_squared_channel_mass=float(mass.square().sum()),near_zero_net_positions=int(near.sum())))
    if pi==0 and ep==0:gsamples.append(dict(row=len(geometry)-1,products=prod.cpu(),Down=D.cpu(),net=net.cpu()))
    for kind,indices in sets.items():
     keep=torch.ones(4608,device=device);keep[indices]=0
     edited=mlp.Down(prod*keep)+mlp.Down_bias
     def patch(i,z,v):return edited if i==layer else v
     lg,_=trace(model,ids,replace=patch);calls+=1
     met=metrics(base,lg,ids)
     rows.append(dict(id=pair['id'],family=pair['family'],endpoint=ep,layer=layer,kind=kind,removed_fraction=len(indices)/4608,**met,local_relative_squared_error=float((edited-y).double().square().sum()/y.double().square().sum()),mean_write_cosine=float(torch.nn.functional.cosine_similarity(y,edited,dim=-1).mean()),mean_write_norm_ratio=float((edited.norm(dim=-1)/y.norm(dim=-1).clamp_min(1e-12)).mean()),task=competence(lg[0,-1],pair['answer_groups'][ep],int(pair['variants'][ep]))))
     if layer==1 and kind in d1['removed_sets']:
      old=prior[(pair['id'],ep,kind)];replay.extend(abs(met[k]-old[k]) for k in met)
     if pi==0 and ep==0 and kind in ('quarter_0','half_01','remove_87_5','all_products'):samples.append(dict(row=len(rows)-1,native=base.cpu(),edited=lg.cpu(),ids=ids.cpu(),native_write=y.cpu(),edited_write=edited.cpu()))
  print('completed',pi+1,80,flush=True)
 summaries=[];gsummary=[]
 for layer in (0,1,2):
  for family in [None]+list(panel['families']):
   gg=[g for g in geometry if g['layer']==layer and (family is None or g['family']==family)]
   totals={k:sum(g[k] for g in gg) for k in ['positions','sum_channel_norms','sum_net_norms','sum_channel_energy','sum_net_energy','sum_squared_channel_mass','near_zero_net_positions']}
   gsummary.append(dict(layer=layer,family=family,**totals,contribution_mass_ratio=totals['sum_channel_norms']/totals['sum_net_norms'],energy_ratio=totals['sum_channel_energy']/totals['sum_net_energy'],signed_pair_interference_over_channel_energy=(totals['sum_net_energy']-totals['sum_channel_energy'])/totals['sum_channel_energy'],energy_weighted_participation=totals['sum_squared_channel_mass']/totals['sum_channel_energy']))
   for kind in sets:
    rr=[r for r in rows if r['layer']==layer and r['kind']==kind and (family is None or r['family']==family)]
    fields=['mean_KL','last_KL','observed_next_token_dCE','top1_agreement','local_relative_squared_error','mean_write_cosine','mean_write_norm_ratio']
    summaries.append(dict(layer=layer,family=family,kind=kind,n=len(rr),**{f:sum(r[f] for r in rr)/len(rr) for f in fields},binary_rate=sum(r['task']['binary'] for r in rr)/len(rr)))
 assert max(replay)==0
 out=dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),D1_sha256=hashlib.sha256(dpath.read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,removed_sets=sets,forward_count=calls,D1_MLP1_replay_max_error=max(replay),geometry=geometry,geometry_summaries=gsummary,rows=rows,summaries=summaries,
 protocol='Same160 opened training endpoints, same channel-index masks across MLP0/1/2 (not semantic alignment). Four quarters/two halves/one87.5%mask/full products, bias retained. Native upstream states, live downstream. Contribution geometry excludes bias. Sum norms/energies before ratios; report near-zero net positions. Energy ratio>1 means net destructive pair interference on this panel, not every pair cancels. No fitting/selection or new task exposure; extreme mask has one replicate.')
 args.output.write_text(json.dumps(out,indent=2)+'\n');torch.save(dict(samples=samples,geometry=gsamples),'/tmp/emlp_m1_samples.pt')
 print(json.dumps([g for g in gsummary if g['family'] is None],indent=2))
if __name__=='__main__':main()
