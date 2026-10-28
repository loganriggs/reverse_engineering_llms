"""M2: empirical native-write spectra and free-factorization bounds, no model fitting."""
import argparse,hashlib,json
from pathlib import Path
import torch
from inspect_model import load_model,trace
P=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 inp=P/'validation/training_source_context_audit.json';panel=json.loads(inp.read_text());torch.set_num_threads(6);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False);model,receipt=load_model();device=next(model.parameters()).device
 hs={i:[] for i in (0,1,2)};ys={i:[] for i in (0,1,2)};occ=[];prefixes=[];prefixmap={};uniqueindices=[];calls=0
 for pair in panel['rows']:
  for ep in (0,1):
   tokens=pair['token_ids'][ep];ids=torch.tensor([tokens],device=device);_,c=trace(model,ids);calls+=1
   for layer in (0,1,2):
    h=c[layer]['products'][0];net=model.transformer.h[layer].mlp.Down(h);hs[layer].append(h.cpu());ys[layer].append(net.cpu())
   for pos in range(len(tokens)):
    key=tuple(tokens[:pos+1])
    if key not in prefixmap:prefixmap[key]=len(prefixes);prefixes.append(list(key));uniqueindices.append(len(occ))
    occ.append(dict(id=pair['id'],endpoint=ep,position=pos,prefix_id=prefixmap[key]))
 results=[];duplicates=[];saved=[];m1=json.loads((P/'validation/matched_layers.json').read_text());replays=[]
 for layer in (0,1,2):
  H=torch.cat(hs[layer]).double().to(device);Y=torch.cat(ys[layer]).double().to(device);D=model.transformer.h[layer].mlp.Down.weight.double();cn=D.square().sum(0)
  assert len(H)==len(occ)
  first=torch.tensor([uniqueindices[o['prefix_id']] for o in occ],device=device);diff=Y-Y[first];relative=diff.norm(dim=-1)/Y.norm(dim=-1).clamp_min(1e-12)
  duplicates.append(dict(layer=layer,max_absolute_error=float(diff.abs().max()),max_relative_row_error=float(relative.max())));assert relative.max()<1e-4
  expected=next(g for g in m1['geometry_summaries'] if g['layer']==layer and g['family'] is None)
  observed=float((H.square()*cn).sum()/Y.square().sum());replays.append(abs(observed-expected['energy_ratio']));assert replays[-1]<1e-6
  saved.append(dict(layer=layer,H=H.cpu().float(),Y=Y.cpu().float(),column_squared_norms=cn.cpu()))
  for sampling in ('occurrences','unique_prefixes'):
   ind=torch.arange(len(H),device=device) if sampling=='occurrences' else torch.tensor(uniqueindices,device=device)
   h=H[ind];y=Y[ind]
   for centered in (False,True):
    hh=h-h.mean(0) if centered else h;yy=y-y.mean(0) if centered else y
    singular=torch.linalg.svdvals(yy);s2=singular.square();total=s2.sum();channel=(hh.square()*cn).sum();er=float(channel/total);nuclear_rank=float(singular.sum().square()/total);bound=nuclear_rank/4608
    assert er+1e-6>=bound
    cumulative=s2.cumsum(0)/total
    results.append(dict(layer=layer,sampling=sampling,centered=centered,rows=len(ind),channel_net_energy_ratio=er,nuclear_effective_rank=nuclear_rank,nuclear_bound=bound,observed_to_bound_ratio=er/bound,participation_rank=float(total.square()/s2.square().sum()),r90=int(torch.searchsorted(cumulative,.9))+1,r99=int(torch.searchsorted(cumulative,.99))+1,top1_energy_fraction=float(s2[0]/total),mean_component_energy_fraction=float(len(y)*y.mean(0).square().sum()/y.square().sum()),singular_values=singular.cpu().tolist()))
 out=dict(status='completed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),panel_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,positions=len(occ),unique_prefix_count=len(prefixes),prefixes=prefixes,occurrences=occ,unique_indices=uniqueindices,duplicate_checks=duplicates,M1_energy_replay_max_error=max(replays),results=results,
 protocol='Same160 opened endpoints, MLP0/1/2 native product writes excluding Down bias. Occurrence-weighted and unique causal-prefix sampling, raw and centered over corresponding sampling. Channel energies center each channel activation consistently. Nuclear bound concerns unconstrained rank1 matrix contributions on this sample, not executable bilinear compression, semantic rank or corpus generality.')
 args.output.write_text(json.dumps(out,indent=2)+'\n');torch.save(saved,'/tmp/emlp_m2_matrices.pt');print(json.dumps({k:out[k] for k in ['positions','unique_prefix_count','duplicate_checks','M1_energy_replay_max_error']},indent=2))
if __name__=='__main__':main()
