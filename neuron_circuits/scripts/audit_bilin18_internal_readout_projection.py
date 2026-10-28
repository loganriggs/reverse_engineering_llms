"""H635 supplemental oracle output-metric projection of fixed internal bases."""
import json,hashlib
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ART
from run_bilin18_qkov_causal_edges import load_full_model
@torch.no_grad()
def main():
 torch.set_num_threads(6);stem='bilin18_internal_response_subspaces';cp=ART/f'{stem}_captures.pt';sp=ART/'bilin18_cross_task_response_panel_captures.pt';mp=ART/'bilin18_cross_task_response_panel_manifest.json';pp=ART/'bilin18_emission_probe_directions_locked.pt';paths={'captures':cp,'source':sp,'manifest':mp,'probes':pp,'runner':Path(__file__)};hashes={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in paths.items()};c=torch.load(cp,weights_only=False);s=torch.load(sp,weights_only=False);probes=torch.load(pp,weights_only=False);rows=json.loads(mp.read_text());names=list(probes);fitnames=json.loads((ART/f'{stem}.json').read_text())['config']['fit_directions'];model=load_full_model('cuda');W=model.lm_head.weight.double();X=c['native_captures'][17].double().cuda();records=[];normalmax=0.
 for i,e in enumerate(rows):
  x=X[i];rr=(x.square().mean()+torch.finfo(torch.float32).eps).sqrt();sat=1-torch.tanh((x/rr)@W.T/30).square();p=s['native_logits'][i].double().cuda().softmax(0)
  def readout(d):return sat*((d/rr-x*(d*x).mean(-1,keepdim=True)/rr.pow(3))@W.T)
  def center(z):return z-(z*p).sum(-1,keepdim=True)
  truth=center(torch.stack([s['responses'][n][i] for n in names]).cuda());skip=center(readout(torch.stack([probes[n]*c['skip_scales'][17] for n in names]).cuda()))
  for kind in ['full','innovation']:
   for rank in [32,100]:
    basis=center(readout(c['bases'][f'17_{kind}'][:rank].cuda()));target=truth if kind=='full' else truth-skip;gram=(basis*p)@basis.T;rhs=(target*p)@basis.T;coef=torch.linalg.solve(gram,rhs.T).T;pred=coef@basis
    if kind=='innovation':pred+=skip
    res=truth-pred;normalmax=max(normalmax,float(((res*p)@basis.T).abs().max()));er=((res.square()*p).sum(-1)/(truth.square()*p).sum(-1)).sqrt()
    for k,n in enumerate(names):records.append(dict(id=e['id'],family=e['family'],form=e['form'],split=e['split'],probe=n,direction_split='fit' if n in fitnames else 'heldout',kind=kind,rank=rank,error=float(er[k])))
 print('NORMAL',normalmax);(ART/f'{stem}_output_optimal_audit.json').write_text(json.dumps(dict(records=records,normal_equation_max=normalmax,hashes=hashes,hashes_unchanged={n:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[n] for n,p in paths.items()},new_forwards=0),indent=2))
if __name__=='__main__':main()
