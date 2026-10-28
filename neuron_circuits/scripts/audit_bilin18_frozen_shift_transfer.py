import json,hashlib
from pathlib import Path
import torch
from run_bilin18_diverse_causal_tasks import ART
from run_bilin18_qkov_causal_edges import load_full_model
@torch.no_grad()
def main():
 torch.set_num_threads(6);paths={'panel':ART/'bilin18_fresh_shift_panel_captures.pt','transfer':ART/'bilin18_frozen_shift_transfer_captures.pt','locked':ART/'bilin18_mean_context_shift_locked.pt','probes':ART/'bilin18_fresh_shift_probes_locked.pt','old_probes':ART/'bilin18_emission_probe_directions_locked.pt','runner':Path(__file__)};hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()};s=torch.load(paths['panel'],weights_only=False);c=torch.load(paths['transfer'],weights_only=False)['final_tangents'];probes=torch.load(paths['probes'],weights_only=False);old=torch.load(paths['old_probes'],weights_only=False);scale=old['original'].norm();new=torch.stack([probes[f'fresh{i}']/scale for i in range(3)]);Q=torch.stack(list(old.values()))/scale;model=load_full_model('cuda');W=model.lm_head.weight.double();X=s['native_captures'][17].double().cuda();r=(X.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();sat=1-torch.tanh((X/r)@W.T/30).square();p=s['native_logits'].double().cuda().softmax(-1);errs=[]
 for n,gg in c.items():
  d=gg['receiver']['native'].cuda();z=sat*((d/r-X*(X*d).mean(-1,keepdim=True)/r.pow(3))@W.T);actual=s['responses'][n].cuda()
  def norm(v):return (((v-(v*p).sum(-1,keepdim=True)).square()*p).sum(-1)).sqrt()
  errs.extend((norm(z-actual)/norm(actual)).cpu().tolist())
 previous=json.loads((ART/'bilin18_mean_context_shift.json').read_text())['config']['hashes']['locked'];out=dict(hashes=hashes,hashes_unchanged={k:hashlib.sha256(p.read_bytes()).hexdigest()==hashes[k] for k,p in paths.items()},frozen_model_matches_H641=hashes['locked']==previous,prior_probe_orthogonality_max=float((new@Q.T).abs().max()),fresh_probe_orthogonality_max=float((new@new.T-torch.eye(3)).abs().max()),native_logit_chain_max_error=max(errs),new_forwards=0);(ART/'bilin18_frozen_shift_transfer_audit.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
