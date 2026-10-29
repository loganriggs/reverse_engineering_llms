"""M99 exact endpoint-normalized task components and coefficient gradients."""
import torch
CORNERS=('native_reconstructed','reconstructed_native','reconstructed_reconstructed')
NAMES=tuple(site+'__'+corner for site in ('source','final') for corner in CORNERS)

def components(e,f,key):
 qr=f(e.input_cache[key]['input'].cuda())[0];qd=f(e.input_cache[e.donor_key(key)]['input'].cuda())[0]
 K,B,intact,branches=e.joint_losses(qr,qd,key);active=[b for b in branches if b['region']!='identity'];parts={name:None for name in NAMES}
 for b in active:
  site={'changed_input_positions':'source','final_position':'final'}[b['region']];name=site+'__'+b['corner'];assert parts[name] is None
  parts[name]=b['R']/e.den[e.cache[key]['family'],b['region']]/len(active)
 assert len(active) in (3,6)
 return K,B,parts,len(active)

def gradient(loss,f):
 if loss is None:return torch.zeros_like(f.coefficients,device='cpu',dtype=torch.float64)
 return torch.autograd.grad(loss,f.coefficients,retain_graph=True)[0].detach().cpu().double()
