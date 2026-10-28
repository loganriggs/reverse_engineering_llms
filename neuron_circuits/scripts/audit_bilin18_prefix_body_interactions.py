"""Verify body subtraction reaches block0 residual; inspect skip/bus coefficients."""
import torch,json
from run_bilin18_diverse_causal_tasks import ART
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_reader_operation_format import run
@torch.inference_mode()
def main():
 torch.set_num_threads(6);d=torch.load(ART/'bilin18_prefix_body_interactions_captures.pt',weights_only=False);rows=json.loads((ART/'bilin18_prefix_boundary_control_manifest.json').read_text());ex={r['id']:r for r in rows};model=load_full_model('cuda')
 for m in [model.embedding,model.blocks,model.lm_head]:m.float().eval()
 checks=[]
 for w in [0,1]:
  k=f'{w}-0-0-1-1';cap={};ids=torch.tensor([ex[k]['ids']],device='cuda');delta=d['components'][k]['full'].cuda().unsqueeze(0);saved=[]
  for patched in [False,True]:
   hs=[]
   if patched:
    def h(m,args,out):v=out.clone();v[:,4:]-=delta.to(v);return v
    hs.append(model.blocks[0].mlp.register_forward_hook(h))
   def post(m,args,out):cap['post']=out[0].cpu()
   hs.append(model.blocks[0].register_forward_hook(post))
   try:run(model,ids)
   finally:
    for h in hs:h.remove()
   saved.append(cap['post'].clone())
  change=saved[1]-saved[0];checks.append(dict(wording=w,body_subtraction_relative_error=float((change[:,4:]+delta.cpu()).norm()/delta.norm()),prefix_position_max_abs_change=float(change[:,:4].abs().max()),applied_body_norm=float(change[:,4:].norm())))
 coeff=[dict(layer=l,residual_skip=b.lambdas.detach().cpu().tolist(),persistent_value_mix=float(b.attn.lamb)) for l,b in enumerate(model.blocks) if l<12];r=dict(checks=checks,coefficients=coeff);(ART/'bilin18_prefix_body_interactions_application_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
