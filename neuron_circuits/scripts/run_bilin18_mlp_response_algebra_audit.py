"""Higher precision arithmetic audit supporting H556; no new interventions."""
import torch,json,hashlib
import torch.nn.functional as F
from run_bilin18_diverse_causal_tasks import ROOT,ART
from run_bilin18_qkov_causal_edges import load_full_model
@torch.inference_mode()
def main():
 torch.set_num_threads(6);p=ART/'bilin18_mlp_response_decomposition.json';j=json.loads(p.read_text());inputs=torch.load(ART/'bilin18_mlp_response_decomposition_inputs.pt',weights_only=False);model=load_full_model('cuda');m=model.blocks[8].mlp.double().eval();xs=torch.stack([v['x'] for v in inputs.values()]).cuda();ds=torch.stack([v['delta'] for v in inputs.values()]).cuda();errors=[];relative=[]
 for scale in j['config']['scales']:
  z=F.rms_norm(xs,(xs.shape[-1],)).double();u=F.rms_norm(xs+scale*ds,(xs.shape[-1],)).double();e=u-z;actual=m(u)-m(z);pred=m.Down(m.Left(z)*m.Right(e)+m.Left(e)*m.Right(z)+m.Left(e)*m.Right(e));errors.append(float((actual-pred).abs().max()));relative.extend(((actual-pred).norm(dim=-1)/actual.norm(dim=-1)).tolist())
 audit=dict(float64_algebra_max_error=max(errors),float64_algebra_max_relative_error=max(relative),scope='Same float32 normalized endpoints promoted to float64; bf16-rounded model weights promoted to float64. Confirms algebra, not exact real-number equality.',script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest());assert max(errors)<1e-9;j['audits'].update(audit);p.write_text(json.dumps(j,indent=2));print(audit)
if __name__=='__main__':main()
