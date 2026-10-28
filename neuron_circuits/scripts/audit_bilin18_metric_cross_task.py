"""Independent float64 finite-difference check of H600 bilinear cross term."""
import json,torch
from run_bilin18_diverse_causal_tasks import ART
from run_bilin18_qkov_causal_edges import load_full_model
@torch.inference_mode()
def main():
 torch.set_num_threads(6);p=ART/'bilin18_metric_cross_task_captures.pt';d=torch.load(p,weights_only=False);model=load_full_model('cuda');m=model.blocks[11].mlp.double();errors=[];replay=[]
 for j,c in d['cache'].items():
  z=d['inputs'][j[0]].cuda().double();dc=c['dc'].cuda().double();e=c['de'].cuda().double();full=(m(z+dc+e)-m(z+dc))-(m(z+e)-m(z));record=d['vectors'][j]['full'].cuda().double();errors.append(float((full-record).norm()/full.norm()));cross=m.Down(m.Left(dc)*m.Right(e)+m.Left(e)*m.Right(dc))-m.Down(torch.zeros_like(m.Left(dc)));replay.append(float((full-cross).norm()/full.norm()))
 out=dict(cases=len(errors),float64_finite_difference_vs_saved_float32_max_relative_error=max(errors),float64_finite_difference_identity_max_relative_error=max(replay));(ART/'bilin18_metric_cross_task_float64_audit.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
