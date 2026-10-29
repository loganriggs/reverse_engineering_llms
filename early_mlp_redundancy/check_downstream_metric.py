"""M74 all fixed preflight cases: AD primal and independent central differences."""
import json
import torch
from inspect_model import load_model,trace
from downstream_residual_metric import directional_logits,perturbed_logits,fisher_energy
from downstream_metric_helpers import P,V,A,setup,identity,sha

def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    execution,encoders,cohort,writes=setup();model,receipt=load_model();model.requires_grad_(False)
    rows=[];files=[];dest=A/'emlp_m74_preflight_samples';dest.mkdir(exist_ok=True)
    op=V/'downstream_metric_preflight.json';assert not op.exists();calls=0
    for doc in execution['preflight_documents']:
        ids=torch.tensor([cohort['token_ids'][doc]],device='cuda');base,cap=trace(model,ids);calls+=1
        w=model.transformer.h[1].mlp.Down(cap[1]['products']);assert torch.equal(w[0].cpu(),writes[doc*128:(doc+1)*128])
        for i,(e,enc) in enumerate(encoders):
            rec,*_=enc.reconstruct(w,e['variant']);residual=rec-w;primal,tangent=directional_logits(model,ids,residual);calls+=1
            assert torch.equal(primal,base);energy=float(fisher_energy(base,tangent).mean());tests=[];samples=[]
            for h in execution['finite_difference_steps']:
                plus=perturbed_logits(model,ids,residual,h);minus=perturbed_logits(model,ids,residual,-h);calls+=2
                fd=(plus.double()-minus.double())/(2*h)
                error=float(torch.linalg.vector_norm(fd-tangent)/torch.linalg.vector_norm(tangent))
                fe=float(fisher_energy(base,fd).mean());ferr=abs(fe-energy)/energy
                tests.append(dict(step=h,relative_derivative_error=error,Fisher_energy=fe,relative_Fisher_error=ferr))
                samples.append(dict(step=h,plus=plus.cpu(),minus=minus.cpu()))
                assert error<execution['preflight_max_relative_derivative_error'] and ferr<execution['preflight_max_relative_Fisher_error'],tests
            path=dest/f'doc{doc}_model{i}.pt';torch.save(dict(document=doc,model=identity(e),native=base.cpu(),residual=residual.cpu(),tangent=tangent.cpu(),tests=samples),path)
            files.append(dict(path=str(path),sha256=sha(path)))
            rows.append(identity(e)|dict(document=doc,primal_exact=True,Fisher_energy=energy,tests=tests))
            print('CHECK',doc,i,flush=True)
    result=dict(status='passed',execution_sha256=sha(V/'downstream_metric_execution.json'),source_sha256=sha(__file__),helper_sha256=sha(P/'downstream_metric_helpers.py'),checkpoint_sha256=receipt.weights_sha256,forward_count=calls,rows=rows,files=files)
    op.write_text(json.dumps(result,indent=2)+'\n');print('DONE',calls,flush=True)
if __name__=='__main__':main()
