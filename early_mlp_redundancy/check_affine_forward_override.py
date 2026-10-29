"""M78 prove the affine replacement does not require native MLP products."""
import json
import torch
from inspect_model import load_model,trace
from affine_input_predictor import AffineInputPredictor
from fit_pca_encoder import sha,V
P=V.parent

def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    execution=json.loads((V/'affine_input_execution.json').read_text());cohort=json.loads((V/'natural_token_context_cohort.json').read_text());ids=torch.tensor([cohort['token_ids'][0]],device='cuda');model,receipt=load_model();mlp=model.transformer.h[1].mlp;predict=AffineInputPredictor()
    native,_=trace(model,ids,layers=(0,2));callback,_=trace(model,ids,layers=(0,2),replace=lambda j,z,w:predict(z)+mlp.Down_bias if j==1 else w)
    original_forward=mlp.forward;original_left=mlp.Left.forward
    def forbidden(*args,**kwargs):raise AssertionError('Native Left branch must not run in affine replacement')
    try:
        mlp.forward=lambda z:predict(z)+mlp.Down_bias
        mlp.Left.forward=forbidden
        replaced,_=trace(model,ids,layers=(0,2))
    finally:
        mlp.forward=original_forward;mlp.Left.forward=original_left
    restored,_=trace(model,ids,layers=(0,2));assert torch.equal(native,restored)
    error=float((callback-replaced).abs().max());assert error<=execution['forward_override_max_logit_difference']
    out=dict(status='passed',source_sha256=sha(__file__),predictor_source_sha256=sha(P/'affine_input_predictor.py'),fit_audit_sha256=sha(V/'affine_input_fit_audit.json'),execution_sha256=sha(V/'affine_input_execution.json'),checkpoint_sha256=receipt.weights_sha256,training_document=0,forward_count=4,callback_override_max_logit_difference=error,native_restored_exact=True,native_Left_guard_not_triggered=True,scope='Full MLPforward override consumes only normalized input; native Left call would raise. Exact native restoration verified. Trace does not request MLP1 debug captures, which would independently call Left. No speed benchmark.')
    (V/'affine_input_forward_preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
