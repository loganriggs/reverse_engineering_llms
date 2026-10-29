"""M77 freeze training-only native input reference and evaluation arithmetic."""
import json
import torch
import numpy as np
from inspect_model import load_model,trace
from fit_pca_encoder import sha,V,A
P=V.parent

def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    pp=V/'native_input_operator_protocol.json';p=json.loads(pp.read_text())
    for name,digest in p['references'].items():assert sha(V/name)==digest
    toy=json.loads((V/'native_input_operator_toys.json').read_text());assert toy['status']=='passed' and toy['operator_source_sha256']==sha(P/'native_input_operator.py')
    execution=dict(protocol_sha256=sha(pp),operator_source_sha256=sha(P/'native_input_operator.py'),toy_sha256=sha(V/'native_input_operator_toys.json'),dtype='Analytic operators in CPUfloat64; predicted changes castfloat32 before native patch. Endpoint z captured in nativefloat32.',midpoint_local_relative_squared_tolerance=1e-6,midpoint_response_error_atol=1e-10,midpoint_response_error_rtol=1e-8,predictors=p['predictors'])
    ep=V/'native_input_operator_execution.json';assert not ep.exists();ep.write_text(json.dumps(execution,indent=2)+'\n')
    cache=json.loads((V/'natural_token_context.json').read_text());assert sha(cache['cache'])==cache['cache_sha256'];data=torch.load(cache['cache'],weights_only=True);indices=data['unique_indices'][data['documents'][data['unique_indices']]<32]
    assert len(indices)==4092 and bool((data['documents'][indices]<32).all());cohort=json.loads((V/'natural_token_context_cohort.json').read_text())
    assert {x['document_hash'] for x in cohort['records'][:32]}.isdisjoint({x['document_hash'] for x in cohort['records'][32:]})
    model,receipt=load_model();mlp=model.transformer.h[1].mlp;assert mlp.Left.bias is None and mlp.Right.bias is None and mlp.Down.bias is None
    captures=[]
    for doc in range(32):
        ids=torch.tensor([cohort['token_ids'][doc]],device='cuda');out,cap=trace(model,ids);w=mlp.Down(cap[1]['products']);assert torch.equal(w[0].cpu(),data['Y'][1][doc*128:(doc+1)*128]);captures.append(cap[1]['normalized_input'][0].cpu())
    z=torch.cat(captures);mean=z[indices].double().mean(0);independent=z[indices].numpy().astype(np.float64).mean(0);assert np.allclose(mean.numpy(),independent,rtol=1e-13,atol=1e-13)
    state=dict(z=z,indices=indices,mean=mean,L=mlp.Left.weight.cpu(),R=mlp.Right.weight.cpu(),D=mlp.Down.weight.cpu())
    path=A/'emlp_m77_input_reference.pt';assert not path.exists();torch.save(state,path)
    result=dict(status='frozen',source_sha256=sha(__file__),execution_sha256=sha(ep),path=str(path),sha256=sha(path),checkpoint_sha256=receipt.weights_sha256,forward_count=32,training_positions=4092,training_documents=list(range(32)),evaluation_documents_used=0,mean_independent_max_error=float(np.abs(mean.numpy()-independent).max()),scope='Native normalized MLP1 inputs; exact cached native-write replay on all32trainingdocuments. Mean anchor is an algebraic reference, not necessarily a natural RMS-normalized state.')
    (V/'native_input_reference.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
