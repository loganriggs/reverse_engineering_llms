"""M104 matched training-only normalized inputs and PCA64 for MLP0/1/2."""
import json
import numpy as np
import torch
from inspect_model import load_model,trace
from fit_pca_encoder import V,A,sha


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    pp=V/'matched_layer_operator_protocol.json';protocol=json.loads(pp.read_text());assert protocol['capture_source_sha256']==sha(__file__)
    for name,h in protocol['inputs'].items():assert sha(V/name)==h
    receipt=json.loads((V/'natural_token_context.json').read_text());assert sha(receipt['cache'])==receipt['cache_sha256'];data=torch.load(receipt['cache'],weights_only=True)
    cohort=json.loads((V/'natural_token_context_cohort.json').read_text());assert {x['document_hash'] for x in cohort['records'][:32]}.isdisjoint({x['document_hash'] for x in cohort['records'][32:]})
    ref=json.loads((V/'native_input_reference.json').read_text());assert sha(ref['path'])==ref['sha256'];old=torch.load(ref['path'],weights_only=True)
    indices=old['indices'];assert len(indices)==4092 and bool((data['documents'][indices]<32).all())
    model,modelreceipt=load_model();assert modelreceipt.weights_sha256==ref['checkpoint_sha256'];zs={k:[] for k in [0,1,2]}
    for doc in range(32):
        _,cap=trace(model,torch.tensor([cohort['token_ids'][doc]],device='cuda'))
        for layer in zs:
            mlp=model.transformer.h[layer].mlp;assert mlp.Left.bias is None and mlp.Right.bias is None and mlp.Down.bias is None
            w=mlp.Down(cap[layer]['products'])[0].cpu();assert torch.equal(w,data['Y'][layer][doc*128:(doc+1)*128]);zs[layer].append(cap[layer]['normalized_input'][0].cpu())
        if doc%8==7:print('REPLAYED',doc+1,flush=True)
    states={};rows=[]
    for layer in zs:
        z=torch.cat(zs[layer]);mean=z[indices].double().mean(0);X=z[indices].double().numpy()-mean.numpy();G=X.T@X/len(X);eig,E=np.linalg.eigh(G);eig=eig[::-1].copy();E=E[:,::-1].copy();B=E[:,:64];assert np.allclose(B.T@B,np.eye(64),atol=1e-10,rtol=1e-10)
        residual=float(np.square(X-(X@B)@B.T).sum()/np.square(X).sum());assert abs(residual-eig[64:].sum()/eig.sum())<1e-10
        mlp=model.transformer.h[layer].mlp;state=dict(z=z,mean=mean,L=mlp.Left.weight.cpu(),R=mlp.Right.weight.cpu(),D=mlp.Down.weight.cpu(),basis=torch.from_numpy(B));states[layer]=state
        if layer==1:
            for key in ['z','mean','L','R','D']:assert torch.equal(state[key],old[key])
        rows.append(dict(layer=layer,centered_variance_captured=1-residual,input_effective_dimension=float(eig.sum()**2/(eig@eig)),boundary_eigenvalues=[float(eig[63]),float(eig[64])]))
    path=A/'emlp_m104_matched_layer_inputs.pt';assert not path.exists();torch.save(dict(states=states,indices=indices,protocol_sha256=sha(pp)),path)
    out=dict(status='passed',source_sha256=sha(__file__),protocol_sha256=sha(pp),path=str(path),sha256=sha(path),checkpoint_sha256=modelreceipt.weights_sha256,forward_count=32,cached_layer_document_writes_exact=96,MLP1_reference_exact=True,rows=rows,scope='Same4092 training positions and32documents perlayer; native normalized inputs and PCA64 perlayer. Equal rank, not equal retained variance. No heldout data or model editing.')
    dest=V/'matched_layer_operator_inputs.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
