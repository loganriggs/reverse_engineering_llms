"""M109 check fixed intervention samples using native GPT/Block.forward hooks."""
import json
import torch
from fit_pca_encoder import V,sha
from inspect_model import load_model


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    cr=json.loads((V/'cube_causal_native.json').read_text());fr=json.loads((V/'cube_featurizer_native.json').read_text())
    for r in [cr,fr]:assert sha(r['path'])==r['sha256']
    causal=torch.load(cr['path'],weights_only=True);features=torch.load(fr['path'],weights_only=True)
    rows=json.loads((V/'repetition_cube_protocol.json').read_text())['rows']
    rows=[r for r in rows if r.get('case')==0 and r.get('format')=='semicolon' and r.get('vertex') in [0,7]]
    model,mr=load_model();assert mr.weights_sha256==cr['checkpoint_sha256']==fr['checkpoint_sha256'];checks=[]
    for row in rows:
      pid=row['id'];ids=torch.tensor([row['token_ids']],device='cuda')
      conditions=[]
      for layer in [0,1,2]:
        for arm in ['remove_cubic','scramble_10901']:
          sign=-1 if row['vertex']==0 else 1
          d=(-sign*causal['directions'][f'0|semicolon|{layer}'][arm]).float().cuda()
          conditions.append((layer,'causal|'+arm,causal['results'][f'{pid}|{layer}|{arm}']['logits'],d,None))
      if row['vertex']==0:
        for key,record in features.items():
          if key.startswith(pid+'|'):conditions.append((1,'feature|'+key,record['logits'],None,record['effective_write'].cuda()))
      for layer,name,reference,delta,replacement in conditions:
        def edit(module,inputs,output):
          result=output.clone()
          if delta is not None:result[0,-1]+=delta
          else:result[0,-1]=replacement
          return result
        heads=[];h=model.transformer.h[layer].mlp.register_forward_hook(edit);hh=model.lm_head.register_forward_hook(lambda m,i,o:heads.append(o.detach()))
        try:model(ids,ids)
        finally:h.remove();hh.remove()
        actual=(30*torch.tanh(heads[0]/30))[0,-1].cpu();err=float((actual-reference).abs().max());assert err<1e-4
        checks.append(dict(id=pid,layer=layer,condition=name,max_abs_error=err,bit_exact=torch.equal(actual,reference)))
    dest=V/'cube_direct_forward_audit.json';assert not dest.exists();out=dict(status='passed',source_sha256=sha(__file__),causal_receipt_sha256=sha(V/'cube_causal_native.json'),featurizer_receipt_sha256=sha(V/'cube_featurizer_native.json'),checks=checks,scope='Fixed first cube vertices 0/7 for each layer removal and first scrambled control; all eight feature conditions at vertex0. Native model.forward hooks independent of trace implementation; not exhaustive replay.')
    dest.write_text(json.dumps(out,indent=2)+'\n');print('CHECKS',len(checks),'MAX',max(x['max_abs_error'] for x in checks),'EXACT',sum(x['bit_exact'] for x in checks))
if __name__=='__main__':main()
