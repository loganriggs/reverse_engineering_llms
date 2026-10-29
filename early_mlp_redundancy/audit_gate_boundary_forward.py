"""M110 sample of one-sided limits independently replayed with native forward hooks."""
import json
import torch
from fit_pca_encoder import V,sha
from inspect_model import load_model


def main():
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_grad_enabled(False)
    r=json.loads((V/'gate_boundary_native.json').read_text());assert sha(r['path'])==r['sha256'];saved=torch.load(r['path'],weights_only=True)
    pairs={x['id']:x for x in json.loads((V/'whole_task_native_protocol.json').read_text())['pairs']};model,mr=load_model();assert mr.weights_sha256==r['checkpoint_sha256']
    checked=set();results=[]
    for row in r['rows']:
      ident=(row['seed'],row['arm'])
      if row['status']!='evaluated' or ident in checked:continue
      checked.add(ident);pid=row['pair_id'];key=f"{pid}|{row['seed']}|{row['arm']}";rec=saved[key];ids=torch.tensor([pairs[pid]['token_ids'][0]],device='cuda')
      for mode in ['left','right','native_path']:
        write=rec[mode+'_write'].float().cuda()+model.transformer.h[1].mlp.Down_bias
        def edit(m,i,o):
          result=o.clone();result[0,-1]=write;return result
        heads=[];h=model.transformer.h[1].mlp.register_forward_hook(edit);hh=model.lm_head.register_forward_hook(lambda m,i,o:heads.append(o.detach()))
        try:model(ids,ids)
        finally:h.remove();hh.remove()
        out=(30*torch.tanh(heads[0]/30))[0,-1].cpu();ref=rec['outputs'][mode];err=float((out-ref).abs().max());assert err<1e-4
        results.append(dict(key=key,mode=mode,max_error=err,bit_exact=torch.equal(out,ref)))
    assert len(checked)==4 and len(results)==12
    dest=V/'gate_boundary_forward_audit.json';assert not dest.exists();dest.write_text(json.dumps(dict(status='passed',source_sha256=sha(__file__),native_sha256=sha(V/'gate_boundary_native.json'),checks=results),indent=2)+'\n');print('CHECKS',len(results),'EXACT',sum(x['bit_exact'] for x in results))
if __name__=='__main__':main()
