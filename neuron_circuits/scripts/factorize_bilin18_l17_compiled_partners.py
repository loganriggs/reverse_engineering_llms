#!/usr/bin/env python3
"""Factor fixed compiled forms; verify storage and full state/response predictions."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bilin18_compiled_program import load_program,factorize_program,forward_normalized
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_l8_source_support_factorial import collect_sources
from run_bilin18_l8_execution_alignment import angle_matched_payload
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition'
@torch.inference_mode()
def main():
    torch.set_num_threads(6);primary=json.loads((ART/'bilin18_l17_compiled_partners.json').read_text());cfg=primary['parent_config'];layer=17;model=load_full_model('cuda');model.embedding.float();model.blocks.float();pre,hw,_=collect_sources(model,np.load(ROOT/cfg['corpus'],mmap_mode='r'),{**cfg,'layer':layer});block=model.blocks[layer]
    output={}
    for m in [4,16]:
        dense=load_program(ART/f'bilin18_l17_compiled_partners_m{m}.npz','cuda');fact,errors=factorize_program(dense,32)
        path=ART/f'bilin18_l17_compiled_partners_m{m}_factorized.npz';np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in fact.items()});fact=load_program(path,'cuda')
        stateerr=0.;stateenergy=0.;resperr=0.;respenergy=0.
        for h in range(9):
            ref=hw[:,h];source=torch.linalg.qr(block.attn.c_proj.weight.reshape(1152,9,128)[:,h]).Q;gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+100*layer+h);noises=[torch.randn(len(pre),128,generator=gen,device='cuda') for _ in range(2)]
            for base in cfg['baselines']:
                raw=pre if base=='intact' else pre-ref;sign=-1 if base=='intact' else 1;x=F.rms_norm(raw,(1152,));fd=forward_normalized(x,dense);ff=forward_normalized(x,fact)
                for p in [ref]+[angle_matched_payload(raw,ref,source,n) for n in noises]:
                    for scale in cfg['scales']:
                        y=F.rms_norm(raw+sign*scale*p,(1152,));yd=forward_normalized(y,dense);yf=forward_normalized(y,fact)
                        stateerr+=float((yd-yf).square().sum());stateenergy+=float(yd.square().sum());resperr+=float(((yd-fd)-(yf-ff)).square().sum());respenergy+=float((yd-fd).square().sum())
        count=sum(v.numel() for v in fact.values());base=primary['programs'][str(m)];actual=base['natural_write_r2']>=.9 and all(primary['results'][f'{b}_actual_s{s:g}_m{m}']['pooled']>=.9 for b in cfg['baselines'] for s in cfg['scales']);universal=actual and all(primary['results'][f'{b}_angle_s{s:g}_m{m}']['pooled']>=.9 for b in cfg['baselines'] for s in cfg['scales'])
        output[str(m)]={'stored_scalars':count,'storage_fraction':count/base['native_scalars'],'float32_bytes':4*count,'max_relative_form_reconstruction_error':max(errors),'state_relative_rms_difference':(stateerr/stateenergy)**.5,'response_relative_rms_difference':(resperr/respenergy)**.5,'dense_mac_estimate':1152*base['reads']+m*base['reads']*32+m*32+m*base['reads']+m*1152,'compact_actual_forward':actual and count/base['native_scalars']<.1 and base['reads']<=288,'compact_universal_forward':universal and count/base['native_scalars']<.1,'scope':'Storage optimization after initial dense gate outcome; fixed rank32 analytic bound, no fitting or changed candidate predictions. Fidelity numbers inherited from dense program only after this equivalence audit.'}
        print(m,output[str(m)],flush=True)
    (ART/'bilin18_l17_compiled_partners_factorization.json').write_text(json.dumps(output,indent=2))
if __name__=='__main__':main()
