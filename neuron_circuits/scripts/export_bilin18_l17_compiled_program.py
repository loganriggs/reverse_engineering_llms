#!/usr/bin/env python3
"""Export experimental standalone program with original Down bias absorbed."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np
import torch
from run_bilin18_qkov_causal_edges import load_full_model
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/bilin18_tensor_decomposition';OUT=Path('/home/loganriggs/Documents/Codex/2026-10-21/he/outputs')
def main():
    torch.set_num_threads(6);model=load_full_model('cpu');bias=model.blocks[17].mlp.Down_bias.detach().float().numpy();source=ART/'bilin18_l17_compiled_partners_m16_factorized.npz'
    with np.load(source) as f:arrays={k:f[k].copy() for k in f.files}
    arrays['bias']+=bias;path=OUT/'l17_quadratic_program.npz';np.savez_compressed(path,**arrays)
    shutil.copyfile(ROOT/'scripts/bilin18_compiled_program.py',OUT/'quadratic_program_inference.py')
    metadata={'status':'Experimental approximation; behavior acceptance gates failed; not applied to the original model.','input':'2D float32 pre-MLP residual at L17, shape[N,1152], before RMS normalization.','output':'Approximate complete L17 MLP residual write including original Down_bias. Add output to pre-MLP residual; do not add Down_bias a second time.','normalization':'Full1152-coordinate RMS, epsilon=float32 machine epsilon.','reads':288,'output_directions':16,'context_rank':16,'stored_scalars':sum(v.size for v in arrays.values()),'source_program':str(source.relative_to(ROOT)),'export_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_next_token_ce':2.9567555256653577,'mean_ce_increase':.10843943801592104,'native_top1_agreement':.84375}
    (OUT/'l17_quadratic_program_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(metadata)
if __name__=='__main__':main()
