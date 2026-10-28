"""Pinned bilin18 loader, early MLP capture and finite write interventions.

Import load_model / trace for notebooks. CLI runs an instrumentation smoke test,
not a reproduction of historical corpus results. All indices are zero based.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(ROOT / 'vendor/basis_aligned/polynomial_causal'))
from bilin18_observed_model_facade import load_bilin18, MODEL_REPO, MODEL_REVISION


def load_model(snapshot=None, device=None, download=False):
    """Find the exact cached revision; download only when explicitly requested."""
    if snapshot is None:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(MODEL_REPO, revision=MODEL_REVISION,
            allow_patterns=['config.json', 'pytorch_model.bin'], local_files_only=not download)
    return load_bilin18(snapshot=Path(snapshot), device=device or (
        'cuda' if torch.cuda.is_available() else 'cpu'), dtype=torch.float32,
        verify_weights_sha256=True)


def encode(text, device):
    import tiktoken
    return torch.tensor([tiktoken.get_encoding('gpt2').encode(text)],
                        dtype=torch.long, device=device)


@torch.no_grad()
def trace(model, ids, layers=(0, 1, 2), replace=None):
    """Return full logits and captured early states.

    replace: optional callable(layer_index, normalized_input, native_write)
    returning the effective write. Later states are recomputed live. Native x0
    and the first-layer value bus are preserved, as in the original model.
    Captures remain on device; use .cpu() before serializing.
    """
    if ids.ndim != 2 or ids.numel() == 0:
        raise ValueError('ids must be a nonempty [batch, time] tensor')
    if min(layers) < 0 or max(layers) >= len(model.transformer.h):
        raise ValueError('layer out of range')
    x0 = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,))
    x, bus, captures = x0, None, {}
    for i, block in enumerate(model.transformer.h):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attn, bus = block.attn(F.rms_norm(mixed, (mixed.shape[-1],)), bus)
        pre = mixed + attn
        z = F.rms_norm(pre, (pre.shape[-1],))
        native = block.mlp(z)
        write = native if replace is None else replace(i, z, native)
        if write.shape != native.shape or not torch.isfinite(write).all():
            raise ValueError('replacement must be a finite write of the native shape')
        x = pre + write
        if i in layers:
            left, right = block.mlp.Left(z), block.mlp.Right(z)
            captures[i] = dict(pre_mlp=pre, normalized_input=z, left=left,
                right=right, products=left*right, native_write=native,
                effective_write=write, post_mlp=x, attention_write=attn)
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (x.shape[-1],))) / 30)
    return logits.float(), captures


def metrics(reference, candidate, ids):
    lr = reference.double().log_softmax(-1)
    lc = candidate.double().log_softmax(-1)
    return dict(mean_output_KL=float((lr.exp()*(lr-lc)).sum(-1).mean()),
        top1_agreement=float((reference.argmax(-1)==candidate.argmax(-1)).float().mean()),
        next_token_CE=float(F.cross_entropy(candidate[:,:-1].reshape(-1,candidate.shape[-1]),
            ids[:,1:].reshape(-1))),
        max_logit_change=float((reference-candidate).abs().max()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot', type=Path)
    ap.add_argument('--download', action='store_true', help='allow pinned checkpoint download (~2.1 GB)')
    ap.add_argument('--device', choices=['cpu','cuda'])
    ap.add_argument('--text', default='The dogs near the house are playing. The cat is sleeping.')
    ap.add_argument('--layers', type=int, nargs='+', default=[0,1,2])
    ap.add_argument('--output', type=Path, default=Path('smoke_results.json'))
    ap.add_argument('--save-activations', type=Path)
    args=ap.parse_args()
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    model, receipt=load_model(args.snapshot,args.device,args.download)
    ids=encode(args.text,next(model.parameters()).device)
    if ids.shape[1]<2: raise ValueError('provide at least two tokens')
    if ids.shape[1]>256: raise ValueError('inspection CLI caps prompts at 256 tokens')
    native,captures=trace(model,ids,args.layers)
    # Independent native forward: GPT.forward uses Block.forward, not this trace.
    head=[]
    handle=model.lm_head.register_forward_hook(lambda module, inputs, output: head.append(output.detach()))
    try:
        with torch.no_grad(): model(ids,ids)
    finally: handle.remove()
    direct=30*torch.tanh(head[0]/30)
    replay=float((native-direct).abs().max())
    if replay>1e-4: raise AssertionError(f'native replay error {replay}')
    identity,_=trace(model,ids,args.layers,lambda i,z,y:y)
    if not torch.equal(native,identity): raise AssertionError('identity edit differs')
    result=dict(kind='instrumentation_smoke_only_not_corpus_reproduction',
        checkpoint=asdict(receipt),torch_version=torch.__version__,device=str(ids.device),
        text=args.text,token_ids=ids.cpu().tolist(),zero_based_layers=args.layers,
        native_replay_max_logit_error=replay,identity_edit_exact=True,
        baseline=metrics(native,native,ids),layers={},zero_write_ablations={})
    gen=torch.Generator(device=ids.device).manual_seed(20261028)
    for i,c in captures.items():
        mlp=model.transformer.h[i].mlp
        rebuilt=mlp.Down(c['products'])+mlp.Down_bias
        err=float((rebuilt-c['native_write']).abs().max())
        if err>1e-4: raise AssertionError('bilinear reconstruction failed')
        # Exact finite difference at the normalized-input boundary. No RMSNorm inside.
        z=c['normalized_input']; d=.01*torch.randn(z.shape,device=z.device,generator=gen)
        with torch.no_grad():
            finite=mlp(z+d)-mlp(z)
            expanded=mlp.Down(mlp.Left(z)*mlp.Right(d)+mlp.Left(d)*mlp.Right(z)+mlp.Left(d)*mlp.Right(d))
        rel=float((finite-expanded).norm()/finite.norm().clamp_min(1e-12))
        if rel>1e-3: raise AssertionError(f'finite bilinear identity failed {rel}')
        result['layers'][str(i)]=dict(weight_shapes={k:list(v.shape) for k,v in mlp.named_parameters()},
            residual_mixing= model.transformer.h[i].lambdas.detach().cpu().tolist(),
            bilinear_reconstruction_max_error=err,finite_identity_relative_error=rel,
            normalized_input_shape=list(z.shape),product_shape=list(c['products'].shape),
            mean_write_norm=float(c['native_write'].norm(dim=-1).mean()))
    # Individual and joint zero WRITE ablations; Down bias is removed too.
    groups=[(i,) for i in args.layers]
    if len(args.layers)>1: groups.append(tuple(args.layers))
    for group in groups:
        edited,_=trace(model,ids,args.layers,lambda i,z,y: torch.zeros_like(y) if i in group else y)
        result['zero_write_ablations'][','.join(map(str,group))]=metrics(native,edited,ids)
    if args.save_activations:
        args.save_activations.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(ids=ids.cpu(),layers={i:{k:v.cpu() for k,v in c.items()} for i,c in captures.items()}),args.save_activations)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'saved':str(args.output),'replay_max_error':replay,
        'layers':args.layers,'tokens':ids.shape[1],'status':'passed'},indent=2))

if __name__=='__main__': main()
