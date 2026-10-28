#!/usr/bin/env python3
"""Test whether fitted affine readouts explain apparent quadratic advantages."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "artifacts/bilin18_tensor_decomposition/bilin18_residual_input_linear_redteam"


def ridge(train, target, alpha):
    mean = train.mean(0); scale = train.std(0).clamp_min(1e-8)
    x = (train - mean) / scale; ym = target.mean(0)
    values, vectors = torch.linalg.eigh(x.T @ x)
    coef = vectors @ ((vectors.T @ x.T @ (target - ym)) / (values.clamp_min(0)[:, None] + alpha))
    return {"mean": mean, "scale": scale, "ym": ym, "coef": coef}


def predict(x, fit):
    return fit["ym"] + ((x - fit["mean"]) / fit["scale"]) @ fit["coef"]


def r2(y, p, mean):
    return float(1 - (y-p).square().sum() / (y-mean).square().sum())


def main():
    torch.set_num_threads(6)
    config = {"status": "locked-before-affine-results", "source": "bilin18_residual_input_forms.npz", "alphas": [.01, .1, 1., 10., 100., 1000.], "validation_group": "every fourth discovery row; all eight positions held together", "hypothesis": "Quadratic rank16 beats fitted affine R2 by >=0.10 in at least two early/middle layers", "scope": "Post-pilot alternative explanation test on the same held-out panel; regularization selected solely within discovery. Affine reads use all 1,152 input dimensions and four output directions."}
    (ROOT / "configs/bilin18_residual_input_linear_redteam.json").write_text(json.dumps(config,indent=2)+"\n")
    data=np.load(ROOT / "artifacts/bilin18_tensor_decomposition" / config["source"])
    results={}; saved={}
    for layer in (1,8,12,17):
        x=torch.from_numpy(data[f"discovery_inputs_L{layer}"]).double(); v=torch.from_numpy(data[f"validation_inputs_L{layer}"]).double()
        axes=torch.from_numpy(data[f"directions_L{layer}"]).double()
        y=torch.from_numpy(data[f"discovery_writes_L{layer}"]).double() @ axes.T
        z=torch.from_numpy(data[f"validation_writes_L{layer}"]).double() @ axes.T
        validation=(torch.arange(len(x))//8)%4==0
        scores=[]
        for alpha in config["alphas"]:
            fitted=ridge(x[~validation],y[~validation],alpha)
            scores.append(r2(y[validation],predict(x[validation],fitted),y[~validation].mean(0)))
        alpha=config["alphas"][int(np.argmax(scores))]
        fitted=ridge(x,y,alpha); p=predict(v,fitted)
        results[str(layer)]={"alpha":alpha,"internal_validation_r2":max(scores),"heldout_joint_r2":r2(z,p,y.mean(0)),"axis_r2":[r2(z[:,a],p[:,a],y[:,a].mean()) for a in range(4)]}
        for key,value in fitted.items():saved[f"L{layer}_{key}"]=value.numpy()
        print(layer,results[str(layer)],flush=True)
    np.savez_compressed(STEM.with_suffix('.npz'),**saved)
    # Optional post-fit comparison once the finite decoder archive exists.
    # Import locally because the finite runner reuses predict() from this module.
    decoder_path=ROOT/'artifacts/bilin18_tensor_decomposition/bilin18_residual_input_forms_finite.npz'
    gate=None
    if decoder_path.exists():
        from run_bilin18_residual_input_forms_finite import decode
        fits=np.load(decoder_path)
        for layer in (1,8,12,17):
            axes=torch.from_numpy(data[f'directions_L{layer}']).double()
            x=torch.from_numpy(data[f'validation_inputs_L{layer}']).double()
            y=torch.from_numpy(data[f'validation_writes_L{layer}']).double()@axes.T
            mean=torch.from_numpy(data[f'discovery_writes_L{layer}']).double().mean(0)@axes.T
            pred=torch.stack([decode(x,{k:torch.from_numpy(fits[f'L{layer}_A{a}_R16_{k}']) for k in ('mean','linear','intercept','reads','values')}) for a in range(4)],-1)
            value=r2(y,pred,mean)
            results[str(layer)]['quadratic16_joint_r2']=value
            results[str(layer)]['quadratic16_advantage']=value-results[str(layer)]['heldout_joint_r2']
        gate=sum(results[str(l)]['quadratic16_advantage']>=.1 for l in (1,8,12))>=2
        lines=['# Fitted linear alternative to residual input quadratics','','Regularization was selected on row-grouped discovery data. This is a post-pilot alternative-explanation test on the same held-out panel; no held-out rows entered fitting.','','| Layer | Fitted affine R2 | Quadratic16 R2 | Difference |','|---|---:|---:|---:|']
        for l,r in results.items():lines.append(f"| {l} | {r['heldout_joint_r2']:.3f} | {r['quadratic16_joint_r2']:.3f} | {r['quadratic16_advantage']:+.3f} |")
        lines += ['',f'Locked two-early/middle-layer +0.10 advantage gate: {gate}.','','The exact tangent at the mean is not the best affine predictor. A large advantage over that tangent alone does not establish nonlinear indispensability. The separate finite tests compare rank64 quadratics against these frozen affine fits.']
        STEM.with_name('BILIN18_RESIDUAL_INPUT_LINEAR_REDTEAM.md').write_text('\n'.join(lines)+'\n')
    STEM.with_suffix('.json').write_text(json.dumps({"config":config,"results":results,"gate_supported":gate},indent=2))


if __name__=='__main__':main()
