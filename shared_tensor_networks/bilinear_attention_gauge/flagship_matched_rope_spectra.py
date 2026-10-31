"""Weighted feature-covariance spectra for matched pre-/post-RoPE pair data."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_matched_rope_fit import LAWS,prepare
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART


def run(seed=41,n_train=32768,n_val=8192):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    datasets,coverage,gate=prepare(state,seed,n_train,n_val)
    records=[]
    for geometry in ("post","pre"):
        for law in LAWS:
            (u,v,w,_),_=datasets[geometry]["train"][law]
            weights=w.numpy()
            for side,z in (("query",u.numpy()),("key",v.numpy())):
                covariance=z.T@(weights[:,None]*z)
                eig=np.linalg.eigvalsh(covariance)[::-1]
                assert eig[-1]>-1e-8
                records.append(dict(geometry=geometry,law=law,side=side,
                    top64_energy_fraction=float(eig[:64].sum()/eig.sum()),
                    top96_energy_fraction=float(eig[:96].sum()/eig.sum()),
                    participation_rank=float(eig.sum()**2/(eig@eig)),
                    eigenvalues=eig.tolist()))
    output=dict(seed=seed,n_train=n_train,n_val=n_val,
        weight_definition="QK2 score squared times OV output norm squared; heldout gaps 32 and 48 excluded",
        pre_post_identity_gate=gate,coverage=coverage,
        limitation="marginal covariance spectra do not determine the paired bilinear rank-96 optimum",
        records=records)
    (ART/"head0_matched_rope_feature_spectra.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
