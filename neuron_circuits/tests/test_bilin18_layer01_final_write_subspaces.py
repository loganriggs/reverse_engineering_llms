from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))

from run_bilin18_layer01_final_write_subspaces import fit_basis,local_fidelity,project  # noqa: E402


def test_fitted_basis_reconstructs_training_span_and_projection_is_orthogonal():
    rows=[torch.tensor([2.,0.,0.]),torch.tensor([0.,3.,0.])]
    basis=fit_basis(rows,2); value=torch.tensor([4.,5.,7.]); predicted=project(value,basis)
    assert torch.allclose(predicted,torch.tensor([4.,5.,0.]),atol=1e-5)
    assert abs(local_fidelity(value,value)-1)<1e-7
