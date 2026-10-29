"""Exact readout elimination and reduced residual Jacobian (squared loss).

Objective: .5*||Phi U-Y||_F^2 + .5*ridge*||U||_F^2.
No sample averaging; positive ridge stabilizes the readout solve.
Zero ridge requires full column rank, checked explicitly.
"""
import numpy as np
from direct_bilinear_readout import predict_jacobian


def reduced_residual(theta, x, y, h, ridge=0.):
    if ridge < 0:
        raise ValueError('ridge must be nonnegative')
    p = h*(h+1)//2
    v = np.concatenate([theta, np.eye(p).ravel()])
    phi, jac = predict_jacobian(v, x, h)
    dphi = jac[:, :, :len(theta)]
    # Augmented least squares avoids normal-equation conditioning in the solve.
    Z = np.concatenate([phi, np.sqrt(ridge)*np.eye(p)])
    target = np.concatenate([y, np.zeros((p, y.shape[1]))])
    U, _, rank, _ = np.linalg.lstsq(Z, target, rcond=None)
    if rank != p:
        raise ValueError('readout design lacks numerical full column rank')
    residual = Z@U-target
    # dU solves (Z^T Z)dU = -dZ^T residual-Z^T dZ U.
    _, R = np.linalg.qr(Z, mode='reduced')
    out = np.empty((*residual.shape, len(theta)))
    for k in range(len(theta)):
        dZ = np.concatenate([dphi[:, :, k], np.zeros((p,p))])
        rhs = -dZ.T@residual-Z.T@(dZ@U)
        dU = np.linalg.solve(R, np.linalg.solve(R.T, rhs))
        out[:, :, k] = dZ@U+Z@dU
    return residual, out, U
