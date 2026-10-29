"""Coordinate attribution versus finite interventions for a fixed input span."""
import numpy as np
from native_projected_quadratic import write, action


def rotation(k, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((k, k)))
    return q * np.where(np.diag(r) < 0, -1., 1.)


def components(L, R, D, z, delta, basis):
    parts = (delta @ basis)[:, None] * basis.T
    combined = parts.sum(0)
    midpoint = action(L, R, D, z + combined / 2, parts)
    single = action(L, R, D, z, parts) + write(L, R, D, parts)
    response = write(L, R, D, z + combined) - write(L, R, D, z)
    interaction = write(L, R, D, combined) - write(L, R, D, parts).sum(0)
    return parts, midpoint, single, response, interaction


def summaries(parts, midpoint, single, response, interaction):
    energy = lambda x: float(np.square(x).sum())
    participation = lambda x: float(np.square(np.square(x).sum(1).sum()) / np.square(np.square(x).sum(1)).sum()) if energy(x) else None
    den = energy(response)
    return dict(input_participation=participation(parts), midpoint_participation=participation(midpoint), single_participation=participation(single), response_energy=den, midpoint_energy_sum=energy(midpoint), single_energy_sum=energy(single), interaction_energy=energy(interaction), interaction_relative_energy=energy(interaction)/den if den else None, midpoint_net_over_sum=den/energy(midpoint) if energy(midpoint) else None)
