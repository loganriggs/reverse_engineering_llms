"""Integrate the Gaussian radius exactly for homogeneous squared errors."""
import math
import numpy as np


def log_radial_moment(d, power):
    """log E[(R/sqrt(d))**power], R chi_d."""
    return power / 2 * math.log(2 / d) + math.lgamma((d + power) / 2) - math.lgamma(d / 2)


def sphere_from_gaussian(x):
    return x * np.sqrt(x.shape[1]) / np.linalg.norm(x, axis=1, keepdims=True)


def gaussian_mse_from_sphere(squared_errors, d, degree):
    """Unbiased IID-direction estimator; rows must be independent sphere draws.

    Both functions must have the SAME homogeneous degree. Fixed linear input
    transforms preserve this property, so anisotropic Gaussian shifts also work.
    Standard errors remain sample diagnostics, not guaranteed tail coverage.
    """
    values = np.asarray(squared_errors) * math.exp(log_radial_moment(d, 2 * degree))
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def radial_second_moment_ratio(d, degree):
    return math.exp(log_radial_moment(d, 4 * degree) - 2 * log_radial_moment(d, 2 * degree))
