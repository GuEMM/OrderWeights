"""Code for 'What measures of high-order interdependence count: order weights, local geometry
and sampling bias' (G. Menesse and H. J. Jensen).

Modules
-------
measures  coefficients c_k of the measures and their order weights lambda_m
bias      exact and simulated plug-in bias for discrete variables
gaussian  exact and simulated plug-in bias for Gaussian variables
slice     the pairwise-triplet slice of the simplex for N = 3
paths     linear-mixing and maximum-entropy paths from the uniform distribution
fourier   Fourier coefficients of Boolean distributions
dyadic    dyadic and triadic distributions of James & Crutchfield
cache     storage of slow simulations in results/
"""
from . import measures, bias, gaussian, slice, paths, fourier, dyadic, cache  # noqa: F401
