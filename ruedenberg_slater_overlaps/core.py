"""
Core utilities and helper functions for Slater overlap integrals.

This module provides common utilities used by both Ruedenberg and Roothaan 
implementations.
"""

import numpy as np
from scipy import special as spec
from functools import lru_cache


@lru_cache(maxsize=2048)
def cached_factorial(n):
    """
    Cached factorial computation for performance.
    
    Parameters
    ----------
    n : int
        Non-negative integer
    
    Returns
    -------
    int
        n! (n factorial)
    """
    return spec.factorial(int(n), exact=True)


@lru_cache(maxsize=2048)
def cached_comb(n, k):
    """
    Cached binomial coefficient computation for performance.
    
    Parameters
    ----------
    n : int
        Total number of items
    k : int
        Number of items to choose
    
    Returns
    -------
    int
        Binomial coefficient C(n, k)
    """
    if k < 0 or n < k:
        return 0
    return spec.comb(int(n), int(k), exact=True)


def kahan_sum(values):
    """
    Kahan (compensated) summation algorithm for improved numerical precision.
    Reduces rounding errors when summing many floating-point numbers.
    
    Parameters
    ----------
    values : array-like
        Values to sum
    
    Returns
    -------
    float
        High-precision sum of values
    """
    if len(values) == 0:
        return 0.0
    
    sum_val = 0.0
    compensation = 0.0
    
    for value in values:
        y = value - compensation
        t = sum_val + y
        compensation = (t - sum_val) - y
        sum_val = t
    
    return sum_val


def parameters(zeta_A, zeta_B, R):
    """
    Calculate dimensionless parameters for overlap integrals.
    
    Parameters
    ----------
    zeta_A : float
        Orbital exponent for center A
    zeta_B : float
        Orbital exponent for center B
    R : float or array-like
        Internuclear distance(s)
    
    Returns
    -------
    tuple
        (rho_A, rho_B) where rho_i = R * zeta_i
        
    References
    ----------
    Silver & Ruedenberg (1968), Eq. 3
    """
    rho_A = R * zeta_A
    rho_B = R * zeta_B
    return (rho_A, rho_B)


def overlap(n_A, n_B, l_A, l_B, m, zeta_A, zeta_B, R):
    """
    General interface for Slater-type orbital overlap integrals.
    
    This function provides a unified interface that can switch between
    Ruedenberg and Roothaan implementations based on availability and
    accuracy requirements.
    
    Parameters
    ----------
    n_A, n_B : int
        Principal quantum numbers (n >= 1)
    l_A, l_B : int
        Azimuthal quantum numbers (0 <= l < n)
    m : int
        Magnetic quantum number (|m| <= min(l_A, l_B))
    zeta_A, zeta_B : float
        Orbital exponents
    R : float or array-like
        Internuclear distance(s)
    
    Returns
    -------
    float or array
        Overlap integral value(s)
    """
    # Import here to avoid circular dependency
    from .ruedenberg import overlap as ruedenberg_overlap
    
    # Use Ruedenberg implementation by default
    return ruedenberg_overlap(n_A, n_B, l_A, l_B, m, zeta_A, zeta_B, R)
