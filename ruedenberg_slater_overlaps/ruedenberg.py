"""
Ruedenberg's analytical expressions for Slater-type orbital overlaps.

Implementation based on:
Silver, D.M. and Ruedenberg, K. (1968). "Overlap Integrals over Slater-Type 
Atomic Orbitals." J. Chem. Phys. 49, 4301-4305.

This is the exact implementation from the debugged notebook.
"""

import numpy as np
from scipy import special as spec
from functools import lru_cache
from .core import cached_factorial, cached_comb, kahan_sum, parameters


def B(n_A, n_B, l_A, l_B, m, mu, v):
    """Eq.29 from Ruedenberg's paper - CORRECTED"""
    
    # Definitions (Part 1)
    array1 = np.array([mu + v + l_A + l_B, mu + n_A + n_B])
    Delta = np.min(array1) + 1
    array2 = np.array([0, mu - n_B - l_A, v - n_B - l_A + m])
    beta_1 = np.max(array2)
    beta_2 = np.min(np.array([n_A - l_A, v]))
    
    beta = beta_1
    terms = []  # Collect all terms for Kahan summation
    while beta <= beta_2:
        array1 = np.array([0, mu - beta - l_A - l_B, v - beta - l_A - l_B + m])
        beta1_prime = np.max(array1)
        beta2_prime = np.min(np.array([n_B - l_B, v - beta]))
        beta_prime = beta1_prime
        while beta_prime <= beta2_prime:
            array1 = np.array([m, v - beta - beta_prime - l_B + m])
            alpha1 = np.max(array1)
            alpha = alpha1
            while alpha <= l_A:
                array1 = np.array([m, v - beta - beta_prime - alpha + m])
                alpha1_prime = np.max(array1)
                alpha_prime = alpha1_prime
                while alpha_prime <= l_B:
                    lamb = 0
                    while lamb <= mu:
                        # Using cached functions for performance
                        factor = cached_comb(n_A - l_A, beta)
                        factor *= cached_comb(n_B - l_B, beta_prime)  # FIX: was beta2_prime
                        factor *= cached_factorial(Delta) / cached_factorial(mu + beta + beta_prime + l_A + l_B + 1)
                        factor *= cached_comb(l_A, alpha) * cached_comb(l_A, alpha - m) * (-1)**alpha
                        factor *= cached_comb(l_B, alpha_prime) * cached_comb(l_B, alpha_prime - m) * (-1)**(alpha_prime)
                        factor *= cached_comb(alpha + alpha_prime - m, v - beta - beta_prime)
                        factor *= cached_factorial(alpha - alpha_prime + beta + l_B) * cached_factorial(alpha_prime - alpha + beta_prime + l_A)
                        factor *= cached_comb(alpha - alpha_prime + beta + l_B + lamb, lamb) * (-1)**(lamb) 
                        factor *= cached_comb(alpha_prime - alpha + beta_prime + l_A + mu - lamb, mu - lamb)
                        factor *= cached_comb(mu, lamb)
                        terms.append(factor)
                        lamb += 1
                    alpha_prime += 1
                alpha += 1
            beta_prime += 1
        beta += 1
    
    # Use Kahan summation for better precision
    value = kahan_sum(terms)
    return value


def A(n_A, n_B, l_A, l_B, m, mu, v):
    """Eq.28 from Ruedenberg's paper"""
    
    numerator = (2*l_A + 1) * (2*l_B + 1) * cached_comb(l_A + m, m) * cached_comb(l_B + m, m)
    denominator = cached_comb(l_A, m) * cached_comb(l_B, m) * cached_comb(2 * n_A, n_A) * cached_comb(2 * n_B, n_B)
    factor1 = np.sqrt(numerator / denominator)
    
    array = np.array([mu + v + l_A + l_B, mu + n_A + n_B])
    Delta = np.min(array) + 1
    numerator = cached_factorial(n_A + n_B - v) * B(n_A, n_B, l_A, l_B, m, mu, v)
    denominator = cached_factorial(n_A) * cached_factorial(n_B) * cached_comb(2 * mu, mu) * cached_factorial(Delta)
    factor2 = numerator / denominator 
    value = factor1 * factor2 * (-1)**(l_A + l_B)
    return value


def f(rho_A, rho_B, mu):
    """
    Auxiliary function f_μ from Ruedenberg's paper
    
    CORRECT formula (Eq. B1):
    f_μ = exp(-ρ_A) × ₁F₁[μ+1, 2μ+2; ρ_A - ρ_B]
    
    Note: This formula also satisfies:
    - The recurrence relation (Eq. B5)
    - The starting values (Eq. B6, B6')
    - The symmetry property (Eq. B3)
    """
    # Convert to arrays for consistent handling
    rho_A = np.atleast_1d(rho_A)
    rho_B = np.atleast_1d(rho_B)
    
    # Exponential prefactor: exp(-ρ_A)
    exp_term = np.exp(-rho_A)
    
    # Argument for confluent hypergeometric function
    a = mu + 1
    b = 2 * mu + 2
    z = rho_A - rho_B
    
    # Confluent hypergeometric function ₁F₁[a, b; z]
    hyp_value = spec.hyp1f1(a, b, z)
    
    # Complete formula
    result = exp_term * hyp_value
    
    # Return scalar if input was scalar
    return result.item() if result.size == 1 else result


def g(l_A, l_B, n_A, n_B, m, rho_A, rho_B, mu):
    """Eq.25 from Ruedenberg's paper - Optimized with vectorization"""
    arrays = np.array([0, np.abs(l_A - l_B) - mu, mu - (l_A + l_B)])
    v_1 = int(np.max(arrays))
    v_2 = int(n_A + n_B - m)
    
    # Vectorize the summation for better performance
    v_range = np.arange(v_1, v_2 + 1)
    
    if len(v_range) == 0:
        return 0.0
    
    # Compute all A values
    A_values = np.array([A(n_A, n_B, l_A, l_B, m, mu, int(v)) for v in v_range])
    
    # Compute all powers
    powers = (rho_A + rho_B) ** v_range
    
    # Sum
    # Use Kahan summation for better precision
    g_mu = kahan_sum(A_values * powers)
    return g_mu


def overlap(n_A, n_B, l_A, l_B, m, zeta_A, zeta_B, R, tolerance=1e-12):
    """
    Eq.23 from Ruedenberg's paper - CORRECTED VERSION
    
    Parameters:
    -----------
    n_A, n_B : int
        Principal quantum numbers
    l_A, l_B : int
        Angular momentum quantum numbers (0=s, 1=p, 2=d, ...)
    m : int
        Magnetic quantum number (0=sigma, 1=pi, 2=delta, ...)
    zeta_A, zeta_B : float
        Slater orbital exponents
    R : float or array
        Internuclear distance(s)
    tolerance : float
        Convergence tolerance for series summation
        
    Returns:
    --------
    overlap : float or array
        Overlap integral S_AB
    """
    rho_A, rho_B = parameters(zeta_A, zeta_B, R)
    
    # Case 1: Equal exponents (homodiatomic)
    if zeta_A == zeta_B:
        v = np.abs(l_A - l_B)
        terms = []
        while v <= n_A + n_B - m:
            terms.append(A(n_A, n_B, l_A, l_B, m, 0, v) * (2 * rho_A)**v)
            v += 1
        series = kahan_sum(terms)
        overlap = series * np.exp(-rho_A)
    
    # Case 2: Unequal exponents (heterodiatomic)
    else:
        sum_rhos = rho_A + rho_B
        diff_rhos = rho_A - rho_B
        
        # Use log-exp trick for numerical stability
        log_factor_1 = (n_A + 0.5) * np.log(2 * rho_A / sum_rhos)
        log_factor_2 = (n_B + 0.5) * np.log(2 * rho_B / sum_rhos)
        
        factor_1 = np.exp(log_factor_1)
        factor_2 = np.exp(log_factor_2)
        
        # For array inputs, need to handle g() differently
        # g() returns a scalar for each mu, but needs array rho values
        is_array = np.ndim(R) > 0
        
        if is_array:
            # Initialize series as array
            series = np.zeros_like(rho_A)
            mu = 0
            
            while mu <= n_A + n_B:
                # Compute f for all R values at once
                f_mu = f(rho_A, rho_B, mu)
                
                # For g(), we need to evaluate it for each R value
                # This is because g() calls A() which returns scalars
                g_values = np.array([g(l_A, l_B, n_A, n_B, m, rho_A[j], rho_B[j], mu) 
                                     for j in range(len(rho_A))])
                
                term = f_mu * g_values * (diff_rhos)**mu
                series += term
                
                # Early termination check
                if np.all(np.abs(term) < tolerance):
                    break
                
                mu += 1
        else:
            # Scalar case - use Kahan summation
            terms = []
            mu = 0
            prev_term = float('inf')
            
            while mu <= n_A + n_B:
                f_mu = f(rho_A, rho_B, mu)
                g_mu = g(l_A, l_B, n_A, n_B, m, rho_A, rho_B, mu)
                term = f_mu * g_mu * (diff_rhos)**mu
                
                terms.append(term)
                
                # Early termination if converged
                if abs(term) < tolerance and abs(term) < abs(prev_term):
                    break
                
                prev_term = term
                mu += 1
            
            series = kahan_sum(terms)
        
        overlap = factor_1 * factor_2 * series
    
    return overlap
