"""
Roothaan's formulations for Slater-type orbital overlaps.

Implementation of selected overlap integrals based on:
Roothaan, C.C.J. (1951). "A Study of Two-Center Integrals Useful in 
Calculations on Molecular Structure. I." J. Chem. Phys. 19, 1445-1458.

These serve as reference implementations for validation and comparison.
"""

import numpy as np
from scipy import special as spec


def overlap_1s1s(R, zeta_A, zeta_B):
    """
    Overlap integral between two 1s orbitals.
    
    Parameters
    ----------
    R : float or array-like
        Internuclear distance(s)
    zeta_A, zeta_B : float
        Orbital exponents
        
    Returns
    -------
    float or array
        S(1s,1s) overlap integral
        
    References
    ----------
    Roothaan (1951), Eq. 25
    From notebook implementation that matches Ruedenberg
    """
    # Handle the special case where zeta_A == zeta_B
    if abs(zeta_A - zeta_B) < 1e-10:
        # Homodiatomic case
        rho = zeta_A * R
        overlap = (1 + rho + rho**2/3) * np.exp(-rho)
        return overlap
    
    # Heterodiatomic case - from notebook
    tau = (zeta_A - zeta_B) / (zeta_A + zeta_B)
    rho = (zeta_A + zeta_B) * R / 2
    kappa = 1/2 * (tau + 1/tau)
    rho_a = zeta_A * R
    rho_b = zeta_B * R
    
    u1 = 1 - kappa
    u2 = 1 + kappa
    term3 = -u1 * (2 * u2 + rho_a) * np.exp(-rho_a)
    term4 = u2 * (2 * u1 + rho_b) * np.exp(-rho_b)
    overlap = np.sqrt(1 - tau**2) / (tau * rho) * (term3 + term4)
    return overlap


def overlap_2s2s(R, zeta_A, zeta_B):
    """
    Overlap integral between two 2s orbitals.
    
    Parameters
    ----------
    R : float or array-like
        Internuclear distance(s)
    zeta_A, zeta_B : float
        Orbital exponents
        
    Returns
    -------
    float or array
        S(2s,2s) overlap integral
        
    References
    ----------
    Roothaan (1951), Eq. 25
    From notebook implementation that matches Ruedenberg
    """
    # Handle the special case where zeta_A == zeta_B
    if abs(zeta_A - zeta_B) < 1e-10:
        # Homodiatomic case
        rho = zeta_A * R
        overlap = (1 + rho + (4/9)*rho**2 + (1/9)*rho**3 + (1/45)*rho**4) * np.exp(-rho)
        return overlap
    
    # Heterodiatomic case - from notebook
    tau = (zeta_A - zeta_B) / (zeta_A + zeta_B)
    rho = (zeta_A + zeta_B) * R / 2
    kappa = 1/2 * (tau + 1/tau)
    rho_a = zeta_A * R
    rho_b = zeta_B * R
    
    u1 = 1 - kappa
    u2 = 1 + kappa
    u3 = 7 - 12 * kappa**2
    u4 = 2 - 3 * kappa
    u5 = 1 - 2 * kappa
    u6 = 2 + 3 * kappa
    u7 = 1 + 2 * kappa
    term3 = -u1 * (2 * u2 * u3 + 4 * u2 * u4 * rho_a + u5 * rho_a**2) * np.exp(-rho_a)
    term4 = u2 * (2 * u1 * u3 + 4 * u1 * u6 * rho_b + u7 * rho_b**2) * np.exp(-rho_b)
    overlap = np.sqrt(1 - tau**2) / (3 * tau * rho) * (term3 + term4)
    return overlap


def overlap_1s2s(R, zeta_A, zeta_B):
    """
    Overlap integral between 1s and 2s orbitals.
    
    Parameters
    ----------
    R : float or array-like
        Internuclear distance(s)
    zeta_A, zeta_B : float
        Orbital exponents (zeta_A for 1s, zeta_B for 2s)
        
    Returns
    -------
    float or array
        S(1s,2s) overlap integral
        
    References
    ----------
    Roothaan (1951), Eq. 25
    """
    # Handle the special case where zeta_A == zeta_B
    if abs(zeta_A - zeta_B) < 1e-10:
        # Homodiatomic case - from notebook
        rho = zeta_A * R
        overlap = (np.sqrt(3)/2) * (1 + rho + 4/9 * rho**2 + 1/9 * rho**3) * np.exp(-rho)
        return overlap
    
    # Heterodiatomic case - from notebook
    tau = (zeta_A - zeta_B) / (zeta_A + zeta_B)
    rho = (zeta_A + zeta_B) * R / 2
    kappa = 1/2 * (tau + 1/tau)
    rho_a = zeta_A * R
    rho_b = zeta_B * R
    
    u1 = 1 - kappa
    u2 = 1 + kappa
    u3 = 2 - 3 * kappa
    u4 = 1 - 2 * kappa
    term3 = -u1 * (2 * u2 * u3 + u4 * rho_a) * np.exp(-rho_a)
    term4 = u2 * (2 * u1 * u3 + 4 * u1 * rho_b + rho_b**2) * np.exp(-rho_b)
    overlap = (np.sqrt(1 - tau**2) / (np.sqrt(3) * tau * rho)) * (term3 + term4)
    return overlap


def overlap_2s2p(R, zeta_A, zeta_B):
    """
    Overlap integral between 2s and 2p (sigma) orbitals.
    
    Parameters
    ----------
    R : float or array-like
        Internuclear distance(s)
    zeta_A, zeta_B : float
        Orbital exponents (zeta_A for 2s, zeta_B for 2p)
        
    Returns
    -------
    float or array
        S(2s,2p_sigma) overlap integral
        
    References
    ----------
    Roothaan (1951), Eq. 25
    From notebook implementation that matches Ruedenberg
    """
    # Handle the special case where zeta_A == zeta_B
    if abs(zeta_A - zeta_B) < 1e-10:
        # Homodiatomic case - from notebook
        rho = zeta_A * R
        overlap = 1/(2 * np.sqrt(3)) * rho * (1 + rho + 7/15 * rho**2 + 2/15 * rho**3) * np.exp(-rho)
        return overlap
    
    # Heterodiatomic case - from notebook
    tau = (zeta_A - zeta_B) / (zeta_A + zeta_B)
    rho = (zeta_A + zeta_B) * R / 2
    kappa = 1/2 * (tau + 1/tau)
    rho_a = zeta_A * R
    rho_b = zeta_B * R
    
    u1 = 1 - kappa
    u2 = 1 + kappa
    u3 = 3 + 4 * kappa
    u4 = 5 + 6 * kappa
    u5 = 2 + 3 * kappa
    u6 = 1 + 2 * kappa
    term3 = (-u1**2) * (6 * u2 * u3 * (1 + rho_a) + 2 * u4 * rho_a**2 + 2 * rho_a**3) * np.exp(-rho_a)
    term4 = u2 * (6 * (u1**2) * u3 * (1 + rho_b) + 4 * u1 * u5 * rho_b**2 + u6 * rho_b**3) * np.exp(-rho_b)
    overlap = np.sqrt((1 + tau)/(1 - tau)) * (1 / (np.sqrt(3) * tau * rho**2)) * (term3 + term4)
    return overlap


def overlap_2p2p_sigma(R, zeta_A, zeta_B):
    """
    Overlap integral between two 2p orbitals (sigma orientation).
    
    Parameters
    ----------
    R : float or array-like
        Internuclear distance(s)
    zeta_A, zeta_B : float
        Orbital exponents
        
    Returns
    -------
    float or array
        S(2p_sigma,2p_sigma) overlap integral
        
    References
    ----------
    Roothaan (1951), Eq. 25
    From notebook implementation that matches Ruedenberg
    """
    # Handle the special case where zeta_A == zeta_B
    if abs(zeta_A - zeta_B) < 1e-10:
        # Homodiatomic case - from notebook (NO rho**2 prefactor!)
        rho = zeta_A * R
        overlap = (-1 - rho - 1/5 * rho**2 + 2/15 * rho**3 + 1/15 * rho**4) * np.exp(-rho)
        return overlap
    
    # Heterodiatomic case - from notebook
    tau = (zeta_A - zeta_B) / (zeta_A + zeta_B)
    rho = (zeta_A + zeta_B) * R / 2
    kappa = 1/2 * (tau + 1/tau)
    rho_a = zeta_A * R
    rho_b = zeta_B * R
    
    u1 = 1 - kappa
    u2 = 1 + kappa
    u3 = 5 + 6 * kappa
    u4 = 5 - 6 * kappa
    term1 = -u1**2 * (48 * u2**2 * (1 + rho_a + 1/2 * rho_a**2) + 2 * u3 * rho_a**3 + 2 * rho_a**4) * np.exp(-rho_a)
    term2 = u2**2 * (48 * u1**2 * (1 + rho_b + 1/2 * rho_b**2) + 2 * u4 * rho_b**3 + 2 * rho_b**4) * np.exp(-rho_b)
    overlap = (1/(np.sqrt(1 - tau**2) * tau * rho**3)) * (term1 + term2)
    return overlap


def overlap_2p2p_pi(R, zeta_A, zeta_B):
    """
    Overlap integral between two 2p orbitals (pi orientation).
    
    Parameters
    ----------
    R : float or array-like
        Internuclear distance(s)
    zeta_A, zeta_B : float
        Orbital exponents
        
    Returns
    -------
    float or array
        S(2p_pi,2p_pi) overlap integral
        
    References
    ----------
    Roothaan (1951), Eq. 25
    From notebook implementation that matches Ruedenberg
    """
    # Handle the special case where zeta_A == zeta_B
    if abs(zeta_A - zeta_B) < 1e-10:
        # Homodiatomic case - from notebook (NO rho**2 prefactor!)
        rho = zeta_A * R
        overlap = (1 + rho + 2/5 * rho**2 + 1/15 * rho**3) * np.exp(-rho)
        return overlap
    
    # Heterodiatomic case - from notebook
    tau = (zeta_A - zeta_B) / (zeta_A + zeta_B)
    rho = (zeta_A + zeta_B) * R / 2
    kappa = 1/2 * (tau + 1/tau)
    rho_a = zeta_A * R
    rho_b = zeta_B * R
    
    u1 = 1 - kappa
    u2 = 1 + kappa
    term3 = -u1**2 * (24 * (1 + rho_a) * u2**2 + 12 * u2 * rho_a**2 + 2 * rho_a**3) * np.exp(-rho_a)
    term4 = u2**2 * (24 * (1 + rho_b) * u1**2 + 12 * u1 * rho_b**2 + 2 * rho_b**3) * np.exp(-rho_b)
    overlap = (1/(np.sqrt(1 - tau**2) * tau * rho**3)) * (term3 + term4)
    return overlap


def overlap_1s2p(R, zeta_A, zeta_B):
    """
    Overlap integral between 1s and 2p (sigma) orbitals.
    
    Parameters
    ----------
    R : float or array-like
        Internuclear distance(s)
    zeta_A, zeta_B : float
        Orbital exponents (zeta_A for 1s, zeta_B for 2p)
        
    Returns
    -------
    float or array
        S(1s,2p_sigma) overlap integral
        
    References
    ----------
    From notebook implementation that matches Ruedenberg
    """
    # Handle the special case where zeta_A == zeta_B
    if abs(zeta_A - zeta_B) < 1e-10:
        # Homodiatomic case - from notebook
        rho = zeta_A * R
        overlap = (1/2) * rho * (1 + rho + 1/3 * rho**2) * np.exp(-rho)
        return overlap
    
    # Heterodiatomic case - from notebook
    tau = (zeta_A - zeta_B) / (zeta_A + zeta_B)
    rho = (zeta_A + zeta_B) * R / 2
    kappa = 1/2 * (tau + 1/tau)
    rho_a = zeta_A * R
    rho_b = zeta_B * R
    
    u1 = 1 - kappa
    u2 = 1 + kappa
    term3 = (-u1**2) * (6 * u2 * (1 + rho_a) + 2 * rho_a**2) * np.exp(-rho_a)
    term4 = u2 * (6 * (u1**2) * (1 + rho_b) + 4 * u1 * rho_b**2 + rho_b**3) * np.exp(-rho_b)
    overlap = np.sqrt((1 + tau)/(1 - tau)) * (1 / (tau * rho**2)) * (term3 + term4)
    return overlap
