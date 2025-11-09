"""
Ruedenberg Slater Overlap Integrals Package
============================================

A Python implementation of Ruedenberg's analytical expressions for overlap integrals 
between Slater-type atomic orbitals, with integration of Roothaan's formulations.

References
----------
1. Silver, D.M. and Ruedenberg, K. (1968). "Overlap Integrals over Slater-Type 
   Atomic Orbitals." J. Chem. Phys. 49, 4301-4305.
   
2. Roothaan, C.C.J. (1951). "A Study of Two-Center Integrals Useful in 
   Calculations on Molecular Structure. I." J. Chem. Phys. 19, 1445-1458.

Author: BLZ11
Repository: https://github.com/BLZ11/ruedenberg_slater_overlaps
"""

from .core import overlap, parameters
from .ruedenberg import B, A, f, g, overlap as ruedenberg_overlap
from .roothaan import (
    overlap_1s1s, 
    overlap_2s2s,
    overlap_1s2s,
    overlap_2s2p,
    overlap_2p2p_sigma,
    overlap_2p2p_pi
)

__version__ = "1.0.0"
__author__ = "BLZ11"
__all__ = [
    'overlap',
    'parameters',
    'B', 'A', 'f', 'g',
    'ruedenberg_overlap',
    'overlap_1s1s',
    'overlap_2s2s', 
    'overlap_1s2s',
    'overlap_2s2p',
    'overlap_2p2p_sigma',
    'overlap_2p2p_pi'
]
