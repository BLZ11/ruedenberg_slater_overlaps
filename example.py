#!/usr/bin/env python
"""
Simple example demonstrating the use of ruedenberg_slater_overlaps package.

This script calculates and plots overlap integrals for various orbital combinations,
comparing Ruedenberg and Roothaan implementations where available.
"""

import numpy as np
import matplotlib.pyplot as plt
from ruedenberg_slater_overlaps import overlap, ruedenberg, roothaan


def main():
    """Main example demonstrating package usage."""
    
    print("=" * 60)
    print("Ruedenberg Slater Overlap Integrals - Example")
    print("=" * 60)
    print()
    
    # Example 1: Simple H2 molecule (1s-1s)
    print("Example 1: H₂ molecule (1s-1s overlap)")
    print("-" * 40)
    
    R_H2 = 1.4  # Typical H-H bond distance in a.u.
    S_H2 = overlap(1, 1, 0, 0, 0, 1.0, 1.0, R_H2)
    print(f"Bond distance: R = {R_H2} a.u.")
    print(f"Overlap integral: S = {S_H2:.6f}")
    print()
    
    # Example 2: Heteronuclear diatomic (CO)
    print("Example 2: CO molecule (various overlaps)")
    print("-" * 40)
    
    R_CO = 2.13  # C-O bond distance in a.u.
    
    # 1s-1s overlap
    zeta_C_1s = 5.67
    zeta_O_1s = 7.66
    S_1s1s = overlap(1, 1, 0, 0, 0, zeta_C_1s, zeta_O_1s, R_CO)
    print(f"1s(C)-1s(O): S = {S_1s1s:.6f}")
    
    # 2p-2p sigma overlap
    zeta_C_2p = 1.72
    zeta_O_2p = 2.25
    S_2p2p_sigma = overlap(2, 2, 1, 1, 0, zeta_C_2p, zeta_O_2p, R_CO)
    print(f"2p_σ(C)-2p_σ(O): S = {S_2p2p_sigma:.6f}")
    
    # 2p-2p pi overlap
    S_2p2p_pi = overlap(2, 2, 1, 1, 1, zeta_C_2p, zeta_O_2p, R_CO)
    print(f"2p_π(C)-2p_π(O): S = {S_2p2p_pi:.6f}")
    print()
    
    # Example 3: Plotting overlap vs distance
    print("Example 3: Overlap vs internuclear distance")
    print("-" * 40)
    
    R_values = np.linspace(0.5, 6.0, 100)
    
    # Calculate overlaps
    S_1s1s = overlap(1, 1, 0, 0, 0, 1.0, 1.0, R_values)
    S_2s2s = overlap(2, 2, 0, 0, 0, 1.0, 1.0, R_values)
    S_2p2p_sigma = overlap(2, 2, 1, 1, 0, 1.0, 1.0, R_values)
    S_2p2p_pi = overlap(2, 2, 1, 1, 1, 1.0, 1.0, R_values)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(R_values, S_1s1s, 'b-', linewidth=2, label='1s-1s')
    plt.plot(R_values, S_2s2s, 'r--', linewidth=2, label='2s-2s')
    plt.plot(R_values, S_2p2p_sigma, 'g-.', linewidth=2, label='2p-2p (σ)')
    plt.plot(R_values, S_2p2p_pi, 'm:', linewidth=2, label='2p-2p (π)')
    
    plt.xlabel('Internuclear Distance R (a.u.)', fontsize=12)
    plt.ylabel('Overlap Integral S', fontsize=12)
    plt.title('Slater Orbital Overlaps vs Distance\n(Homodiatomic, ζ=1.0)', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 6.0)
    plt.ylim(-0.2, 1.0)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    print("Plotting overlap integrals...")
    plt.savefig('overlap_vs_distance.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'overlap_vs_distance.png'")
    plt.show()
    print()
    
    # Example 4: Comparison with Roothaan
    print("Example 4: Validation against Roothaan")
    print("-" * 40)
    
    R_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Calculate with both methods
    S_rued = ruedenberg.overlap(1, 1, 0, 0, 0, 1.0, 1.0, R_test)
    S_root = roothaan.overlap_1s1s(R_test, 1.0, 1.0)
    
    print(f"{'R (a.u.)':<10} {'Ruedenberg':<15} {'Roothaan':<15} {'Difference':<15}")
    print("-" * 55)
    for i, R in enumerate(R_test):
        diff = S_rued[i] - S_root[i]
        print(f"{R:<10.1f} {S_rued[i]:<15.6f} {S_root[i]:<15.6f} {diff:<15.2e}")
    
    rmse = np.sqrt(np.mean((S_rued - S_root)**2))
    print(f"\nRMSE: {rmse:.2e}")
    print()
    
    # Example 5: Higher quantum numbers
    print("Example 5: Higher quantum numbers")
    print("-" * 40)
    
    R = 3.0
    orbitals = [
        ("3s-3s", 3, 3, 0, 0, 0),
        ("3p-3p σ", 3, 3, 1, 1, 0),
        ("3d-3d σ", 3, 3, 2, 2, 0),
        ("4f-4f σ", 4, 4, 3, 3, 0),
    ]
    
    print(f"Overlaps at R = {R} a.u., ζ = 2.0:")
    print()
    for name, n_A, n_B, l_A, l_B, m in orbitals:
        S = overlap(n_A, n_B, l_A, l_B, m, 2.0, 2.0, R)
        print(f"{name:<12}: S = {S:>10.6f}")
    
    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
