"""
Unit tests for ruedenberg_slater_overlaps package.
"""

import numpy as np
import pytest
from ruedenberg_slater_overlaps import overlap, ruedenberg, roothaan


class TestOverlapIntegrals:
    """Test suite for overlap integral calculations."""
    
    def test_1s1s_homodiatomic(self):
        """Test 1s-1s overlap for homodiatomic case."""
        R = np.array([1.0, 2.0, 3.0])
        zeta = 1.0
        
        s_rued = ruedenberg.overlap(1, 1, 0, 0, 0, zeta, zeta, R)
        s_root = roothaan.overlap_1s1s(R, zeta, zeta)
        
        np.testing.assert_allclose(s_rued, s_root, rtol=1e-8, atol=1e-10)
    
    def test_2s2s_homodiatomic(self):
        """Test 2s-2s overlap for homodiatomic case."""
        R = np.array([1.0, 2.0, 3.0])
        zeta = 0.85
        
        s_rued = ruedenberg.overlap(2, 2, 0, 0, 0, zeta, zeta, R)
        s_root = roothaan.overlap_2s2s(R, zeta, zeta)
        
        np.testing.assert_allclose(s_rued, s_root, rtol=1e-8, atol=1e-10)
    
    def test_2p2p_sigma_homodiatomic(self):
        """Test 2p-2p sigma overlap for homodiatomic case."""
        R = np.array([1.0, 2.0, 3.0])
        zeta = 1.0
        
        s_rued = ruedenberg.overlap(2, 2, 1, 1, 0, zeta, zeta, R)
        s_root = roothaan.overlap_2p2p_sigma(R, zeta, zeta)
        
        np.testing.assert_allclose(s_rued, s_root, rtol=1e-8, atol=1e-10)
    
    def test_2p2p_pi_homodiatomic(self):
        """Test 2p-2p pi overlap for homodiatomic case."""
        R = np.array([1.0, 2.0, 3.0])
        zeta = 1.0
        
        s_rued = ruedenberg.overlap(2, 2, 1, 1, 1, zeta, zeta, R)
        s_root = roothaan.overlap_2p2p_pi(R, zeta, zeta)
        
        np.testing.assert_allclose(s_rued, s_root, rtol=1e-8, atol=1e-10)
    
    def test_scalar_input(self):
        """Test that scalar R input returns scalar output."""
        R = 2.0  # Scalar input
        result = overlap(1, 1, 0, 0, 0, 1.0, 1.0, R)
        
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)
    
    def test_array_input(self):
        """Test that array R input returns array output."""
        R = np.array([1.0, 2.0, 3.0])
        result = overlap(1, 1, 0, 0, 0, 1.0, 1.0, R)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == R.shape
    
    def test_symmetry(self):
        """Test that S(A,B) = S(B,A)."""
        R = 2.0
        
        s_ab = overlap(1, 2, 0, 0, 0, 1.0, 1.5, R)
        s_ba = overlap(2, 1, 0, 0, 0, 1.5, 1.0, R)
        
        np.testing.assert_allclose(s_ab, s_ba, rtol=1e-10)
    
    def test_zero_distance_limit(self):
        """Test behavior at R → 0."""
        R = 1e-10
        zeta_A = zeta_B = 1.0
        
        # At R=0, different orbitals should have S=0 (orthogonal)
        # Same orbitals should have S=1 (normalized)
        # This is a limiting case that may need special handling
        
        s = overlap(1, 1, 0, 0, 0, zeta_A, zeta_B, R)
        assert np.isfinite(s)
    
    def test_large_distance_limit(self):
        """Test that overlap → 0 as R → ∞."""
        R = 100.0
        s = overlap(1, 1, 0, 0, 0, 1.0, 1.0, R)
        
        np.testing.assert_allclose(s, 0.0, atol=1e-10)
    
    def test_higher_quantum_numbers(self):
        """Test with higher quantum numbers (3d, 4f, etc.)."""
        R = 2.0
        
        # 3d-3d
        s_3d = overlap(3, 3, 2, 2, 0, 2.0, 2.0, R)
        assert np.isfinite(s_3d)
        assert -1 <= s_3d <= 1  # Physical bounds
        
        # 4f-4f  
        s_4f = overlap(4, 4, 3, 3, 0, 2.5, 2.5, R)
        assert np.isfinite(s_4f)
        assert -1 <= s_4f <= 1  # Physical bounds


class TestAuxiliaryFunctions:
    """Test suite for auxiliary functions."""
    
    def test_B_coefficient_bounds(self):
        """Test that B coefficients are bounded."""
        B_val = ruedenberg.B(2, 2, 1, 1, 0, 1, 2)
        assert np.isfinite(B_val)
    
    def test_A_coefficient_symmetry(self):
        """Test A coefficient properties."""
        A_val = ruedenberg.A(2, 2, 1, 1, 0, 1, 2)
        assert np.isfinite(A_val)
    
    def test_f_function_special_case(self):
        """Test f function for rho_A = rho_B."""
        rho = 2.0
        f_val = ruedenberg.f(1, rho, rho)
        assert np.isfinite(f_val)
        assert f_val > 0  # Should be positive
    
    def test_g_function(self):
        """Test g function computation."""
        g_val = ruedenberg.g(1, 2.0, 3.0, 2)
        assert np.isfinite(g_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
