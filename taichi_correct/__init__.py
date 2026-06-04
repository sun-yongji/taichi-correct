"""
TaiChi-Correct: C6 Symmetry Error Correction Engine

Detects and corrects prediction errors across 6 experts by exploiting
C6 coupling structure. Decomposes residuals into symmetry eigenmodes
and provides conformal uncertainty bounds.

M5 module of the TaiChi Matrix (CCF OSS 2026).
"""

from .error_correction import TaiChiCorrector, c6_residual_decomposition

__all__ = ["TaiChiCorrector", "c6_residual_decomposition"]
__version__ = "0.1.0"
