"""
TaiChi-Correct: C6 Symmetry Error Correction Engine

Detects and corrects prediction errors across 6 experts/modes by
exploiting the C6 coupling structure. Decomposes residuals into
symmetry eigenmodes and provides conformal uncertainty bounds.

Core insight: errors in one expert propagate predictably through
C6-coupled neighbors, enabling detection and iterative correction.

M5 of TaiChi Matrix (CCF OSS 2026).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# C6 Constants
# ---------------------------------------------------------------------------

NUM_EXPERTS: int = 6

C6_COUPLING: np.ndarray = np.array([
    [1.00, 0.50, 0.00, 0.00, 0.00, 0.50],
    [0.50, 1.00, 0.50, 0.00, 0.00, 0.00],
    [0.00, 0.50, 1.00, 0.50, 0.00, 0.00],
    [0.00, 0.00, 0.50, 1.00, 0.50, 0.00],
    [0.00, 0.00, 0.00, 0.50, 1.00, 0.50],
    [0.50, 0.00, 0.00, 0.00, 0.50, 1.00],
], dtype=np.float64)

# Precomputed: coupling eigenvector decomposition
_C_EIGENVALS: np.ndarray
_C_EIGENVECS: np.ndarray
_C_EIGENVALS, _C_EIGENVECS = np.linalg.eigh(C6_COUPLING)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CorrectionReport:
    """Result of one C6 correction pass."""
    original: np.ndarray  # (n_samples, 6) original predictions
    corrected: np.ndarray  # (n_samples, 6) after correction
    residuals: np.ndarray  # (n_samples, 6) original - consensus
    anomaly_mask: np.ndarray  # (n_samples, 6) bool, flagged BEFORE correction
    initial_z_scores: np.ndarray  # (n_samples, 6) Z-scores before correction
    confidence: np.ndarray  # (n_samples,) per-sample confidence
    iterations: int  # number of refinement iterations used
    convergence: List[float] = field(default_factory=list)  # residual norm per iteration


@dataclass
class ConformalSet:
    """Conformal prediction interval."""
    lower: np.ndarray  # (n_samples,)
    upper: np.ndarray  # (n_samples,)
    alpha: float  # significance level
    calib_nonconformity: np.ndarray  # calibration scores


# ---------------------------------------------------------------------------
# C6 Residual Decomposition
# ---------------------------------------------------------------------------

def c6_residual_decomposition(
    residuals: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Decompose (n, 6) residuals into 6 C6 symmetry eigenmodes.

    Returns dict mapping mode_name → (n,) component magnitudes.
    The 6 modes correspond to:
      λ₀ (1.97): uniform/common mode — global bias
      λ₁ (1.50): dipole I — opposing expert pairs
      λ₂ (1.50): dipole II — 60° rotated dipole
      λ₃ (0.50): quadrupole I — 4-expert oscillation
      λ₄ (0.50): quadrupole II
      λ₅ (0.03): hexapole — checkerboard, largest noise mode
    """
    assert residuals.ndim == 2 and residuals.shape[1] == 6

    components: Dict[str, np.ndarray] = {}
    coords = residuals @ _C_EIGENVECS  # (n, 6) in eigenbasis

    mode_names = [
        ("uniform_bias", 5),   # λ₅=1.966 — common shift (largest eigenvalue)
        ("dipole_I", 4),       # λ₄=1.500 — first dipole
        ("dipole_II", 3),      # λ₃=1.500 — second dipole
        ("quadrupole_I", 2),   # λ₂=0.500 — first quadrupole
        ("quadrupole_II", 1),  # λ₁=0.500 — second quadrupole
        ("hexapole_noise", 0), # λ₀=0.034 — noise floor
    ]

    for name, col in mode_names:
        components[name] = coords[:, col] * _C_EIGENVALS[col]

    return components


def eigenmode_magnitude(residuals: np.ndarray) -> np.ndarray:
    """Return (6,) total energy per eigenmode across all samples."""
    components = c6_residual_decomposition(residuals)
    mags = np.zeros(6)
    for i, (name, _) in enumerate([
        ("uniform_bias", 0),
        ("dipole_I", 1),
        ("dipole_II", 2),
        ("quadrupole_I", 3),
        ("quadrupole_II", 4),
        ("hexapole_noise", 5),
    ]):
        mags[i] = np.sqrt(np.mean(components[name] ** 2))
    return mags


# ---------------------------------------------------------------------------
# TaiChiCorrector
# ---------------------------------------------------------------------------

class TaiChiCorrector:
    """C6 symmetry-guided prediction error corrector.

    Takes 6-expert predictions, detects outlier experts via C6 coupling
    consensus, and corrects using iterative refinement.

    Parameters
    ----------
    threshold : float
        Z-score threshold for flagging an expert as anomalous (default 1.5).
    max_iter : int
        Maximum refinement iterations (default 5).
    tol : float
        Convergence tolerance on residual norm change (default 1e-4).
    coupling : np.ndarray
        C6 coupling matrix, default uses standard hex coupling.
    """

    _DEFAULT_COUPLING: np.ndarray = C6_COUPLING.copy()

    def __init__(
        self,
        threshold: float = 1.5,
        max_iter: int = 5,
        tol: float = 1e-4,
        coupling: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        assert 0 < threshold <= 10, f"threshold={threshold} out of range"
        self.threshold = threshold
        self.max_iter = max_iter
        self.tol = tol
        self.coupling = coupling.copy() if coupling is not None else self._DEFAULT_COUPLING.copy()
        self.rng = rng or np.random.default_rng()
        self._calib_scores: Optional[np.ndarray] = None
        self._n_experts = self.coupling.shape[0]

    # ------------------------------------------------------------------
    # Core correction
    # ------------------------------------------------------------------

    def _consensus(self, predictions: np.ndarray) -> np.ndarray:
        """Compute C6-weighted consensus for each sample.

        predictions: (n_samples, 6)
        returns: (n_samples, 6) C6-expected value per expert
        """
        row_sums = self.coupling.sum(axis=1, keepdims=True)
        consensus = predictions @ self.coupling.T / row_sums.T
        return consensus

    def _anomaly_scores(
        self, predictions: np.ndarray, consensus: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-expert anomaly scores using global residual scale.

        Critical: we use a single global std across all experts and samples
        as reference, NOT per-expert std. Per-expert std would normalize away
        the systematic noise of a single faulty expert.
        """
        residuals = predictions - consensus
        global_std = float(np.std(residuals))
        global_std = max(global_std, 1e-10)
        z_scores = np.abs(residuals / global_std)
        anomaly_mask = z_scores > self.threshold
        return z_scores, anomaly_mask

    def correct(
        self,
        predictions: np.ndarray,
        verbose: bool = False,
    ) -> CorrectionReport:
        """Apply C6 iterative refinement to correct 6-expert predictions.

        Parameters
        ----------
        predictions : ndarray of shape (n_samples, 6)
            Raw predictions from the 6 experts.

        verbose : bool
            If True, store per-iteration residual norms.

        Returns
        -------
        CorrectionReport with corrected predictions and diagnostics.
        """
        assert predictions.ndim == 2 and predictions.shape[1] == 6

        original = predictions.copy()
        corrected = original.copy()
        convergence: List[float] = []

        # Detect anomalies BEFORE any correction
        consensus = self._consensus(corrected)
        initial_z_scores, initial_anomaly_mask = self._anomaly_scores(corrected, consensus)
        residuals = corrected - consensus
        prev_norm = float(np.sqrt(np.mean(residuals ** 2)))

        for iteration in range(self.max_iter):
            consensus = self._consensus(corrected)
            residuals = corrected - consensus

            # Anomaly detection
            z_scores, anomaly_mask = self._anomaly_scores(corrected, consensus)

            # Correction: pull anomalous experts toward C6 consensus
            # Non-anomalous experts use 10% consensus blend for stability
            alpha_anomalous = 0.5  # aggressive correction for flagged
            alpha_normal = 0.1     # mild blend for non-flagged

            for e in range(6):
                flagged = anomaly_mask[:, e]
                if np.any(flagged):
                    corrected[flagged, e] = (
                        (1.0 - alpha_anomalous) * corrected[flagged, e]
                        + alpha_anomalous * consensus[flagged, e]
                    )
                # Apply light consensus blending to all experts
                not_flagged = ~anomaly_mask[:, e]
                if np.any(not_flagged):
                    corrected[not_flagged, e] = (
                        (1.0 - alpha_normal) * corrected[not_flagged, e]
                        + alpha_normal * consensus[not_flagged, e]
                    )

            # Check convergence
            new_residuals = corrected - self._consensus(corrected)
            new_norm = float(np.sqrt(np.mean(new_residuals ** 2)))
            conv = max(0.0, prev_norm - new_norm)
            if verbose:
                convergence.append(float(new_norm))

            if conv < self.tol and iteration > 0:
                break
            prev_norm = new_norm

        # Final diagnostics
        final_consensus = self._consensus(corrected)
        final_residuals = corrected - final_consensus
        final_z, final_anomaly = self._anomaly_scores(corrected, final_consensus)

        # Per-sample confidence: 1 - mean(|residual| / |prediction|)
        pred_mag = np.maximum(np.abs(corrected), 1e-10)
        confidence = 1.0 - np.mean(np.abs(final_residuals) / pred_mag, axis=1)
        confidence = np.clip(confidence, 0.0, 1.0)

        return CorrectionReport(
            original=original,
            corrected=corrected,
            residuals=final_residuals,
            anomaly_mask=initial_anomaly_mask,
            initial_z_scores=initial_z_scores,
            confidence=confidence,
            iterations=iteration + 1,
            convergence=convergence if verbose else [],
        )

    # ------------------------------------------------------------------
    # Conformal prediction
    # ------------------------------------------------------------------

    def fit_conformal(
        self,
        calib_predictions: np.ndarray,
        calib_targets: np.ndarray,
    ) -> None:
        """Calibrate conformal prediction using holdout calibration set.

        Parameters
        ----------
        calib_predictions : (n_calib, 6) expert predictions
        calib_targets : (n_calib,) or (n_calib, 1) ground truth
        """
        calib_predictions = np.atleast_2d(calib_predictions)
        calib_targets = np.atleast_1d(calib_targets).reshape(-1, 1)

        # Nonconformity: distance of C6-consensus from true target
        report = self.correct(calib_predictions)
        ensemble_mean = np.mean(report.corrected, axis=1)
        self._calib_scores = np.abs(ensemble_mean - calib_targets.ravel())

    def predict_interval(
        self,
        predictions: np.ndarray,
        alpha: float = 0.1,
    ) -> ConformalSet:
        """Produce conformal prediction intervals for new samples.

        Parameters
        ----------
        predictions : (n_samples, 6) expert predictions
        alpha : float, significance level (90% confidence by default)

        Returns
        -------
        ConformalSet with lower/upper bounds.
        """
        if self._calib_scores is None:
            raise ValueError("Must call fit_conformal() first.")

        report = self.correct(predictions)
        ensemble_mean = np.mean(report.corrected, axis=1)

        n_calib = len(self._calib_scores)
        q_idx = int(np.ceil((1.0 - alpha) * (n_calib + 1)))
        q_idx = min(q_idx, n_calib - 1)
        t_hat = np.sort(self._calib_scores)[q_idx]

        return ConformalSet(
            lower=ensemble_mean - t_hat,
            upper=ensemble_mean + t_hat,
            alpha=alpha,
            calib_nonconformity=self._calib_scores.copy(),
        )

    def coverage_score(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute empirical coverage of conformal intervals."""
        ci = self.predict_interval(predictions)
        targets = np.atleast_1d(targets).ravel()
        covered = (targets >= ci.lower) & (targets <= ci.upper)
        return float(np.mean(covered))

    # ------------------------------------------------------------------
    # Dumpable
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TaiChiCorrector(threshold={self.threshold:.2f}, "
            f"max_iter={self.max_iter}, tol={self.tol:.0e}, "
            f"n_experts={self._n_experts})"
        )

    @property
    def is_calibrated(self) -> bool:
        return self._calib_scores is not None
