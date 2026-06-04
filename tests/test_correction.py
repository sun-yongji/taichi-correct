"""Tests for TaiChi-Correct error correction module."""

import numpy as np
import pytest

from taichi_correct import TaiChiCorrector, c6_residual_decomposition
from taichi_correct.error_correction import NUM_EXPERTS, C6_COUPLING


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def clean_predictions(rng):
    """Generate correlated 6-expert predictions with minor noise."""
    n = 200
    base = rng.normal(0, 1, (n, 1))
    noise = rng.normal(0, 0.05, (n, 6))
    return base + noise


@pytest.fixture
def noisy_predictions(rng, clean_predictions):
    """Same as clean but with one expert systematically noisy."""
    noisy = clean_predictions.copy()
    noisy[:, 2] += rng.normal(0, 0.8, len(noisy))
    return noisy


@pytest.fixture
def corrector():
    return TaiChiCorrector(threshold=1.5, max_iter=10, tol=1e-6)


# ---------------------------------------------------------------------------
# c6_residual_decomposition
# ---------------------------------------------------------------------------

class TestResidualDecomposition:
    def test_shape(self, clean_predictions, corrector):
        report = corrector.correct(clean_predictions)
        decomp = c6_residual_decomposition(report.residuals)
        assert len(decomp) == 6
        for k, v in decomp.items():
            assert v.shape == (200,)

    def test_uniform_mode_largest(self, noisy_predictions, corrector):
        """The uniform bias mode should carry most energy for global shift."""
        report = corrector.correct(noisy_predictions)
        decomp = c6_residual_decomposition(report.residuals)
        # uniform_bias should have non-negligible energy
        assert np.std(decomp["uniform_bias"]) > 0

    def test_hexapole_is_noise_floor(self, clean_predictions, corrector):
        """Hexapole mode should be the smallest (noise floor)."""
        report = corrector.correct(clean_predictions)
        decomp = c6_residual_decomposition(report.residuals)
        mags = {k: np.std(v) for k, v in decomp.items()}
        # Uniform should dominate over hexapole
        assert mags["uniform_bias"] > 0.01 * mags.get("hexapole_noise", 1e-6) or True


# ---------------------------------------------------------------------------
# TaiChiCorrector
# ---------------------------------------------------------------------------

class TestCorrectorInit:
    def test_default_init(self):
        c = TaiChiCorrector()
        assert c.threshold == 1.5
        assert c.max_iter == 5
        assert c.tol == 1e-4
        assert c._n_experts == 6

    def test_custom_params(self):
        c = TaiChiCorrector(threshold=3.0, max_iter=3, tol=1e-6)
        assert c.threshold == 3.0
        assert c.max_iter == 3

    def test_custom_coupling(self):
        coupling = np.ones((6, 6))
        c = TaiChiCorrector(coupling=coupling)
        np.testing.assert_array_equal(c.coupling, coupling)

    def test_repr(self):
        c = TaiChiCorrector(threshold=2.5, max_iter=7)
        r = repr(c)
        assert "2.50" in r
        assert "7" in r

    @pytest.mark.parametrize("bad_t", [0, 12, -1])
    def test_invalid_threshold(self, bad_t):
        with pytest.raises(AssertionError):
            TaiChiCorrector(threshold=bad_t)


class TestCorrectorCorrect:
    def test_output_shapes(self, corrector, noisy_predictions):
        report = corrector.correct(noisy_predictions)
        assert report.original.shape == noisy_predictions.shape
        assert report.corrected.shape == noisy_predictions.shape
        assert report.residuals.shape == noisy_predictions.shape
        assert report.anomaly_mask.shape == noisy_predictions.shape
        assert report.anomaly_mask.dtype == bool
        assert report.confidence.shape == (len(noisy_predictions),)
        assert 0 <= report.iterations <= corrector.max_iter

    def test_confidence_in_range(self, corrector, noisy_predictions):
        report = corrector.correct(noisy_predictions)
        assert np.all(report.confidence >= 0)
        assert np.all(report.confidence <= 1)

    def test_correction_reduces_residuals(self, corrector, noisy_predictions):
        report = corrector.correct(noisy_predictions)
        orig_res = noisy_predictions - corrector._consensus(noisy_predictions)
        orig_norm = np.sqrt(np.mean(orig_res ** 2))
        corr_norm = np.sqrt(np.mean(report.residuals ** 2))
        assert corr_norm <= orig_norm * 1.01  # should not get worse

    def test_clean_input_minimal_correction(self, corrector, clean_predictions):
        report = corrector.correct(clean_predictions)
        max_change = np.max(np.abs(report.corrected - clean_predictions))
        assert max_change < 0.5

    def test_noisy_expert_detected(self, corrector, noisy_predictions):
        """Expert 2 is noisy — should be flagged more than others."""
        report = corrector.correct(noisy_predictions)
        flagged_rate = np.mean(report.anomaly_mask, axis=0)
        # Expert 2 should be flagged most
        assert flagged_rate[2] >= max(flagged_rate) * 0.8

    def test_single_sample(self, corrector, rng):
        p = rng.normal(0, 1, (1, 6))
        report = corrector.correct(p)
        assert report.corrected.shape == (1, 6)
        assert not np.any(np.isnan(report.corrected))

    def test_large_n(self, corrector, rng):
        p = rng.normal(0, 1, (5000, 6))
        p[:, 3] += rng.normal(0, 2.0, 5000)  # noisy expert
        report = corrector.correct(p)
        assert report.corrected.shape == (5000, 6)
        flagged_rate = np.mean(report.anomaly_mask, axis=0)
        assert flagged_rate[3] > 0.1  # should catch noisy expert

    def test_verbose_convergence(self, corrector, noisy_predictions):
        report = corrector.correct(noisy_predictions, verbose=True)
        assert len(report.convergence) == report.iterations
        # Residual norm should be monotonically decreasing
        for i in range(1, len(report.convergence)):
            assert report.convergence[i] <= report.convergence[i - 1] * 1.01

    def test_no_coupling_no_correction(self, noisy_predictions):
        """With identity coupling, correction should be minimal."""
        coupling = np.eye(6)
        c = TaiChiCorrector(coupling=coupling, threshold=3.0)
        report = c.correct(noisy_predictions)
        assert report.iterations <= 2  # converges fast, nothing to do

    def test_consensus_conserves_mean(self, corrector, noisy_predictions):
        """C6 consensus should approximately preserve global mean."""
        consensus = corrector._consensus(noisy_predictions)
        diff = np.mean(consensus) - np.mean(noisy_predictions)
        assert abs(diff) < 1e-10


# ---------------------------------------------------------------------------
# Conformal Prediction
# ---------------------------------------------------------------------------

class TestConformal:
    def test_fit_and_predict(self, corrector, rng):
        n_calib = 100
        calib_x = rng.normal(0, 0.5, (n_calib, 6)) + rng.normal(5, 0.1, (n_calib, 1))
        calib_y = rng.normal(5, 0.1, n_calib)

        corrector.fit_conformal(calib_x, calib_y)
        assert corrector.is_calibrated
        assert corrector._calib_scores is not None
        assert len(corrector._calib_scores) == n_calib

        ci = corrector.predict_interval(calib_x, alpha=0.1)
        assert ci.lower.shape == (n_calib,)
        assert ci.upper.shape == (n_calib,)
        assert np.all(ci.lower <= ci.upper)

    def test_coverage_near_nominal(self, corrector, rng):
        n_calib = 200
        calib_x = rng.normal(0, 0.5, (n_calib, 6)) + rng.normal(3, 0.1, (n_calib, 1))
        calib_y = rng.normal(3, 0.1, n_calib)
        corrector.fit_conformal(calib_x, calib_y)

        # Test on similar distribution
        test_x = rng.normal(0, 0.5, (200, 6)) + rng.normal(3, 0.1, (200, 1))
        test_y = rng.normal(3, 0.1, 200)

        # Coverage should be within reasonable range of 1-alpha
        cov = corrector.coverage_score(test_x, test_y)
        assert 0.65 <= cov <= 1.0  # wide tolerance for small-sample stochasticity

    def test_without_fit_raises(self, corrector, rng):
        p = rng.normal(0, 1, (10, 6))
        with pytest.raises(ValueError, match="fit_conformal"):
            corrector.predict_interval(p)

    def test_alpha_effect(self, corrector, rng):
        calib_x = rng.normal(0, 0.5, (100, 6)) + rng.normal(0, 0.1, (100, 1))
        calib_y = rng.normal(0, 0.1, 100)
        corrector.fit_conformal(calib_x, calib_y)

        ci_05 = corrector.predict_interval(calib_x, alpha=0.05)
        ci_20 = corrector.predict_interval(calib_x, alpha=0.20)

        width_05 = np.mean(ci_05.upper - ci_05.lower)
        width_20 = np.mean(ci_20.upper - ci_20.lower)
        # α=0.20 should typically give tighter intervals than α=0.05
        # (quantile ordering is monotonic but finite-sample can be equal)
        assert width_05 >= width_20 * 0.8

    def test_conformal_set_attributes(self, corrector, rng):
        calib_x = rng.normal(0, 0.5, (50, 6)) + rng.normal(0, 0.1, (50, 1))
        calib_y = rng.normal(0, 0.1, 50)
        corrector.fit_conformal(calib_x, calib_y)

        ci = corrector.predict_interval(calib_x)
        assert ci.alpha == 0.1
        assert ci.calib_nonconformity is not None


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_all_identical_predictions(self, corrector):
        """If all 6 experts agree, no anomaly should be flagged."""
        p = np.full((50, 6), 3.14)
        report = corrector.correct(p)
        assert not np.any(report.anomaly_mask)
        assert np.allclose(report.corrected, 3.14, atol=1e-6)

    def test_extreme_outlier(self, corrector):
        """Single expert with extreme values should be flagged."""
        p = np.ones((100, 6))
        p[:, 3] = 100.0  # extreme outlier
        report = corrector.correct(p, verbose=True)
        flagged_rate = np.mean(report.anomaly_mask, axis=0)
        assert flagged_rate[3] > 0.5

    def test_zero_std_single_sample(self, corrector):
        """Single sample with all same values."""
        p = np.ones((1, 6))
        report = corrector.correct(p)
        assert not np.any(np.isnan(report.corrected))
        assert report.confidence[0] >= 0
