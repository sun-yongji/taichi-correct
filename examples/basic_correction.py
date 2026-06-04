"""TaiChi-Correct demonstration: C6 error correction on 6-expert predictions."""

import numpy as np

from taichi_correct import TaiChiCorrector, c6_residual_decomposition
from taichi_correct.error_correction import eigenmode_magnitude


def main():
    rng = np.random.default_rng(42)
    print("=" * 70)
    print("  TaiChi-Correct: C6 Symmetry Error Correction Engine")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Generate synthetic 6-expert predictions
    # ------------------------------------------------------------------
    n_samples = 500
    base = 5.0 + rng.normal(0, 0.3, (n_samples, 1))
    clean = base + rng.normal(0, 0.05, (n_samples, 6))

    # Inject errors: experts 2 and 4 are noisy
    noisy = clean.copy()
    noisy[:, 2] += rng.normal(0, 1.5, n_samples)   # Expert 3 (离爻) noisy
    noisy[:, 4] += rng.normal(0, 0.8, n_samples)   # Expert 5 (坎爻) moderately noisy

    print(f"\nGenerated {n_samples} samples, 6 experts each")
    print(f"Std per expert (noisy):  {np.std(noisy, axis=0).round(3)}")
    print(f"Std per expert (clean):  {np.std(clean, axis=0).round(3)}")

    # ------------------------------------------------------------------
    # 2. Apply C6 correction
    # ------------------------------------------------------------------
    corrector = TaiChiCorrector(
        threshold=2.5,
        max_iter=10,
        tol=1e-5,
    )

    report = corrector.correct(noisy, verbose=True)

    print(f"\n--- Correction Report ---")
    print(f"Iterations:      {report.iterations}")
    print(f"Mean confidence: {report.confidence.mean():.4f}")
    print(f"Min confidence:  {report.confidence.min():.4f}")

    # Anomaly detection rates
    flagged = np.mean(report.anomaly_mask, axis=0)
    print(f"\nAnomaly flag rate per expert:")
    labels = ["E1-unif", "E2-dipI", "E3-dipII", "E4-quadI", "E5-quadII", "E6-hexa"]
    for i in range(6):
        bar = "█" * int(flagged[i] * 40)
        print(f"  {labels[i]:<10s} {flagged[i]:.3f}  {bar}")
    print(f"  {'avg':<10s} {flagged.mean():.3f}")

    # Residual improvement
    orig_res = np.std(noisy - corrector._consensus(noisy))
    corr_res = np.std(report.residuals)
    print(f"\nResidual std:  {orig_res:.4f} → {corr_res:.4f}  "
          f"({(1 - corr_res / orig_res) * 100:.1f}% reduction)")

    # ------------------------------------------------------------------
    # 3. C6 eigenmode decomposition of residuals
    # ------------------------------------------------------------------
    print(f"\n--- C6 Eigenmode Decomposition ---")
    mags = eigenmode_magnitude(report.residuals)
    mode_names = [
        "hexapole_noise ",
        "quadrupole_II  ",
        "quadrupole_I   ",
        "dipole_II      ",
        "dipole_I       ",
        "uniform_bias   ",
    ]
    for name, mag in zip(mode_names, mags):
        bar = "▓" * int(mag / max(mags) * 40)
        print(f"  {name} (λ)  {mag:.4f}  {bar}")

    # ------------------------------------------------------------------
    # 4. Conformal prediction demo
    # ------------------------------------------------------------------
    print(f"\n--- Conformal Prediction (α=0.1, 90% confidence) ---")
    calib_x = clean[:200]
    calib_y = base[:200].ravel()
    test_x = noisy[200:]
    test_y = base[200:].ravel()

    corrector.fit_conformal(calib_x, calib_y)
    ci = corrector.predict_interval(test_x, alpha=0.1)
    coverage = corrector.coverage_score(test_x, test_y)

    print(f"Interval width (avg):  {np.mean(ci.upper - ci.lower):.4f}")
    print(f"Empirical coverage:    {coverage:.3f}")
    print(f"Expected coverage:     {1 - ci.alpha:.3f}")

    print("\n" + "=" * 70)
    print("  M5 TaiChi-Correct demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
