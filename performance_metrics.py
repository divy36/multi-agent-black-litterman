"""Shared, deterministic performance statistics."""

import numpy as np


def bootstrap_sharpe_ci(returns, n_boot=10000, seed=42, risk_free_annual=0.0, periods_per_year=12):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 3 or n_boot < 1:
        return float('nan'), float('nan'), float('nan')

    r = r - ((1 + risk_free_annual) ** (1 / periods_per_year) - 1)

    def sharpe(values):
        sd = values.std(ddof=1)
        tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(values))))
        return float('nan') if sd <= tolerance else float(values.mean() / sd * np.sqrt(periods_per_year))

    point = sharpe(r)
    if not np.isfinite(point):
        return point, float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    estimates = np.array([sharpe(r[rng.integers(0, len(r), len(r))]) for _ in range(n_boot)])
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return point, float('nan'), float('nan')
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return point, float(lo), float(hi)
