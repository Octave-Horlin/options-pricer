import numpy as np
from delta_hedge import simulate_gbm_path, delta_hedge_simulation

S0, K, T, r = 100.0, 100.0, 1.0, 0.05
sigma = 0.20
N_STEPS = 252
N_PATHS = 200


def test_gbm_reproducibility():
    p1 = simulate_gbm_path(S0, T, r, sigma, N_STEPS, seed=42)
    p2 = simulate_gbm_path(S0, T, r, sigma, N_STEPS, seed=42)
    np.testing.assert_array_equal(p1, p2)


def test_gbm_different_seeds_differ():
    p1 = simulate_gbm_path(S0, T, r, sigma, N_STEPS, seed=0)
    p2 = simulate_gbm_path(S0, T, r, sigma, N_STEPS, seed=1)
    assert not np.array_equal(p1, p2)


def test_gbm_path_length():
    path = simulate_gbm_path(S0, T, r, sigma, N_STEPS, seed=0)
    assert len(path) == N_STEPS + 1
    assert path[0] == S0


def test_hedge_pnl_mean_near_zero_when_vols_equal():
    pnls = np.array([
        delta_hedge_simulation(S0, K, T, r, sigma, sigma, N_STEPS, seed=s)[1]
        for s in range(N_PATHS)
    ])
    mean = pnls.mean()
    sem  = pnls.std(ddof=1) / np.sqrt(N_PATHS)
    assert abs(mean) < 3 * sem, (
        f"|mean P&L| = {abs(mean):.4f} >= 3*sem = {3*sem:.4f} "
        f"(mean={mean:.4f}, std={pnls.std():.4f})"
    )
