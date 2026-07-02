import numpy as np
from black_scholes import bs_call, bs_put
from monte_carlo import mc_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N_SIMS = 100_000
SEED = 42


def test_mc_call_within_bs_confidence_interval():
    mc, stderr = mc_price(S, K, T, r, sigma, N_SIMS, 'call', seed=SEED)
    bs = bs_call(S, K, T, r, sigma)
    assert abs(mc - bs) < 3 * stderr


def test_mc_put_within_bs_confidence_interval():
    mc, stderr = mc_price(S, K, T, r, sigma, N_SIMS, 'put', seed=SEED)
    bs = bs_put(S, K, T, r, sigma)
    assert abs(mc - bs) < 3 * stderr


def test_mc_reproducibility():
    p1, _ = mc_price(S, K, T, r, sigma, N_SIMS, 'call', seed=SEED)
    p2, _ = mc_price(S, K, T, r, sigma, N_SIMS, 'call', seed=SEED)
    assert p1 == p2


def test_mc_different_seeds_give_different_prices():
    p1, _ = mc_price(S, K, T, r, sigma, N_SIMS, 'call', seed=0)
    p2, _ = mc_price(S, K, T, r, sigma, N_SIMS, 'call', seed=1)
    assert p1 != p2
