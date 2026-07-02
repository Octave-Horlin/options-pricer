import numpy as np
import pytest
from black_scholes import bs_call, bs_put

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


def test_call_atm_price():
    assert abs(bs_call(S, K, T, r, sigma) - 10.4506) < 1e-4


def test_put_atm_price():
    assert abs(bs_put(S, K, T, r, sigma) - 5.5735) < 1e-4


def test_put_call_parity():
    C = bs_call(S, K, T, r, sigma)
    P = bs_put(S, K, T, r, sigma)
    assert abs((C - P) - (S - K * np.exp(-r * T))) < 1e-9


@pytest.mark.parametrize("S_,K_,T_,r_,sigma_", [
    (100.0, 100.0, 1.0, 0.05, 0.20),
    (110.0, 100.0, 0.5, 0.03, 0.15),
    (90.0,  100.0, 2.0, 0.01, 0.30),
    (100.0, 120.0, 1.0, 0.05, 0.25),
    (80.0,   80.0, 0.25, 0.00, 0.40),
])
def test_call_bounds(S_, K_, T_, r_, sigma_):
    price = bs_call(S_, K_, T_, r_, sigma_)
    lower = max(0.0, S_ - K_ * np.exp(-r_ * T_))
    upper = S_
    assert price >= lower - 1e-10
    assert price <= upper + 1e-10
