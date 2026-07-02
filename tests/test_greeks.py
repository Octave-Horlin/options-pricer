import numpy as np
import pytest
from black_scholes import bs_call, bs_put
from greeks import delta, gamma, vega, theta, rho, vanna, volga, charm

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
TOL_REF = 1e-3
TOL_FD  = 1e-4


# ── Reference values ──────────────────────────────────────────────────────────

def test_delta_call_reference():
    assert abs(delta(S, K, T, r, sigma, 'call') - 0.6368) < TOL_REF


def test_gamma_reference():
    assert abs(gamma(S, K, T, r, sigma) - 0.0188) < TOL_REF


def test_vega_reference():
    assert abs(vega(S, K, T, r, sigma) - 0.3752) < TOL_REF


def test_theta_call_reference():
    assert abs(theta(S, K, T, r, sigma, 'call') - (-0.0176)) < TOL_REF


def test_rho_call_reference():
    assert abs(rho(S, K, T, r, sigma, 'call') - 0.5323) < TOL_REF


def test_delta_call_minus_put_equals_one():
    dc = delta(S, K, T, r, sigma, 'call')
    dp = delta(S, K, T, r, sigma, 'put')
    assert abs(dc - dp - 1.0) < 1e-12


# ── Finite-difference validation — 1st-order Greeks ──────────────────────────

def test_fd_delta():
    h = 0.01
    fd = (bs_call(S + h, K, T, r, sigma) - bs_call(S - h, K, T, r, sigma)) / (2 * h)
    assert abs(delta(S, K, T, r, sigma, 'call') - fd) < TOL_FD


def test_fd_gamma():
    h = 0.01
    fd = (
        bs_call(S + h, K, T, r, sigma)
        - 2 * bs_call(S, K, T, r, sigma)
        + bs_call(S - h, K, T, r, sigma)
    ) / h ** 2
    assert abs(gamma(S, K, T, r, sigma) - fd) < TOL_FD


def test_fd_vega():
    h = 1e-4
    # vega returns vega_brut / 100 ; fd on price / h gives vega_brut per unit sigma
    fd = (bs_call(S, K, T, r, sigma + h) - bs_call(S, K, T, r, sigma - h)) / (2 * h * 100)
    assert abs(vega(S, K, T, r, sigma) - fd) < TOL_FD


def test_fd_theta():
    h = 1e-4
    # theta() = dC/dt / 365  with dC/dt = -dC/dT
    fd_dT = (bs_call(S, K, T + h, r, sigma) - bs_call(S, K, T - h, r, sigma)) / (2 * h)
    assert abs(theta(S, K, T, r, sigma, 'call') * 365 - (-fd_dT)) < TOL_FD


def test_fd_rho():
    h = 1e-4
    # rho() = dC/dr / 100
    fd = (bs_call(S, K, T, r + h, sigma) - bs_call(S, K, T, r - h, sigma)) / (2 * h * 100)
    assert abs(rho(S, K, T, r, sigma, 'call') - fd) < TOL_FD


# ── Finite-difference validation — 2nd-order Greeks ──────────────────────────

def test_fd_vanna():
    h = 1e-4
    fd = (
        delta(S, K, T, r, sigma + h, 'call') - delta(S, K, T, r, sigma - h, 'call')
    ) / (2 * h)
    assert abs(vanna(S, K, T, r, sigma) - fd) < TOL_FD


def test_fd_volga():
    h = 1e-4
    vega_brut = lambda s: vega(S, K, T, r, s) * 100.0
    fd = (vega_brut(sigma + h) - vega_brut(sigma - h)) / (2 * h)
    assert abs(volga(S, K, T, r, sigma) - fd) < TOL_FD


def test_fd_charm():
    h = 1e-4
    fd = -(delta(S, K, T + h, r, sigma, 'call') - delta(S, K, T - h, r, sigma, 'call')) / (2 * h)
    assert abs(charm(S, K, T, r, sigma, 'call') - fd) < TOL_FD
