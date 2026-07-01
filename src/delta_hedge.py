"""
Delta-hedging simulation for a short European call.

Simulates the sale of a call hedged with daily delta rebalancing.
The spot follows GBM (sigma_realized); the hedge uses sigma_implied.
When both are equal the cumulative P&L should be close to zero.
"""

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from black_scholes import bs_call, _d1_d2          # noqa: F401 (_d1_d2 available for callers)
from greeks import delta as bs_delta, gamma as bs_gamma, theta as bs_theta, vega as bs_vega


def simulate_gbm_path(S0: float, T: float, r: float, sigma: float,
                      n_steps: int, seed: int = 42) -> np.ndarray:
    """
    Generate one GBM spot path.

    Parameters
    ----------
    S0      : initial spot price
    T       : horizon (years)
    r       : risk-free rate (continuous)
    sigma   : volatility
    n_steps : number of time steps
    seed    : RNG seed for reproducibility

    Returns
    -------
    prices : np.ndarray of shape (n_steps + 1,)
        S[0] = S0, subsequent values drawn from GBM.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal(n_steps)
    log_ret = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    prices = np.empty(n_steps + 1)
    prices[0] = S0
    prices[1:] = S0 * np.exp(np.cumsum(log_ret))
    return prices


def delta_hedge_simulation(
    S0: float, K: float, T: float, r: float,
    sigma_implied: float, sigma_realized: float,
    n_steps: int, seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """
    Simulate delta-neutral hedging of a short European call.

    At t=0 the call is sold at BS(sigma_implied); delta shares are bought
    immediately.  At each subsequent step the position is rebalanced to the
    new delta; cash earns/pays the risk-free rate.

    P&L attribution per step (Taylor expansion on short-call P&L):
      pnl_step ≈ −gamma_term − theta_term + vega_term + interest
      where
        gamma_term = ½ Γ (dS)²                           [cost of convexity]
        theta_term = Θ_annual · dt                        [negative → gain for short]
        vega_term  = ½ Γ S² (σ_r² − σ_i²) dt            [vol mis-pricing P&L]
        interest   = r · cash · dt

    Parameters
    ----------
    sigma_implied  : vol used for pricing and computing delta
    sigma_realized : vol driving the GBM path

    Returns
    -------
    df        : pd.DataFrame with one row per rebalancing step
    pnl_total : float — cumulative P&L at expiry
    """
    dt = T / n_steps
    prices = simulate_gbm_path(S0, T, r, sigma_realized, n_steps, seed)

    # ── t = 0 : sell call, immediately delta-hedge ────────────────────────────
    V0 = bs_call(S0, K, T, r, sigma_implied)
    d0 = bs_delta(S0, K, T, r, sigma_implied, 'call')

    position = d0            # shares held (long)
    cash = V0 - d0 * S0     # premium received minus cost of shares bought

    records = []
    pnl_cumul = 0.0

    for i in range(n_steps):
        S_cur = prices[i]
        S_nxt = prices[i + 1]
        dS = S_nxt - S_cur
        tau_cur = T - i * dt
        tau_nxt = T - (i + 1) * dt

        # Greeks at the current node (priced with sigma_implied)
        g   = bs_gamma(S_cur, K, tau_cur, r, sigma_implied)
        # th is per-calendar-day; multiply by 365 to get annual rate
        th  = bs_theta(S_cur, K, tau_cur, r, sigma_implied, 'call') * 365.0
        v   = bs_vega(S_cur,  K, tau_cur, r, sigma_implied)   # per 1% vol
        V_cur = bs_call(S_cur, K, tau_cur, r, sigma_implied)

        # Option value and new delta at next node
        if tau_nxt > 1e-10:
            V_nxt = bs_call(S_nxt, K, tau_nxt, r, sigma_implied)
            d_nxt = bs_delta(S_nxt, K, tau_nxt, r, sigma_implied, 'call')
        else:
            V_nxt = max(S_nxt - K, 0.0)
            d_nxt = 1.0 if S_nxt > K else 0.0

        # ── Step P&L (mark-to-market) ─────────────────────────────────────────
        interest   = r * cash * dt
        stock_pnl  = position * dS          # gain on long shares
        option_pnl = -(V_nxt - V_cur)       # gain on short call
        pnl_step   = stock_pnl + option_pnl + interest
        pnl_cumul += pnl_step

        # ── Attribution via Taylor expansion ──────────────────────────────────
        gamma_term = 0.5 * g * dS ** 2
        theta_term = th * dt                # annual theta × dt
        # vol P&L: difference between realized and implied variance, gamma-weighted
        vega_term  = 0.5 * g * S_cur ** 2 * (sigma_realized ** 2 - sigma_implied ** 2) * dt

        records.append({
            'step':       i,
            't':          i * dt,
            'S':          S_cur,
            'tau':        tau_cur,
            'V':          V_cur,
            'delta':      position,
            'gamma':      g,
            'theta':      th,        # stored as annual rate
            'vega':       v,
            'dS':         dS,
            'cash':       cash,
            'interest':   interest,
            'stock_pnl':  stock_pnl,
            'option_pnl': option_pnl,
            'pnl_step':   pnl_step,
            'pnl_cumul':  pnl_cumul,
            'gamma_term': gamma_term,
            'theta_term': theta_term,
            'vega_term':  vega_term,
        })

        # Rebalance: cash accrues interest, then adjust stock position
        cash = cash * (1.0 + r * dt) - (d_nxt - position) * S_nxt
        position = d_nxt

    return pd.DataFrame(records), pnl_cumul


if __name__ == '__main__':
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma_implied = sigma_realized = 0.20
    n_steps = 252
    seed = 42

    df, pnl = delta_hedge_simulation(
        S0, K, T, r, sigma_implied, sigma_realized, n_steps, seed,
    )

    print('=== Simulation delta-hedge : call ATM vendu ===')
    print(f'S0={S0}  K={K}  T={T}  r={r}'
          f'  sigma_impl={sigma_implied}  sigma_real={sigma_realized}  n={n_steps}')
    print()

    gamma_tot = -df['gamma_term'].sum()   # négatif → coût pour short gamma
    theta_tot = -df['theta_term'].sum()   # positif → gain theta (theta_annual < 0)
    vega_tot  =  df['vega_term'].sum()    # 0 ici car sigma_r = sigma_i
    int_tot   =  df['interest'].sum()
    reconst   = gamma_tot + theta_tot + vega_tot + int_tot
    residu    = pnl - reconst

    print(f'P&L total                        : {pnl:+.6f}')
    print()
    print('Attribution (décomposition Taylor) :')
    print(f'  −Σ gamma_term  (coût gamma)    : {gamma_tot:+.6f}')
    print(f'  −Σ theta_term  (gain theta)    : {theta_tot:+.6f}')
    print(f'   Σ vega_term   (vol mismatch)  : {vega_tot:+.6f}')
    print(f'   Σ interest    (cash r·dt)     : {int_tot:+.6f}')
    print(f'  {"─"*44}')
    print(f'  Total reconstitué              : {reconst:+.6f}')
    print(f'  Résidu Taylor / arrondi        : {residu:+.6f}')
