import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from black_scholes import bs_call, bs_put, _d1_d2


def implied_vol(
    price_market: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Volatilité implicite d'une option européenne par inversion de Black-Scholes.

    Utilise Newton-Raphson en premier recours, puis brentq si NR échoue.
    Retourne np.nan si le prix de marché est hors des bornes d'arbitrage.

    Paramètres
    ----------
    price_market : prix observé sur le marché
    S            : prix spot
    K            : strike
    T            : maturité en années
    r            : taux sans risque continu
    option_type  : 'call' ou 'put'
    tol          : tolérance de convergence sur sigma
    max_iter     : nombre maximum d'itérations Newton-Raphson
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type doit être 'call' ou 'put'")

    disc = np.exp(-r * T)

    # Bornes d'arbitrage
    if option_type == "call":
        lower_bound = max(0.0, S - K * disc)
        upper_bound = S
    else:
        lower_bound = max(0.0, K * disc - S)
        upper_bound = K * disc

    if price_market <= lower_bound or price_market >= upper_bound:
        return np.nan

    bs_price = bs_call if option_type == "call" else bs_put

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma) - price_market

    def vega_brut(sigma: float) -> float:
        d1, _ = _d1_d2(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)

    # Newton-Raphson
    sigma = 0.2
    for _ in range(max_iter):
        v = vega_brut(sigma)
        if v < 1e-10:
            break
        step = objective(sigma) / v
        sigma -= step
        if abs(step) < tol:
            return sigma

    # Repli sur brentq
    try:
        return brentq(objective, 1e-6, 5.0, xtol=tol, maxiter=1000)
    except ValueError:
        return np.nan


if __name__ == "__main__":
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma_true = 0.20

    # --- Call ---
    call_price = bs_call(S, K, T, r, sigma_true)
    sigma_call = implied_vol(call_price, S, K, T, r, "call")
    print("=== Call round-trip ===")
    print(f"  Prix théorique  : {call_price:.8f}")
    print(f"  sigma vrai      : {sigma_true:.8f}")
    print(f"  sigma implicite : {sigma_call:.8f}")
    print(f"  Écart           : {abs(sigma_call - sigma_true):.2e}")
    print(f"  Convergé < 1e-6 : {abs(sigma_call - sigma_true) < 1e-6}")

    print()

    # --- Put ---
    put_price = bs_put(S, K, T, r, sigma_true)
    sigma_put = implied_vol(put_price, S, K, T, r, "put")
    print("=== Put round-trip ===")
    print(f"  Prix théorique  : {put_price:.8f}")
    print(f"  sigma vrai      : {sigma_true:.8f}")
    print(f"  sigma implicite : {sigma_put:.8f}")
    print(f"  Écart           : {abs(sigma_put - sigma_true):.2e}")
    print(f"  Convergé < 1e-6 : {abs(sigma_put - sigma_true) < 1e-6}")
