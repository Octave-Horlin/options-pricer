import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Calcule d1 et d2 de la formule Black-Scholes.

    d1 = [ln(S/K) + (r + sigma²/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Paramètres
    ----------
    S     : prix spot du sous-jacent
    K     : prix d'exercice (strike)
    T     : maturité en années
    r     : taux sans risque (continu, annualisé)
    sigma : volatilité implicite annualisée
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Prix d'un call européen selon Black-Scholes.

    C = S * N(d1) - K * exp(-rT) * N(d2)

    Paramètres
    ----------
    S     : prix spot du sous-jacent
    K     : prix d'exercice (strike)
    T     : maturité en années
    r     : taux sans risque (continu, annualisé)
    sigma : volatilité implicite annualisée

    Retourne
    --------
    Prix théorique du call européen.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Prix d'un put européen selon Black-Scholes.

    P = K * exp(-rT) * N(-d2) - S * N(-d1)

    Paramètres
    ----------
    S     : prix spot du sous-jacent
    K     : prix d'exercice (strike)
    T     : maturité en années
    r     : taux sans risque (continu, annualisé)
    sigma : volatilité implicite annualisée

    Retourne
    --------
    Prix théorique du put européen.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    call = bs_call(S, K, T, r, sigma)
    put  = bs_put(S, K, T, r, sigma)

    print(f"Call ATM : {call:.6f}")
    print(f"Put  ATM : {put:.6f}")

    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    print(f"\nParité call-put")
    print(f"  C - P              = {lhs:.9f}")
    print(f"  S - K*exp(-rT)     = {rhs:.9f}")
    print(f"  Égaux (tol 1e-9)   : {abs(lhs - rhs) < 1e-9}")
