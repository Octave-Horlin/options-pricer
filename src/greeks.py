import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from black_scholes import _d1_d2


def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Sensibilité du prix au prix spot (dV/dS).

    Call : N(d1)
    Put  : N(d1) - 1  =  -N(-d1)

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    option_type       : "call" ou "put"
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Sensibilité du delta au prix spot (d²V/dS²), identique call et put.

    Gamma = n(d1) / (S * sigma * sqrt(T))

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    # n(d1) : densité de la loi normale standard évaluée en d1
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Sensibilité du prix à la volatilité (dV/dsigma), par +1 point de vol.

    Vega_brut = S * n(d1) * sqrt(T)
    Retourne Vega_brut / 100  (valeur pour +1 % de vol)

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Sensibilité du prix au passage du temps (dV/dT), par jour calendaire.

    Call : -[S*n(d1)*sigma/(2*sqrt(T))] - r*K*exp(-rT)*N(d2)
    Put  : -[S*n(d1)*sigma/(2*sqrt(T))] + r*K*exp(-rT)*N(-d2)
    Retourne Theta_annuel / 365

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    option_type       : "call" ou "put"
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    decay = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if option_type == "call":
        return (decay - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    return (decay + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0


def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Sensibilité du prix au taux sans risque (dV/dr), par +1 % de taux.

    Call : K*T*exp(-rT)*N(d2)
    Put  : -K*T*exp(-rT)*N(-d2)
    Retourne Rho_brut / 100  (valeur pour +1 % de taux)

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    option_type       : "call" ou "put"
    """
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def all_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict:
    """Retourne un dict avec les 5 Greeks analytiques pour une option européenne.

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    option_type       : "call" ou "put"
    """
    return {
        "Delta": delta(S, K, T, r, sigma, option_type),
        "Gamma": gamma(S, K, T, r, sigma),
        "Vega":  vega(S, K, T, r, sigma),
        "Theta": theta(S, K, T, r, sigma, option_type),
        "Rho":   rho(S, K, T, r, sigma, option_type),
    }


if __name__ == "__main__":
    import pandas as pd

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    data = {
        "Call": all_greeks(S, K, T, r, sigma, "call"),
        "Put":  all_greeks(S, K, T, r, sigma, "put"),
    }

    df = pd.DataFrame(data).T
    df.index.name = "Option"
    print(f"Greeks ATM  (S={S}, K={K}, T={T}, r={r}, sigma={sigma})\n")
    print(df.to_string(float_format=lambda x: f"{x:+.6f}"))
