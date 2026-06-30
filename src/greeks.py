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


def vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Sensibilité croisée ∂Delta/∂σ = ∂Vega/∂S, identique call et put.

    Vanna = -n(d1) * d2 / sigma

    Interprétation : variation du delta pour +1 unité de vol (non normalisée par 100).
    Sert à couvrir simultanément le risque directionnel et de volatilité.

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    """
    # -n(d1) * d2 / sigma
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -norm.pdf(d1) * d2 / sigma


def volga(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Convexité par rapport à la volatilité ∂Vega_brut/∂σ (aussi appelé vomma),
    identique call et put.

    Volga = S * n(d1) * sqrt(T) * d1 * d2 / sigma

    Dérivée du Vega BRUT (S*n(d1)*sqrt(T)) par rapport à sigma.
    Positif quand d1 et d2 ont le même signe (options hors de la monnaie).

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    """
    # S * n(d1) * sqrt(T) * d1 * d2 / sigma
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) * d1 * d2 / sigma


def charm(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str) -> float:
    """Variation du delta avec le passage du temps −∂Delta/∂T (delta bleed),
    en unités annuelles. Diviser par 365 pour obtenir la variation par jour
    calendaire (convention analogue au Theta).

    Formule (analytiquement identique call et put, car Delta_put = N(d1)−1
    et la dérivée de la constante −1 est nulle) :

      Charm = −n(d1) · (2rT − d2·σ√T) / (2T·σ√T)

    Convention de signe : valeur négative = delta décroît avec le temps
    (cas habituel d'un call ATM dont le delta revient vers 0.5 en fin de vie).

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    option_type       : "call" ou "put" (accepté pour cohérence d'API)
    """
    # −n(d1) * (2rT − d2*sigma*sqrt(T)) / (2T*sigma*sqrt(T))
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return -norm.pdf(d1) * (2 * r * T - d2 * sigma * np.sqrt(T)) / (
        2 * T * sigma * np.sqrt(T)
    )


def second_order_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict:
    """Retourne un dict avec les 3 Greeks de second ordre analytiques.

    Paramètres
    ----------
    S, K, T, r, sigma : paramètres Black-Scholes standards
    option_type       : "call" ou "put"
    """
    return {
        "Vanna": vanna(S, K, T, r, sigma),
        "Volga": volga(S, K, T, r, sigma),
        "Charm": charm(S, K, T, r, sigma, option_type),
    }


def all_greeks(S: float, K: float, T: float, r: float, sigma: float,
               option_type: str, include_second_order: bool = False) -> dict:
    """Retourne un dict avec les Greeks analytiques pour une option européenne.

    Paramètres
    ----------
    S, K, T, r, sigma    : paramètres Black-Scholes standards
    option_type          : "call" ou "put"
    include_second_order : si True, ajoute Vanna, Volga, Charm
    """
    g = {
        "Delta": delta(S, K, T, r, sigma, option_type),
        "Gamma": gamma(S, K, T, r, sigma),
        "Vega":  vega(S, K, T, r, sigma),
        "Theta": theta(S, K, T, r, sigma, option_type),
        "Rho":   rho(S, K, T, r, sigma, option_type),
    }
    if include_second_order:
        g.update(second_order_greeks(S, K, T, r, sigma, option_type))
    return g


if __name__ == "__main__":
    import pandas as pd

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    # ── Greeks du premier ordre ───────────────────────────────────────────────
    data = {
        "Call": all_greeks(S, K, T, r, sigma, "call"),
        "Put":  all_greeks(S, K, T, r, sigma, "put"),
    }
    df = pd.DataFrame(data).T
    df.index.name = "Option"
    print(f"Greeks du 1er ordre  (S={S}, K={K}, T={T}, r={r}, sigma={sigma})\n")
    print(df.to_string(float_format=lambda x: f"{x:+.6f}"))

    # ── Validation par différences finies — Greeks du 2nd ordre ───────────────
    print("\n" + "=" * 65)
    print("Validation différences finies — Greeks du 2nd ordre (call ATM)")
    print("=" * 65)

    h = 1e-4

    # Vanna ≈ [delta(sigma+h) - delta(sigma-h)] / (2h)
    vanna_an = vanna(S, K, T, r, sigma)
    vanna_fd = (
        delta(S, K, T, r, sigma + h, "call") -
        delta(S, K, T, r, sigma - h, "call")
    ) / (2 * h)

    # Volga ≈ [vega_brut(sigma+h) - vega_brut(sigma-h)] / (2h)
    # vega() retourne Vega/100, on remultiplie pour obtenir le Vega brut
    def _vega_brut(s: float) -> float:
        return vega(S, K, T, r, s) * 100.0

    volga_an = volga(S, K, T, r, sigma)
    volga_fd = (_vega_brut(sigma + h) - _vega_brut(sigma - h)) / (2 * h)

    # Charm ≈ -[delta(T+h) - delta(T-h)] / (2h)
    # Signe négatif : Charm = −∂Delta/∂T (convention temps décroissant)
    charm_an = charm(S, K, T, r, sigma, "call")
    charm_fd = -(
        delta(S, K, T + h, r, sigma, "call") -
        delta(S, K, T - h, r, sigma, "call")
    ) / (2 * h)

    rows = [
        ("Vanna", vanna_an, vanna_fd),
        ("Volga", volga_an, volga_fd),
        ("Charm", charm_an, charm_fd),
    ]

    fmt_hdr = f"{'Greek':<8}  {'Analytique':>14}  {'Diff. finie':>14}  {'Écart':>12}  {'< 1e-4'}"
    print(fmt_hdr)
    print("-" * 65)
    for name, an, fd in rows:
        err = abs(an - fd)
        ok  = "✓" if err < 1e-4 else "✗"
        print(f"{name:<8}  {an:>+14.8f}  {fd:>+14.8f}  {err:>12.2e}  {ok}")

    print("\nNote : Volga = ∂Vega_brut/∂σ (non divisé par 100)")
    print("       Charm en unités annuelles ; diviser par 365 → variation/jour")
