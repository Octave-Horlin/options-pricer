import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from black_scholes import bs_call, bs_put


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int,
    option_type: str,
    seed: int = 42,
) -> tuple[float, float]:
    """Prix d'une option européenne par simulation Monte Carlo (GBM vectorisé).

    Simule n_sims trajectoires du sous-jacent sous la mesure risque-neutre :
        S_T = S * exp((r - sigma²/2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)

    Paramètres
    ----------
    S           : prix spot
    K           : strike
    T           : maturité en années
    r           : taux sans risque continu
    sigma       : volatilité annualisée
    n_sims      : nombre de simulations
    option_type : "call" ou "put"
    seed        : graine pour la reproductibilité (np.random.default_rng)

    Retourne
    --------
    (prix, erreur_standard)
        prix            : moyenne des payoffs actualisés
        erreur_standard : std(payoffs actualisés) / sqrt(n_sims)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)

    # Simulation vectorisée — une seule opération NumPy, pas de boucle Python
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoffs
    prix = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(n_sims)
    return prix, stderr


if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    n_sims = 100_000

    prix_mc, stderr = mc_price(S, K, T, r, sigma, n_sims, "call")
    prix_bs = bs_call(S, K, T, r, sigma)
    erreur_rel = abs(prix_mc - prix_bs) / prix_bs * 100

    print(f"Parametres")
    print(f"  S={S}, K={K}, T={T}, r={r}, sigma={sigma}, n_sims={n_sims:,}")
    print()
    print(f"{'Prix MC     ':20}: {prix_mc:.6f}")
    print(f"{'Erreur std  ':20}: {stderr:.6f}  (IC 95% ~+/-{1.96*stderr:.4f})")
    print(f"{'Prix BS ref ':20}: {prix_bs:.6f}")
    print(f"{'Erreur rel. ':20}: {erreur_rel:.4f} %")
