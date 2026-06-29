# Options Pricer & Greeks Analyzer

Pricing analytique et Monte Carlo d'options européennes Black-Scholes, avec calcul des Greeks et analyse complète des sensibilités.

---

## Objectif

Ce projet implémente de bout en bout la valorisation d'options européennes vanilles sous le modèle Black-Scholes : prix analytiques (call et put), Greeks du premier et second ordre, analyse des sensibilités aux paramètres de marché, et validation indépendante par simulation Monte Carlo. Il constitue une base de travail pour l'étude du delta-hedging, de la gestion du risque de portefeuille d'options et de la convergence des méthodes numériques.

---

## Méthodes

### Black-Scholes analytique

Sous la mesure risque-neutre, le prix d'un call européen est :

$$C = S \cdot N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S/K) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

La parité call-put $C - P = S - Ke^{-rT}$ est vérifiée analytiquement (erreur $< 10^{-9}$).

### Greeks analytiques

| Greek | Formule (call) | Interprétation |
|-------|---------------|----------------|
| $\Delta$ | $N(d_1)$ | Exposition directionnelle au sous-jacent |
| $\Gamma$ | $n(d_1) / (S\sigma\sqrt{T})$ | Convexité — vitesse de variation du delta |
| $\mathcal{V}$ | $S n(d_1)\sqrt{T} / 100$ | Sensibilité à la volatilité implicite (+1 pt) |
| $\Theta$ | $\left[-\frac{S n(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)\right] / 365$ | Érosion temporelle par jour |
| $\rho$ | $KTe^{-rT}N(d_2) / 100$ | Sensibilité au taux sans risque (+1 pt) |

### Simulation Monte Carlo

Chaque trajectoire suit le mouvement brownien géométrique (GBM) sous $\mathbb{Q}$ :

$$S_T = S_0 \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T} Z\right], \quad Z \sim \mathcal{N}(0,1)$$

Le prix est estimé par $\hat{C} = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]$. L'erreur standard décroît en $1/\sqrt{N}$ (théorème central limite), vérifiée empiriquement par régression log-log.

---

## Résultats clés

Paramètres de référence : $S = K = 100$, $T = 1$ an, $r = 5\%$, $\sigma = 20\%$ (option ATM).

### Prix et Greeks

| Grandeur | Call | Put |
|----------|-----:|----:|
| **Prix Black-Scholes** | **10.4506** | **5.5735** |
| Delta | +0.6368 | −0.3632 |
| Gamma | 0.01876 | 0.01876 |
| Vega (par +1% $\sigma$) | +0.3752 | +0.3752 |
| Theta (par jour) | −0.01757 | −0.00454 |
| Rho (par +1% $r$) | +0.5323 | −0.4189 |

### Validation Monte Carlo (N = 100 000 simulations)

| Grandeur | Valeur |
|----------|-------:|
| Prix MC | 10.4205 |
| Erreur standard | 0.0468 |
| IC 95 % | [10.329 ; 10.512] |
| Prix BS dans l'IC 95 % | **oui** |
| Erreur relative | **0.29 %** |
| Pente log-log mesurée | −0.500 (théorique : −0.5) |

---

## Visualisations

### 1. Prix call & put vs Spot

![Prix vs Spot](figures/01_price_vs_spot.png)

Le prix BS est toujours supérieur au payoff à maturité (valeur temps > 0). La valeur temps est maximale ATM et s'annule deep ITM/OTM. La convergence vers les droites de payoff pour $S \to 0$ et $S \to \infty$ confirme la cohérence du modèle.

### 2. Delta vs Spot

![Delta vs Spot](figures/02_delta_vs_spot.png)

Le delta call vaut 0.64 ATM (et non 0.5) en raison du drift $r$ dans $d_1$. Sa forme sigmoïde illustre la non-linéarité de l'exposition : un portefeuille delta-neutre doit être rééquilibré continuellement lorsque le spot évolue. Le delta put est le symétrique par parité ($\Delta_P = \Delta_C - 1$).

### 3. Gamma & Vega vs Spot

![Gamma et Vega vs Spot](figures/03_gamma_vega_vs_spot.png)

Gamma et Vega ont la même forme de cloche centrée ATM : ce sont les régions où la couverture est la plus coûteuse. Le centre de la cloche est légèrement décalé au-dessus du strike à cause de la distribution log-normale de $S_T$ (le mode est inférieur à la médiane). Un vendeur d'options est short Gamma et short Vega — il perd si la volatilité réalisée dépasse la volatilité implicite.

### 4. Theta vs Maturité

![Theta vs Maturité](figures/04_theta_vs_maturity.png)

L'accélération de l'érosion temporelle à l'approche de la maturité ($T \to 0$) est caractéristique du terme $1/(2\sqrt{T})$ dans la formule du Theta. Un acheteur d'option supporte ce coût quotidien en échange de la convexité (Gamma positif) : c'est le fondement de la relation $\text{PnL} \approx \frac{1}{2}\Gamma(\Delta S)^2 + \Theta \Delta t$.

### 5. Convergence Monte Carlo

![Convergence MC](figures/06_mc_convergence.png)

L'erreur absolue $|MC - BS|$ suit une droite de pente $-0.500$ en log-log, conforme à la décroissance théorique en $1/\sqrt{N}$. Avec 500 000 simulations, l'erreur tombe sous $0.005$ € ($< 0.05\%$). L'erreur standard suit le même régime, validant l'implémentation vectorisée NumPy.

---

## Structure du projet

```
options-pricer/
├── src/
│   ├── black_scholes.py     # Pricing BS analytique (call, put, parité)
│   ├── greeks.py            # Greeks analytiques (delta, gamma, vega, theta, rho)
│   └── monte_carlo.py       # Pricer Monte Carlo vectorisé (GBM, seed reproductible)
├── notebooks/
│   ├── 03_sensibilites.ipynb    # Analyse des sensibilités et surface 3D Plotly
│   └── 04_monte_carlo.ipynb     # Convergence MC vs BS en log-log
├── figures/
│   ├── 01_price_vs_spot.png
│   ├── 02_delta_vs_spot.png
│   ├── 03_gamma_vega_vs_spot.png
│   ├── 04_theta_vs_maturity.png
│   ├── 05_call_surface_3d.html  # Surface interactive Plotly
│   └── 06_mc_convergence.png
├── requirements.txt
└── README.md
```

---

## Installation & usage

```bash
git clone https://github.com/Octave-Horlin/options-pricer.git
cd options-pricer
python -m venv venv && source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Lancer les scripts autonomes :

```bash
python src/black_scholes.py   # Prix BS + vérification parité call-put
python src/greeks.py          # Tableau des Greeks (pandas)
python src/monte_carlo.py     # Validation MC vs BS (100 000 simulations)
```

Lancer les notebooks :

```bash
jupyter lab
# Ouvrir notebooks/03_sensibilites.ipynb ou notebooks/04_monte_carlo.ipynb
```

---

## Limites & extensions

**Limites du modèle actuel**

- Volatilité constante (pas de smile ni de skew de volatilité implicite)
- Options européennes uniquement (pas d'exercice anticipé)
- Taux sans risque déterministe et constant
- Absence de dividendes

**Extensions envisagées**

- Volatilité implicite et surface $\sigma(K, T)$ par inversion numérique de BS
- Options américaines : algorithme de Longstaff-Schwartz (Monte Carlo avec régression)
- Modèle de Heston : volatilité stochastique avec retour à la moyenne
- Modèle à sauts de Merton pour capturer les queues épaisses

---

## Stack technique

Python 3.11 — `numpy` · `scipy` · `pandas` · `matplotlib` · `seaborn` · `plotly` · `jupyter`

---

**Statut : ✅ Terminé**
