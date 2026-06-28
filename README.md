# Options Pricer & Greeks Analyzer

Outil de pricing d'options européennes avec le modèle Black-Scholes, calcul des Greeks, analyse de sensibilité et validation par simulation Monte Carlo.

## Description

Ce projet implémente :
- **Pricing Black-Scholes** : valorisation d'options call et put européennes
- **Greeks** : Delta, Gamma, Vega, Theta, Rho
- **Analyse de sensibilité** : surface de volatilité implicite, heatmaps
- **Validation Monte Carlo** : vérification des prix par simulation stochastique

## Stack technique

- Python 3.x
- numpy, scipy — calculs numériques et distributions
- pandas — manipulation de données
- matplotlib, seaborn, plotly — visualisations statiques et interactives
- jupyter — notebooks d'analyse

## Structure

```
options-pricer/
├── src/          # Code source (pricer, greeks, monte carlo)
├── notebooks/    # Notebooks Jupyter d'analyse
├── figures/      # Graphiques exportés
├── requirements.txt
└── README.md
```

## Statut

🚧 En construction
