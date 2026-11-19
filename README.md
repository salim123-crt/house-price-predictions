# Prédiction des prix immobiliers (Seattle, 2014)

Application Streamlit qui estime le prix d'une maison et explique chaque estimation avec SHAP.
Le pipeline nettoie les données, compare quatre modèles (référence naïve, Ridge, Random Forest,
XGBoost) en validation croisée groupée par adresse, puis évalue le modèle retenu sur un jeu de
test indépendant.

## Installation

Python 3.12 :

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate
# macOS / Linux : source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

Les modèles ne sont pas versionnés : les générer une première fois (quelques minutes sur CPU),
puis lancer l'application.

```bash
python -m src.train       # nettoyage, découpage, entraînement -> models/, data/processed/, reports/
python -m src.evaluate    # évaluation sur le test, figures SHAP -> reports/
streamlit run app/streamlit_app.py
```

## Tests

```bash
python -m pytest
```

Les tests des pages Streamlit et du modèle sauvegardé sont ignorés tant que `src.train` n'a pas
été lancé.

## Structure

| Dossier | Contenu |
|---|---|
| `src/` | chargement, nettoyage, variables, modèles, entraînement, évaluation, prédiction, SHAP |
| `app/` | application Streamlit (présentation, exploration, modèles, prédiction) |
| `data/raw/data.csv` | données brutes, jamais modifiées (empreinte SHA-256 vérifiée) |
| `reports/` | résultats et figures produits par `src.train` et `src.evaluate` |
| `models/` | modèles générés (non versionnés) |
| `tests/` | tests pytest |

## Données

La source exacte et la licence de `data/raw/data.csv` ne sont pas documentées. Sa structure
ressemble à des jeux publics de ventes immobilières de la région de Seattle.
