"""Configuration centralisée du projet.

Tous les chemins, graines aléatoires, listes de variables, seuils de nettoyage
et grilles d'hyperparamètres sont définis ici, pour qu'aucun module ne contienne
de « nombre magique » caché.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Chemins (relatifs à la racine du dépôt -> portables, sans Google Drive)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"          # copie originale, jamais modifiée
PROCESSED_DIR = DATA_DIR / "processed"                   # généré par src.train
CLEAN_DATA_PATH = PROCESSED_DIR / "clean.csv"
SPLIT_PATH = PROCESSED_DIR / "split.csv"                 # row_id -> train / test

MODELS_DIR = ROOT_DIR / "models"                         # généré, non versionné
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.csv"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Empreinte SHA-256 du fichier original (vérifiée par les tests)
RAW_DATA_SHA256 = "6ee4b80d887b92bafed19d11e59614d60b78b9cfc1aed7ad5c37e8cd5f20fa1f"

# --------------------------------------------------------------------------
# Reproductibilité
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = -1  # tous les cœurs CPU ; aucun GPU requis

# --------------------------------------------------------------------------
# Schéma des données brutes
# --------------------------------------------------------------------------
TARGET = "price"
CURRENCY = "USD"
RAW_COLUMNS = [
    "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "yr_built", "yr_renovated", "street", "city", "statezip", "country",
]

# Colonnes fournies au modèle (avant création des variables dérivées).
# - street  : quasi-identifiant (presque une valeur par ligne) -> sert seulement
#             à construire les groupes du découpage, pas comme variable.
# - country : constante ("USA") -> aucune information.
# - sqft_above : exactement sqft_living - sqft_basement -> redondante.
MODEL_INPUT_COLUMNS = [
    "date", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_basement", "yr_built",
    "yr_renovated", "city", "statezip",
]

# Variables après création des variables dérivées (src.features)
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_basement", "view", "condition", "house_age", "years_since_renovation",
]
BINARY_FEATURES = ["waterfront", "has_basement", "is_renovated"]
CATEGORICAL_FEATURES = ["city", "statezip"]
# Surfaces passées en log pour le modèle linéaire (relation log-prix / log-surface)
LOG_FEATURES_LINEAR = ["sqft_living", "sqft_lot"]

# Libellés français pour l'affichage
FEATURE_LABELS = {
    "bedrooms": "Chambres",
    "bathrooms": "Salles de bain",
    "sqft_living": "Surface habitable (sqft)",
    "sqft_lot": "Surface du terrain (sqft)",
    "floors": "Niveaux",
    "sqft_basement": "Surface du sous-sol (sqft)",
    "view": "Qualité de la vue (0-4)",
    "condition": "État général (1-5)",
    "house_age": "Âge à la vente (ans)",
    "years_since_renovation": "Années depuis construction/rénovation",
    "waterfront": "Bord de l'eau",
    "has_basement": "Présence d'un sous-sol",
    "is_renovated": "Rénovée",
    "city": "Ville",
    "statezip": "Code postal",
}

# --------------------------------------------------------------------------
# Nettoyage
# --------------------------------------------------------------------------
# Un prix au pied carré plus de 5 fois supérieur (ou inférieur) à la médiane
# de son code postal est jugé incohérent (erreur de saisie probable).
# La règle porte sur la cohérence prix / surface / localisation, PAS sur le
# niveau de prix : les maisons chères mais plausibles sont conservées.
PPSF_RATIO_MAX = 5.0
# Les 248 dernières lignes du fichier forment un bloc distinct : re-trié par
# date, il contient tous les prix nuls et tous les prix non entiers.
# Elles sont conservées (hors prix nuls) mais signalées et évaluées à part.
SUSPECT_BLOCK_START_ROW = 4352

# Regroupement des catégories rares dans l'encodage one-hot
MIN_CATEGORY_FREQUENCY = 5

# Tranches de prix (USD) pour l'analyse des erreurs
PRICE_BANDS = [0, 300_000, 450_000, 650_000, 1_000_000, float("inf")]
PRICE_BAND_LABELS = ["< 300 k$", "300-450 k$", "450-650 k$", "650 k$-1 M$", "> 1 M$"]

# --------------------------------------------------------------------------
# Modèles et recherche d'hyperparamètres (volontairement limitée : portable CPU)
# --------------------------------------------------------------------------
SELECTION_METRIC = "mae"  # choix du modèle sur la MAE moyenne en validation croisée

PARAM_SPACES = {
    "dummy": {},
    "ridge": {
        "regressor__model__alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    },
    "random_forest": {
        "regressor__model__n_estimators": [300],
        "regressor__model__max_depth": [None, 20],
        "regressor__model__min_samples_leaf": [1, 2, 4],
        "regressor__model__max_features": [0.3, 0.5],
    },
    "xgboost": {
        "regressor__model__n_estimators": [400, 800],
        "regressor__model__learning_rate": [0.03, 0.06],
        "regressor__model__max_depth": [3, 4, 6],
        "regressor__model__subsample": [0.7, 0.9],
        "regressor__model__colsample_bytree": [0.5, 0.8],
        "regressor__model__min_child_weight": [1, 3],
        "regressor__model__reg_lambda": [1.0, 5.0],
    },
}
# Nombre de combinaisons tirées au hasard (None = grille complète)
N_ITER = {"dummy": None, "ridge": None, "random_forest": 8, "xgboost": 12}

# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
SHAP_BACKGROUND_SIZE = 200   # échantillon de référence tiré du jeu d'ENTRAÎNEMENT
SHAP_GLOBAL_SAMPLE = 500     # observations de test expliquées pour les graphiques globaux
SHAP_ADDITIVITY_ATOL = 1e-3  # tolérance (espace log) pour base + somme = sortie
SHAP_TOP_DEPENDENCE = 4      # nombre de graphiques de dépendance
