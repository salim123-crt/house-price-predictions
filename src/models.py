"""Définition des modèles candidats.

Chaque modèle est un TransformedTargetRegressor :
    entrée brute -> add_features -> ColumnTransformer -> régresseur
et la cible est apprise en log1p(prix), puis reconvertie en dollars par expm1.
Les prédictions (et donc les métriques) sont toujours en dollars.
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src import config
from src.features import build_preprocessor, feature_builder

MODEL_LABELS = {
    "dummy": "DummyRegressor (référence)",
    "ridge": "Ridge (linéaire régularisé)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_NAMES = ["dummy", "ridge", "random_forest", "xgboost"]


def _estimator(name):
    if name == "dummy":
        return DummyRegressor(strategy="mean")  # moyenne du log = moyenne géométrique
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_STATE,
                                     n_jobs=config.N_JOBS)
    if name == "xgboost":
        return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                            tree_method="hist", random_state=config.RANDOM_STATE,
                            n_jobs=config.N_JOBS)
    raise ValueError(f"Modèle inconnu : {name}")


def build_model(name, categorical=None):
    pipe = Pipeline([
        ("features", feature_builder()),
        ("preprocess", build_preprocessor(linear=(name == "ridge"), categorical=categorical)),
        ("model", _estimator(name)),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1,
                                      check_inverse=False)
