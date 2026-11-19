"""Variables dérivées et prétraitement (ColumnTransformer).

`add_features` est une transformation ligne par ligne, sans paramètre appris :
elle est placée DANS le pipeline, ce qui garantit exactement le même calcul à
l'entraînement, en validation croisée et dans l'application.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_features(X):
    """Crée les variables dérivées à partir des colonnes d'entrée du modèle.

    - house_age : âge du logement à la date de vente (année de vente - yr_built)
    - is_renovated : yr_renovated > 0 (0 signifie « jamais rénové / inconnu »)
    - years_since_renovation : années depuis la dernière intervention
      (rénovation si elle existe, sinon construction)
    - has_basement : présence d'un sous-sol
    """
    X = X.copy()
    sale_year = pd.to_datetime(X["date"]).dt.year
    yr_built = X["yr_built"].astype(float)
    yr_renov = X["yr_renovated"].fillna(0).astype(float)
    X["house_age"] = (sale_year - yr_built).clip(lower=0)
    X["is_renovated"] = (yr_renov > 0).astype(int)
    last_work = np.where(yr_renov > 0, np.maximum(yr_renov, yr_built), yr_built)
    X["years_since_renovation"] = (sale_year - last_work).clip(lower=0)
    X["has_basement"] = (X["sqft_basement"] > 0).astype(int)
    cols = config.NUMERIC_FEATURES + config.BINARY_FEATURES + config.CATEGORICAL_FEATURES
    return X[cols]


def build_preprocessor(linear,
                       categorical = None):
    """Prétraitement adapté à la famille de modèle.

    linear=True  : log des surfaces + standardisation (utile pour Ridge).
    linear=False : pas de mise à l'échelle (inutile pour les arbres).
    Les catégories inconnues ou rares sont regroupées dans une colonne
    « infrequent » (handle_unknown='infrequent_if_exist') : pas d'erreur
    à la prédiction pour une ville jamais vue.
    """
    categorical = config.CATEGORICAL_FEATURES if categorical is None else categorical
    num_cols = list(config.NUMERIC_FEATURES)

    transformers = []
    if linear:
        log_cols = [c for c in config.LOG_FEATURES_LINEAR if c in num_cols]
        other = [c for c in num_cols if c not in log_cols]
        transformers.append(("log", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), log_cols))
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), other))
    else:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))

    transformers.append(("bin", SimpleImputer(strategy="most_frequent"),
                         config.BINARY_FEATURES))
    # Un transformateur par variable catégorielle -> regroupement SHAP trivial
    for col in categorical:
        transformers.append((col, Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=config.MIN_CATEGORY_FREQUENCY,
                                     sparse_output=False)),
        ]), [col]))

    return ColumnTransformer(transformers, remainder="drop",
                             verbose_feature_names_out=True)


def feature_builder():
    return FunctionTransformer(add_features, validate=False)


def original_feature_of(transformed_name):
    """Variable d'origine d'une colonne transformée.

    'num__sqft_living' -> 'sqft_living' ; 'city__city_Seattle' -> 'city'.
    """
    prefix, rest = transformed_name.split("__", 1)
    if prefix in ("num", "log", "bin"):
        return rest
    return prefix


def pretty_feature_name(transformed_name):
    """Nom lisible : 'city__city_Seattle' -> 'Ville = Seattle'."""
    orig = original_feature_of(transformed_name)
    label = config.FEATURE_LABELS.get(orig, orig)
    if orig in config.CATEGORICAL_FEATURES:
        value = transformed_name.split("__", 1)[1][len(orig) + 1:]
        if value == "infrequent_sklearn":
            value = "catégorie rare/inconnue"
        return f"{label} = {value}"
    return label
