"""Chargement du modèle sauvegardé et prédiction (sans ré-entraînement)."""
import json

import joblib
import pandas as pd

from src import config


class ArtifactError(FileNotFoundError):
    """Artefact (modèle, métadonnées, données traitées) absent."""


def _require(path, what):
    if not path.exists():
        raise ArtifactError(
            f"{what} introuvable ({path.relative_to(config.ROOT_DIR)}). "
            "Lancez d'abord : python -m src.train"
        )


def load_artifacts():
    """Renvoie (modèle, métadonnées, fond SHAP issu de l'entraînement)."""
    _require(config.MODEL_PATH, "Modèle")
    _require(config.METADATA_PATH, "Métadonnées du modèle")
    _require(config.SHAP_BACKGROUND_PATH, "Échantillon de référence SHAP")
    model = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    background = pd.read_csv(config.SHAP_BACKGROUND_PATH, parse_dates=["date"])
    return model, meta, background


def load_split_data():
    """Données nettoyées + étiquette train/test alignée (même ordre)."""
    _require(config.CLEAN_DATA_PATH, "Données nettoyées")
    _require(config.SPLIT_PATH, "Découpage train/test")
    df = pd.read_csv(config.CLEAN_DATA_PATH, parse_dates=["date"])
    split = pd.read_csv(config.SPLIT_PATH)
    if not (split["row_id"].values == df["row_id"].values).all():
        raise ArtifactError("split.csv ne correspond pas à clean.csv : relancez src.train")
    return df, pd.Series(split["split"].values, index=df.index)


def to_frame(record):
    """Une saisie (dict) -> DataFrame d'une ligne au schéma attendu par le modèle."""
    row = {c: record.get(c) for c in config.MODEL_INPUT_COLUMNS}
    df = pd.DataFrame([row])
    df["date"] = pd.to_datetime(df["date"])
    return df


def predict_price(model, record):
    return float(model.predict(to_frame(record))[0])
