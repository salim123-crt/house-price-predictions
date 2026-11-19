"""Fonctions partagées par les pages Streamlit (chargement mis en cache)."""
import json

import matplotlib
import pandas as pd
import streamlit as st

from src import config
from src.explain import make_explainer
from src.predict import ArtifactError, load_artifacts, load_split_data

matplotlib.use("Agg")  # graphiques générés côté serveur, sans fenêtre


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_artifacts():
    """Modèle, métadonnées et fond SHAP : chargés une seule fois (pas de ré-entraînement)."""
    return load_artifacts()


@st.cache_resource(show_spinner="Préparation de l'explainer SHAP…")
def get_explainer():
    model, _, background = get_artifacts()
    return make_explainer(model, background)


@st.cache_data(show_spinner=False)
def get_data():
    return load_split_data()


@st.cache_data(show_spinner=False)
def read_report(name):
    path = config.REPORTS_DIR / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def require_artifacts():
    """Arrête proprement la page si le modèle n'a pas encore été entraîné."""
    try:
        return get_artifacts()
    except ArtifactError as e:
        st.error(f"{e}\n\nPuis, pour les graphiques et métriques : `python -m src.evaluate`.")
        st.stop()


def require_data():
    try:
        return get_data()
    except ArtifactError as e:
        st.error(str(e))
        st.stop()


def scope_banner(meta=None):
    period = meta["period"] if meta else ["2014-05-02", "2014-07-10"]
    st.info(
        f"**Périmètre des données** : ventes de maisons du **{period[0]} au {period[1]}**, "
        "région de **Seattle (État de Washington, États-Unis)**, prix en **dollars US (USD)** "
        "de 2014, surfaces en **pieds carrés** (1 sqft ≈ 0,093 m²). "
        "Ce modèle n'estime **ni les prix actuels, ni des biens situés en France**.",
        icon=":material/info:",
    )


def usd(x):
    return f"{x:,.0f}".replace(",", " ") + " $"


def missing_report_message(what):
    st.warning(f"{what} absent : lancez `python -m src.evaluate` pour le générer.")
