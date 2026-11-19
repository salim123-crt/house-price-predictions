"""Point d'entrée Streamlit.

Lancement (depuis la racine du dépôt) :
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Estimation immobilière · Seattle 2014",
                   page_icon=":material/home:", layout="wide")

pages = [
    st.Page("pages/1_presentation.py", title="Présentation", icon=":material/info:", default=True),
    st.Page("pages/2_exploration.py", title="Exploration des données", icon=":material/insights:"),
    st.Page("pages/3_modeles.py", title="Comparaison des modèles", icon=":material/leaderboard:"),
    st.Page("pages/4_prediction.py", title="Prédiction et explication SHAP",
            icon=":material/calculate:"),
]
st.navigation(pages).run()
