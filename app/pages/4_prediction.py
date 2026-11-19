import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from app.common import (get_artifacts, get_explainer, read_report, require_artifacts,
                        require_data, scope_banner, usd)
from src import config
from src.explain import explain, factors_table, french_summary, group_by_variable, plot_waterfall
from src.predict import to_frame
from src.validation import validate_input

model, meta, _ = require_artifacts()
df, split = require_data()
period = [dt.date.fromisoformat(d) for d in meta["period"]]
INT_FIELDS = ["bedrooms", "sqft_living", "sqft_lot", "sqft_basement", "view", "condition",
              "waterfront", "yr_built", "yr_renovated"]

# ------------------------------------------------------------------ exemples
# 6 maisons du jeu de TEST (jamais vues à l'entraînement), de la moins chère à la plus chère.
test = df[split == "test"]
known_zip = pd.Series([z in meta["city_to_statezip"].get(c, [])
                       for c, z in zip(test["city"], test["statezip"])], index=test.index)
test = test[known_zip & test["bedrooms"].notna() & test["bathrooms"].notna()].sort_values("price")
examples = test.iloc[[int(q * (len(test) - 1)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]]


def example_record(i):
    row = examples.iloc[i]
    record = {col: row[col] for col in config.MODEL_INPUT_COLUMNS}
    record["date"] = row["date"].date()
    for col in INT_FIELDS:
        record[col] = int(record[col])
    record["bathrooms"], record["floors"] = float(record["bathrooms"]), float(record["floors"])
    return record


def load_example(i):
    """Remplit le formulaire avec l'exemple i (appelé par le bouton)."""
    st.session_state.update(example_record(i))
    st.session_state["example"] = i


if "example" not in st.session_state:
    load_example(2)  # exemple médian au premier affichage


# ------------------------------------------------------------------ calcul (mis en cache)
@st.cache_data(show_spinner=False, max_entries=256)
def estimate(record_items):
    """Prix prédit + explication SHAP d'une maison (calculés une seule fois par saisie)."""
    model, _, _ = get_artifacts()
    x = to_frame(dict(record_items))
    price = float(model.predict(x)[0])
    contributions, base, _ = explain(model, get_explainer(), x)  # vérifie l'additivité
    factors = factors_table(group_by_variable(contributions).iloc[0], x)
    return price, base, factors


# ------------------------------------------------------------------ interface
st.title("Prédiction et explication SHAP")
scope_banner(meta)
tab_house, tab_global = st.tabs(["Estimer une maison", "Explications globales du modèle"])

with tab_house:
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox(
        "Exemples préremplis (maisons du jeu de test)", range(len(examples)), index=2,
        format_func=lambda i: f"{examples.iloc[i]['city']} · {examples.iloc[i]['sqft_living']} "
                              f"sqft · vendue {usd(examples.iloc[i]['price'])}")
    c2.write("")
    c2.button("Charger l'exemple", on_click=load_example, args=(choice,), width="stretch")

    with st.container(border=True):
        a, b, c, d = st.columns(4)
        with a:
            st.markdown("**Localisation et date**")
            st.selectbox("Ville", meta["categories"]["city"], key="city")
            zips = meta["city_to_statezip"][st.session_state["city"]]
            if st.session_state.get("statezip") not in zips:  # la ville a changé
                st.session_state["statezip"] = zips[0]
            st.selectbox("Code postal", zips, key="statezip")
            st.date_input("Date de vente", min_value=period[0], max_value=period[1], key="date",
                          format="DD/MM/YYYY", help="Le modèle ne connaît que cette période.")
        with b:
            st.markdown("**Surfaces (pieds carrés)**")
            st.number_input("Surface habitable totale", 100, 20000, step=10, key="sqft_living")
            st.number_input("dont sous-sol", 0, 10000, step=10, key="sqft_basement",
                            help="Le sous-sol est inclus dans la surface habitable.")
            st.number_input("Terrain", 100, 2_000_000, step=100, key="sqft_lot")
            st.caption(f"Habitable ≈ {st.session_state['sqft_living'] * 0.0929:.0f} m² · "
                       f"terrain ≈ {st.session_state['sqft_lot'] * 0.0929:.0f} m²")
        with c:
            st.markdown("**Pièces et niveaux**")
            st.number_input("Chambres", 0, 15, step=1, key="bedrooms")
            st.number_input("Salles de bain", 0.0, 10.0, step=0.25, key="bathrooms",
                            help="Convention américaine : 0,5 = WC + lavabo, 1 = complète.")
            st.selectbox("Niveaux", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], key="floors")
        with d:
            st.markdown("**État et âge**")
            st.slider("État général (1 à 5)", 1, 5, key="condition")
            st.slider("Qualité de la vue (0 à 4)", 0, 4, key="view")
            st.selectbox("Bord de l'eau", [0, 1], key="waterfront",
                         format_func=lambda v: "Oui" if v else "Non")
            st.number_input("Année de construction", 1800, period[1].year, step=1, key="yr_built")
            st.number_input("Année de rénovation (0 = jamais)", 0, period[1].year, step=1,
                            key="yr_renovated")

    record = {col: st.session_state[col] for col in config.MODEL_INPUT_COLUMNS}

    errors, warnings = validate_input(record, meta)
    for w in warnings:
        st.warning(w, icon=":material/warning:")
    for e in errors:
        st.error(e, icon=":material/error:")
    if errors:
        st.stop()

    record["date"] = str(record["date"])
    price, base, factors = estimate(tuple(sorted(record.items())))

    r1, r2, r3 = st.columns(3)
    r1.metric("Prix estimé (USD, marché 2014)", usd(price))
    r2.metric("Valeur de base SHAP (log1p)", f"{base:.3f}",
              help="Prédiction moyenne du modèle sur 200 maisons d'entraînement.")
    r3.metric("Somme des contributions (log1p)", f"{factors['contribution'].sum():+.3f}")

    i = st.session_state["example"]
    unchanged = {**example_record(i), "date": str(example_record(i)["date"])} == record
    if unchanged:
        real = examples.iloc[i]["price"]
        st.success(f"Prix de vente réel de cet exemple : **{usd(real)}** "
                   f"(écart {usd(price - real)}).")
    metrics = read_report("metrics.json")
    if metrics:
        st.caption(f"Ordre de grandeur de l'erreur : erreur absolue moyenne mesurée sur le jeu "
                   f"de test = {usd(metrics['test']['mae'])}. Ce n'est pas un intervalle de "
                   "confiance pour cette maison.")

    st.markdown("#### Pourquoi cette estimation ?")
    st.write(french_summary(price, base, factors))

    left, right = st.columns([3, 2])
    with left:
        st.pyplot(plot_waterfall(factors, base), clear_figure=True)
        st.caption("Waterfall en **log1p(prix)** : on part de la valeur de base, chaque barre "
                   "ajoute la contribution d'une variable, et on arrive à log1p(prix estimé). "
                   "Prix = expm1(base + somme des contributions). Les barres ne sont pas des "
                   "dollars.")
    with right:
        st.markdown("**Facteurs** (colonnes one-hot regroupées par variable)")
        table = factors.assign(effet=np.where(factors["contribution"] > 0, "↑ augmente",
                                              np.where(factors["contribution"] < 0, "↓ diminue", "—")))
        st.dataframe(table[["libelle", "valeur", "contribution", "effet"]]
                     .astype({"valeur": str})
                     .rename(columns={"libelle": "Variable", "valeur": "Valeur",
                                      "contribution": "Contribution (log1p)", "effet": "Effet"})
                     .style.format({"Contribution (log1p)": "{:+.4f}"}),
                     hide_index=True, width="stretch", height=420)

    export = pd.concat([
        pd.DataFrame({"type": "saisie", "variable": list(record),
                      "valeur": [str(v) for v in record.values()]}),
        pd.DataFrame({"type": "resultat",
                      "variable": ["prix_estime_usd", "valeur_base_shap_log1p", "modele"],
                      "valeur": [f"{price:.2f}", f"{base:.6f}", meta["selected_label"]]}),
        pd.DataFrame({"type": "contribution_log1p", "variable": factors["variable"],
                      "valeur": factors["contribution"].round(6).astype(str)}),
    ])
    st.download_button("Exporter le résultat (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name="estimation_shap.csv", mime="text/csv")

    with st.expander("Comment lire ces explications ? (limites)"):
        st.markdown("""
- Les contributions sont en **log1p(prix)** : elles s'additionnent dans cet espace et ne sont
  pas des dollars.
- Une ville est encodée en plusieurs colonnes 0/1 ; on additionne leurs contributions (avec
  leur signe) pour obtenir celle de la variable « Ville ».
- Les variables **corrélées** (surface, chambres, salles de bain ; ville et code postal) se
  partagent l'effet : une contribution faible ne veut pas dire « sans lien avec le prix ».
- SHAP décrit **le comportement du modèle**, pas une relation de cause à effet.
""")

with tab_global:
    importance = read_report("shap_importance.csv")
    if importance is None:
        st.warning("Graphiques absents : lancez `python -m src.evaluate`.")
    else:
        top3 = ", ".join(importance.head(3)["libelle"])
        st.markdown(f"Calculé sur 500 maisons du jeu de test. Variables les plus influentes : "
                    f"**{top3}**.")
        cols = st.columns(2)
        for col, name in zip(cols, ["shap_importance_grouped.png", "shap_beeswarm.png"]):
            if (config.FIGURES_DIR / name).exists():
                col.image(str(config.FIGURES_DIR / name))
        st.markdown("**Graphiques de dépendance** (contribution selon la valeur de la variable)")
        cols = st.columns(2)
        for k, path in enumerate(sorted(config.FIGURES_DIR.glob("shap_dependence_*.png"))):
            cols[k % 2].image(str(path))
        if meta["selected_model"] == "ridge":
            st.caption("Avec Ridge, chaque contribution vaut coefficient × (valeur transformée − "
                       "moyenne de référence) : les dépendances sont des droites, ou des "
                       "courbes logarithmiques pour les surfaces passées en log.")
