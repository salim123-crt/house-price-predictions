import altair as alt
import streamlit as st

from app.common import missing_report_message, read_report, require_artifacts, usd
from src import config

model, meta, _ = require_artifacts()
cv = read_report("cv_results.csv")
test_all = read_report("test_metrics_all_models.csv")
metrics = read_report("metrics.json")
bands = read_report("errors_by_price_band.csv")
blocks = read_report("errors_by_block.csv")
ablation = read_report("location_ablation.csv")

st.title("Comparaison et évaluation des modèles")
st.markdown(
    f"""
**Protocole.** Les données nettoyées sont coupées **une seule fois** : 80 % pour
l'entraînement, 20 % pour le test, **par adresse** (deux ventes à la même adresse ne sont
jamais séparées). Sur l'entraînement, chaque modèle est réglé par validation croisée à
{config.CV_FOLDS} plis, eux aussi groupés par adresse ; le prétraitement est ré-appris dans
chaque pli. La cible est apprise en **log1p(prix)** ; toutes les métriques sont calculées
**en dollars**, après retour à l'échelle d'origine.

**Règle de choix** : {meta['selection_rule']}. Modèle retenu : **{meta['selected_label']}**.
"""
)

st.subheader("1. Validation croisée (jeu d'entraînement : sert au choix)")
if cv is not None:
    show = cv[["label", "cv_mae_mean", "cv_mae_std", "cv_rmse_mean", "cv_r2_mean", "cv_r2_std",
               "n_candidates", "search_seconds", "best_params"]].rename(columns={
        "label": "Modèle", "cv_mae_mean": "MAE moy. ($)", "cv_mae_std": "MAE écart-type",
        "cv_rmse_mean": "RMSE moy. ($)", "cv_r2_mean": "R² moy.", "cv_r2_std": "R² écart-type",
        "n_candidates": "Combinaisons testées", "search_seconds": "Durée (s)",
        "best_params": "Meilleurs hyperparamètres"})
    st.dataframe(show.style.format({"MAE moy. ($)": "{:,.0f}", "MAE écart-type": "{:,.0f}",
                                    "RMSE moy. ($)": "{:,.0f}", "R² moy.": "{:.3f}",
                                    "R² écart-type": "{:.3f}"}),
                 hide_index=True, width="stretch")
    best, second = cv.iloc[0], cv.iloc[1]
    gap = second["cv_mae_mean"] - best["cv_mae_mean"]
    st.caption(f"Écart de MAE entre les deux premiers : {usd(gap)}, à comparer à l'écart-type "
               f"entre plis ({usd(best['cv_mae_std'])}) : les deux meilleurs modèles sont "
               "proches ; la règle fixée à l'avance tranche.")

st.subheader("2. Jeu de test indépendant (évaluation finale)")
if metrics is None or test_all is None:
    missing_report_message("Les métriques de test")
else:
    t = metrics["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", usd(t["mae"]))
    c2.metric("RMSE", usd(t["rmse"]))
    c3.metric("R²", f"{t['r2']:.3f}")
    c4.metric("Erreur relative médiane", f"{100 * t['erreur_relative_mediane']:.1f} %")
    st.caption(f"Modèle retenu, {t['n']} maisons jamais vues pendant l'entraînement.")
    st.dataframe(test_all[["label", "mae", "rmse", "r2", "erreur_relative_mediane"]].rename(
        columns={"label": "Modèle", "mae": "MAE ($)", "rmse": "RMSE ($)", "r2": "R²",
                 "erreur_relative_mediane": "Erreur relative médiane"})
        .style.format({"MAE ($)": "{:,.0f}", "RMSE ($)": "{:,.0f}", "R²": "{:.3f}",
                       "Erreur relative médiane": "{:.1%}"}),
        hide_index=True, width="stretch")
    st.caption("Tous les candidats sont affichés sur le test à titre d'information : le choix "
               "a été fait avant, sur la validation croisée.")

st.subheader("3. Erreurs par tranche de prix réel")
if bands is not None:
    st.altair_chart(alt.Chart(bands).mark_bar().encode(
        alt.X("tranche_prix_reel:N", sort=config.PRICE_BAND_LABELS, title="Tranche de prix réel"),
        alt.Y("mae:Q", title="MAE (USD)"),
        tooltip=["n", alt.Tooltip("mae:Q", format=",.0f"),
                 alt.Tooltip("erreur_relative_mediane:Q", format=".1%"),
                 alt.Tooltip("biais_moyen:Q", format=",.0f")]), width="stretch")
    st.dataframe(bands.rename(columns={"tranche_prix_reel": "Tranche", "mae": "MAE ($)",
                                       "rmse": "RMSE ($)", "biais_moyen": "Biais moyen ($)",
                                       "erreur_relative_mediane": "Erreur rel. médiane"})
                 .style.format({"MAE ($)": "{:,.0f}", "RMSE ($)": "{:,.0f}",
                                "Biais moyen ($)": "{:+,.0f}", "Erreur rel. médiane": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.caption("Biais = moyenne(prédit - réel). Un biais négatif sur les biens chers signifie "
               "que le modèle les sous-estime en moyenne (effet classique de régression "
               "vers la moyenne).")

st.subheader("4. Résidus")
for f in ["pred_vs_true_residuals.png", "residuals_hist.png"]:
    p = config.FIGURES_DIR / f
    if p.exists():
        st.image(str(p))

st.subheader("5. Contrôles complémentaires")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Apport de la localisation (CV, modèle retenu)**")
    if ablation is not None:
        st.dataframe(ablation.rename(columns={"variables_localisation": "Variables",
                                              "cv_mae_mean": "MAE ($)", "cv_mae_std": "Écart-type",
                                              "cv_rmse_mean": "RMSE ($)", "cv_r2_mean": "R²"})
                     .style.format({"MAE ($)": "{:,.0f}", "Écart-type": "{:,.0f}",
                                    "RMSE ($)": "{:,.0f}", "R²": "{:.3f}"}),
                     hide_index=True, width="stretch")
with c2:
    st.markdown("**Bloc final du fichier (lignes signalées)**")
    if blocks is not None:
        st.dataframe(blocks[["sous_ensemble", "n", "mae", "erreur_relative_mediane"]].rename(
            columns={"sous_ensemble": "Sous-ensemble", "mae": "MAE ($)",
                     "erreur_relative_mediane": "Erreur rel. médiane"})
            .style.format({"MAE ($)": "{:,.0f}", "Erreur rel. médiane": "{:.1%}"}),
            hide_index=True, width="stretch")
if metrics:
    tc = metrics["controle_temporel"]
    st.markdown(
        f"**Découpage temporel** (robustesse) : entraînement sur les ventes avant le "
        f"{tc['date_coupure']} ({tc['n_train']} maisons), test sur les {tc['n']} suivantes : "
        f"MAE = {usd(tc['mae'])}, R² = {tc['r2']:.3f}. Sur une période de 10 semaines, "
        "l'écart avec le découpage aléatoire groupé permet de vérifier l'absence de forte "
        "dérive temporelle.")
st.caption("Aucun intervalle de confiance n'est affiché : le modèle ne fournit pas "
           "d'incertitude calibrée par prédiction.")
