"""Évaluation finale sur le jeu de test indépendant + figures SHAP globales.

Usage (après `python -m src.train`) :
    python -m src.evaluate

Le jeu de test n'a servi à AUCUN choix (modèle, hyperparamètres, seuils).
Toutes les métriques sont calculées en dollars, sur l'échelle d'origine.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config
from src.explain import (explain, factors_table, group_by_variable, make_explainer,
                         plot_waterfall, shap_for_plots, split_model)
from src.features import add_features
from src.models import MODEL_LABELS
from src.predict import load_artifacts, load_split_data

plt.switch_backend("Agg")  # génère les images sans ouvrir de fenêtre


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "erreur_relative_mediane": float(np.median(np.abs(y_pred - y_true) / y_true)),
        "biais_moyen": float(np.mean(y_pred - y_true)),
    }


def errors_by_band(y_true, y_pred):
    bands = pd.cut(y_true, config.PRICE_BANDS, labels=config.PRICE_BAND_LABELS, right=False)
    rows = []
    for label in config.PRICE_BAND_LABELS:
        m = np.asarray(bands == label)
        if m.sum() == 0:
            continue
        r = regression_metrics(np.asarray(y_true)[m], np.asarray(y_pred)[m])
        r.pop("r2")  # R² peu interprétable à l'intérieur d'une tranche étroite
        rows.append({"tranche_prix_reel": label, **r})
    return pd.DataFrame(rows)


def temporal_check(model, df):
    """Contrôle de robustesse : entraîner sur les ventes les plus anciennes
    (80 %) et tester sur les plus récentes, sans adresse commune."""
    cutoff = df["date"].quantile(0.8)
    tr = df[df["date"] < cutoff]
    te = df[(df["date"] >= cutoff) & ~df["address_id"].isin(tr["address_id"])]
    m = clone(model).fit(tr[config.MODEL_INPUT_COLUMNS], tr[config.TARGET])
    res = regression_metrics(te[config.TARGET], m.predict(te[config.MODEL_INPUT_COLUMNS]))
    return {"date_coupure": str(cutoff.date()), "n_train": int(len(tr)), **res}


def _save(fig, name):
    fig.savefig(config.FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(y, p, label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(y, p, s=8, alpha=0.4)
    lim = [min(y.min(), p.min()), max(y.max(), p.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Prix réel (USD, échelle log)"); ax.set_ylabel("Prix prédit (USD, échelle log)")
    ax.set_title(f"{label} : prédit vs réel (test)")
    ax = axes[1]
    ax.scatter(y, p - y, s=8, alpha=0.4)
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Prix réel (USD, échelle log)"); ax.set_ylabel("Résidu : prédit - réel (USD)")
    ax.set_title("Résidus en fonction du prix réel")
    _save(fig, "pred_vs_true_residuals.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log(p) - np.log(y), bins=50)
    ax.set_xlabel("log(prédit) - log(réel)"); ax.set_ylabel("Nombre de maisons")
    ax.set_title("Distribution des erreurs relatives (test)")
    _save(fig, "residuals_hist.png")


def plot_bands(bands):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bands["tranche_prix_reel"], bands["mae"] / 1000)
    for i, r in bands.iterrows():
        ax.text(i, r["mae"] / 1000, f"n={r['n']}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAE (milliers USD)"); ax.set_title("Erreur absolue moyenne par tranche de prix")
    _save(fig, "mae_by_price_band.png")


def shap_global(model, background, X_test):
    """Graphiques SHAP globaux sur un échantillon du test (fond = entraînement)."""
    sample = X_test.sample(n=min(config.SHAP_GLOBAL_SAMPLE, len(X_test)),
                           random_state=config.RANDOM_STATE)
    explainer = make_explainer(model, background)
    contributions, base, output = explain(model, explainer, sample)
    grouped = group_by_variable(contributions)

    # Importance globale = moyenne des |contributions| regroupées par variable d'origine
    importance = grouped.abs().mean().sort_values(ascending=False)
    imp_df = pd.DataFrame({"variable": importance.index,
                           "libelle": [config.FEATURE_LABELS[v] for v in importance.index],
                           "mean_abs_shap_log": importance.to_numpy()})
    imp_df.to_csv(config.REPORTS_DIR / "shap_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["libelle"][::-1], imp_df["mean_abs_shap_log"][::-1])
    ax.set_xlabel("Moyenne |SHAP| (espace log1p(prix))")
    ax.set_title("Importance globale par variable d'origine (one-hot regroupé)")
    _save(fig, "shap_importance_grouped.png")

    # Beeswarm : la couleur indique si la valeur de la variable est haute ou basse
    preprocess, _, _ = split_model(model)
    plt.figure()
    shap.plots.beeswarm(shap_for_plots(contributions, preprocess.transform(sample), base),
                        max_display=15, show=False)
    plt.title("Beeswarm SHAP (contributions en log1p(prix))")
    _save(plt.gcf(), "shap_beeswarm.png")

    # Dépendance : contribution SHAP en fonction de la valeur réelle de la variable
    values = add_features(sample)
    top_numeric = [v for v in importance.index if v in config.NUMERIC_FEATURES]
    top_numeric = top_numeric[:config.SHAP_TOP_DEPENDENCE]
    for v in top_numeric:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(values[v], grouped[v], s=10, alpha=0.5)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel(config.FEATURE_LABELS[v])
        ax.set_ylabel("Contribution SHAP (log1p(prix))")
        ax.set_title(f"Dépendance SHAP : {config.FEATURE_LABELS[v]}")
        _save(fig, f"shap_dependence_{v}.png")

    # Exemple individuel : la première maison de l'échantillon
    factors = factors_table(grouped.iloc[0], sample.iloc[[0]])
    factors.to_csv(config.REPORTS_DIR / "shap_example_grouped.csv", index=False)
    fig = plot_waterfall(factors, base)
    plt.title("Exemple : contributions regroupées par variable (log1p(prix))")
    _save(fig, "shap_waterfall_example_grouped.png")

    additivity_error = float(np.max(np.abs(base + contributions.sum(axis=1) - output)))
    return {"n_explique": int(len(sample)), "n_fond_entrainement": int(len(background)),
            "valeur_de_base_log": base, "valeur_de_base_usd_equivalent": float(np.expm1(base)),
            "ecart_additivite_max": additivity_error, "dependence_variables": top_numeric}


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model, meta, background = load_artifacts()
    df, split = load_split_data()
    test = df[split == "test"]
    X_test, y_test = test[config.MODEL_INPUT_COLUMNS], test[config.TARGET]

    # 1) Tous les candidats sur le test (pour information ; le choix est déjà fait en CV)
    rows = []
    for name in MODEL_LABELS:
        path = config.MODELS_DIR / f"candidate_{name}.joblib"
        if path.exists():
            cand = joblib.load(path)
            rows.append({"model": name, "label": MODEL_LABELS[name],
                         **regression_metrics(y_test, cand.predict(X_test))})
    pd.DataFrame(rows).to_csv(config.REPORTS_DIR / "test_metrics_all_models.csv", index=False)

    # 2) Modèle retenu : métriques, tranches, bloc suspect, résidus
    pred = model.predict(X_test)
    test_metrics = regression_metrics(y_test, pred)
    bands = errors_by_band(y_test, pred)
    bands.to_csv(config.REPORTS_DIR / "errors_by_price_band.csv", index=False)
    by_block = pd.DataFrame([
        {"sous_ensemble": lab, **regression_metrics(y_test[m], pred[m.values])}
        for lab, m in [("lignes principales", ~test["suspect_block"]),
                       ("bloc final signalé", test["suspect_block"])] if m.sum() > 1])
    by_block.to_csv(config.REPORTS_DIR / "errors_by_block.csv", index=False)
    pd.DataFrame({"row_id": test["row_id"], "prix_reel": y_test, "prix_predit": pred,
                  "residu": pred - y_test}).to_csv(config.REPORTS_DIR / "test_predictions.csv",
                                                     index=False)
    plot_predictions(y_test.values, pred, meta["selected_label"])
    plot_bands(bands)

    # 3) Robustesse temporelle (le modèle est ré-entraîné, avec les mêmes hyperparamètres)
    temporal = temporal_check(model, df)

    # 4) SHAP global
    shap_info = shap_global(model, background, X_test)

    summary = {"modele_retenu": meta["selected_label"], "test": test_metrics,
               "controle_temporel": temporal, "shap": shap_info}
    (config.REPORTS_DIR / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTous les candidats sur le test :")
    print(pd.DataFrame(rows)[["label", "mae", "rmse", "r2"]].to_string(index=False))
    print("\nErreurs par tranche de prix :\n" + bands.to_string(index=False))
    print("\nErreurs par sous-ensemble :\n" + by_block[["sous_ensemble", "n", "mae", "r2"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
