"""Explications SHAP du modèle sauvegardé.

Le modèle apprend log1p(prix). Les valeurs SHAP sont donc dans cet espace :
    valeur_de_base + somme(contributions) = log1p(prix prédit)
    prix prédit = expm1(valeur_de_base + somme(contributions))
On ne convertit jamais une contribution seule en dollars.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.features import add_features, original_feature_of, pretty_feature_name


def split_model(model):
    """Découpe le modèle entraîné en : prétraitement, régresseur final, noms des colonnes."""
    pipe = model.regressor_                     # pipeline entraîné (dans le TransformedTargetRegressor)
    preprocess = pipe[:-1]                      # add_features + ColumnTransformer
    estimator = pipe.named_steps["model"]       # Ridge, RandomForest ou XGBoost
    names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    return preprocess, estimator, names


def make_explainer(model, background_raw):
    """Crée l'explainer adapté au modèle. Le fond de référence vient du jeu d'entraînement."""
    preprocess, estimator, _ = split_model(model)
    background = preprocess.transform(background_raw)
    # max_samples : sans lui, SHAP n'utiliserait que 100 des 200 lignes du fond
    masker = shap.maskers.Independent(background, max_samples=len(background))
    model_type = type(estimator).__name__
    if model_type == "Ridge":
        return shap.LinearExplainer(estimator, masker)
    if model_type in ("RandomForestRegressor", "XGBRegressor"):
        return shap.TreeExplainer(estimator, masker, feature_perturbation="interventional")
    raise TypeError(f"Pas d'explainer SHAP prévu pour {model_type}")


def explain(model, explainer, X_raw):
    """Calcule les valeurs SHAP de plusieurs maisons.

    Renvoie :
    - contributions : DataFrame (une ligne par maison, une colonne par variable transformée)
    - base : valeur de base SHAP (sortie moyenne du modèle sur le fond, en log1p)
    - output : sortie du modèle pour chaque maison (en log1p)
    """
    preprocess, estimator, names = split_model(model)
    X_t = preprocess.transform(X_raw)
    if isinstance(explainer, shap.TreeExplainer):
        sv = explainer(X_t, check_additivity=False)  # on fait notre propre vérification ci-dessous
    else:
        sv = explainer(X_t)
    contributions = pd.DataFrame(sv.values, columns=names, index=X_raw.index)
    base = float(np.ravel(sv.base_values)[0])
    output = estimator.predict(X_t)

    # Vérification d'additivité : base + somme des contributions = sortie du modèle
    error = np.max(np.abs(base + contributions.sum(axis=1).to_numpy() - output))
    if error > config.SHAP_ADDITIVITY_ATOL:
        raise AssertionError(f"Additivité SHAP non respectée (écart {error:.2e})")
    return contributions, base, output


def group_by_variable(contributions):
    """Additionne les colonnes one-hot d'une même variable (ex. toutes les city_*).

    Les signes sont conservés, donc la somme totale ne change pas.
    """
    variables = [original_feature_of(c) for c in contributions.columns]
    return contributions.T.groupby(variables, sort=False).sum().T


def factors_table(grouped_row, x_raw):
    """Tableau des facteurs d'une maison, trié par importance (|contribution|).

    grouped_row : une ligne de group_by_variable ; x_raw : la maison (DataFrame d'une ligne).
    """
    values = add_features(x_raw).iloc[0]
    table = pd.DataFrame({
        "variable": grouped_row.index,
        "libelle": [config.FEATURE_LABELS.get(v, v) for v in grouped_row.index],
        "valeur": [values[v] for v in grouped_row.index],
        "contribution": grouped_row.to_numpy(),
    })
    order = table["contribution"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def plot_waterfall(factors, base, max_display=12):
    """Waterfall SHAP (en log1p) à partir du tableau des facteurs."""
    explanation = shap.Explanation(
        values=factors["contribution"].to_numpy(dtype=float),
        base_values=base,
        data=factors["valeur"].to_numpy(dtype=object),
        feature_names=factors["libelle"].tolist(),
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 0.45 * min(len(factors), max_display) + 1.5)
    return fig


def shap_for_plots(contributions, X_t, base):
    """Objet shap.Explanation pour les graphiques globaux (bar, beeswarm)."""
    base_values = np.full(len(contributions), base)  # une valeur de base par maison
    return shap.Explanation(values=contributions.to_numpy(), base_values=base_values, data=X_t,
                            feature_names=[pretty_feature_name(c) for c in contributions.columns])


def usd(x):
    return f"{x:,.0f}".replace(",", " ") + " $"


def _show(value):
    if isinstance(value, (int, float, np.number)) and abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:g}" if isinstance(value, (float, np.floating)) else str(value)


def french_summary(price, base, factors, top=3):
    """Courte explication en français, construite à partir des contributions réelles."""
    def listing(rows):
        return ", ".join(f"{r.libelle} ({_show(r.valeur)} ; {r.contribution:+.3f})"
                         for r in rows.itertuples())

    up = factors[factors["contribution"] > 0].head(top)
    down = factors[factors["contribution"] < 0].head(top)
    text = (f"Le modèle estime ce bien à environ {usd(price)}. "
            f"Son point de départ (valeur de base SHAP) est {base:.3f} en log1p(prix), "
            f"ce qui correspond à environ {usd(np.expm1(base))}. "
            f"Les caractéristiques du bien déplacent l'estimation de "
            f"{factors['contribution'].sum():+.3f} en échelle logarithmique. ")
    if len(up):
        text += f"Ce qui l'augmente le plus : {listing(up)}. "
    if len(down):
        text += f"Ce qui la diminue le plus : {listing(down)}. "
    text += ("Les contributions sont en log1p(prix) : elles s'additionnent en échelle "
             "logarithmique et ne sont pas des dollars. Elles décrivent le comportement du "
             "modèle, pas une relation de cause à effet.")
    return text
