import joblib
import numpy as np
import pytest

from src import config
from src.explain import (explain, factors_table, french_summary, group_by_variable,
                         make_explainer, plot_waterfall, shap_for_plots, split_model)
from src.predict import ArtifactError, load_artifacts, predict_price, to_frame


@pytest.fixture(params=["ridge", "xgboost"])
def model(request, small_models):
    """Chaque test est exécuté avec Ridge (LinearExplainer) et XGBoost (TreeExplainer)."""
    return small_models[0][request.param]


@pytest.fixture
def X_new(small_models):
    return small_models[2]


def test_unknown_category_still_predicts(model, valid_record):
    price = predict_price(model, {**valid_record, "city": "Paris", "statezip": "WA 00000"})
    assert np.isfinite(price) and price > 0


def test_same_predictions_after_save_and_load(model, X_new, tmp_path):
    joblib.dump(model, tmp_path / "model.joblib")
    reloaded = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_array_equal(model.predict(X_new), reloaded.predict(X_new))


def test_shap_uses_the_model_features(model, X_new, background):
    _, estimator, names = split_model(model)
    contributions, _, output = explain(model, make_explainer(model, background), X_new)
    assert list(contributions.columns) == names           # mêmes noms, même ordre
    assert estimator.n_features_in_ == len(names)
    # la sortie expliquée est bien celle du modèle complet, en log1p(prix)
    np.testing.assert_allclose(np.expm1(output), model.predict(X_new), rtol=1e-5)


def test_shap_additivity(model, X_new, background):
    contributions, base, output = explain(model, make_explainer(model, background), X_new)
    np.testing.assert_allclose(base + contributions.sum(axis=1), output,
                               atol=config.SHAP_ADDITIVITY_ATOL)


def test_grouping_keeps_signed_sum(model, X_new, background):
    contributions, _, _ = explain(model, make_explainer(model, background), X_new)
    grouped = group_by_variable(contributions)
    np.testing.assert_allclose(grouped.sum(axis=1), contributions.sum(axis=1))
    city_columns = [c for c in contributions.columns if c.startswith("city__")]
    np.testing.assert_allclose(grouped["city"], contributions[city_columns].sum(axis=1))


def test_shap_plots_can_be_drawn(model, X_new, background):
    import matplotlib.pyplot as plt
    import shap
    plt.switch_backend("Agg")
    contributions, base, _ = explain(model, make_explainer(model, background), X_new)
    preprocess, _, _ = split_model(model)
    shap.plots.bar(shap_for_plots(contributions, preprocess.transform(X_new), base), show=False)
    factors = factors_table(group_by_variable(contributions).iloc[0], X_new.iloc[[0]])
    plot_waterfall(factors, base)
    plt.close("all")


def test_full_path_train_save_load_predict_explain(small_models, background, tmp_path,
                                                   valid_record):
    joblib.dump(small_models[0]["ridge"], tmp_path / "model.joblib")
    model = joblib.load(tmp_path / "model.joblib")
    x = to_frame(valid_record)
    price = predict_price(model, valid_record)
    contributions, base, output = explain(model, make_explainer(model, background), x)
    assert np.isclose(np.expm1(output[0]), price)
    factors = factors_table(group_by_variable(contributions).iloc[0], x)
    text = french_summary(price, base, factors)
    assert "log1p" in text and "$" in text


def test_saved_model_if_trained(valid_record):
    """Vérifie le modèle réellement sauvegardé par `python -m src.train`."""
    try:
        model, meta, background = load_artifacts()
    except ArtifactError:
        pytest.skip("modèle non entraîné : lancez python -m src.train")
    _, _, names = split_model(model)
    assert names == meta["transformed_feature_names"]
    _, _, output = explain(model, make_explainer(model, background), to_frame(valid_record))
    assert np.isclose(np.expm1(output[0]), predict_price(model, valid_record))
