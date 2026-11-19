"""Entraînement : nettoyage -> découpage par groupes -> validation croisée -> sauvegarde.

Usage :
    python -m src.train

Points clés contre la fuite de données :
- le jeu de test (20 %) est mis de côté AVANT tout apprentissage ;
- le découpage est fait par adresse (GroupShuffleSplit / GroupKFold) : deux
  ventes à la même adresse ne peuvent pas être de part et d'autre ;
- le prétraitement fait partie du pipeline, il est donc ré-appris dans chaque
  pli de validation croisée ;
- le modèle et ses hyperparamètres sont choisis sur la CV uniquement.
"""
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     RandomizedSearchCV, cross_validate)

from src import config
from src.data import audit_raw, clean_data, file_sha256, load_raw
from src.models import MODEL_LABELS, MODEL_NAMES, build_model

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def make_split(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["address_id"]))
    return train_idx, test_idx


def make_cv():
    """Validation croisée groupée : une adresse est toujours entièrement dans un seul pli."""
    return GroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def tune_model(name, X, y, groups):
    space = config.PARAM_SPACES[name]
    n_iter = config.N_ITER[name]
    common = dict(scoring=SCORING, refit=config.SELECTION_METRIC, cv=make_cv(),
                  n_jobs=1, error_score="raise", return_train_score=False)
    if n_iter is None:
        search = GridSearchCV(build_model(name), space, **common)
    else:
        search = RandomizedSearchCV(build_model(name), space, n_iter=n_iter,
                                    random_state=config.RANDOM_STATE, **common)
    search.fit(X, y, groups=groups)
    return search


def summarize_search(name, search, seconds):
    i = search.best_index_
    r = search.cv_results_
    return {
        "model": name,
        "label": MODEL_LABELS[name],
        "cv_mae_mean": -r["mean_test_mae"][i], "cv_mae_std": r["std_test_mae"][i],
        "cv_rmse_mean": -r["mean_test_rmse"][i], "cv_rmse_std": r["std_test_rmse"][i],
        "cv_r2_mean": r["mean_test_r2"][i], "cv_r2_std": r["std_test_r2"][i],
        "n_candidates": len(r["params"]),
        "search_seconds": round(seconds, 1),
        "best_params": json.dumps({k.split("__")[-1]: v for k, v in search.best_params_.items()},
                                  default=str),
    }


def location_ablation(name, best_params, X, y, groups):
    """Apport de city / statezip : même modèle, mêmes hyperparamètres, même CV."""
    rows = []
    for cats in ([], ["city"], ["statezip"], ["city", "statezip"]):
        model = build_model(name, categorical=cats).set_params(**best_params)
        res = cross_validate(model, X, y, groups=groups, cv=make_cv(), scoring=SCORING)
        rows.append({
            "variables_localisation": " + ".join(cats) if cats else "aucune",
            "cv_mae_mean": -res["test_mae"].mean(), "cv_mae_std": res["test_mae"].std(),
            "cv_rmse_mean": -res["test_rmse"].mean(), "cv_r2_mean": res["test_r2"].mean(),
        })
    return pd.DataFrame(rows)


def build_metadata(name, search, df_train, df_test, cv_table, raw_sha):
    model = search.best_estimator_
    pre = model.regressor_.named_steps["preprocess"]
    num_ranges = {c: [float(df_train[c].min()), float(df_train[c].max())]
                  for c in ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                            "sqft_basement", "yr_built"]}
    city_zip = (df_train.groupby("city")["statezip"].apply(lambda s: sorted(s.unique()))
                .to_dict())
    import shap
    import xgboost
    return {
        "selected_model": name,
        "selected_label": MODEL_LABELS[name],
        "selection_rule": f"plus faible {config.SELECTION_METRIC.upper()} moyenne en CV "
                          f"({config.CV_FOLDS} plis groupés par adresse)",
        "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
        "target": config.TARGET,
        "target_transform": "log1p (prédictions reconverties par expm1)",
        "shap_output_space": "log1p(prix en USD)",
        "currency": config.CURRENCY,
        "model_input_columns": config.MODEL_INPUT_COLUMNS,
        "transformed_feature_names": list(pre.get_feature_names_out()),
        "categories": {"city": sorted(df_train["city"].unique().tolist()),
                       "statezip": sorted(df_train["statezip"].unique().tolist())},
        "city_to_statezip": city_zip,
        "numeric_ranges_train": num_ranges,
        "period": [str(df_train["date"].min().date()), str(df_train["date"].max().date())],
        "n_train": int(len(df_train)), "n_test": int(len(df_test)),
        "cv_results": cv_table.to_dict(orient="records"),
        "raw_data_sha256": raw_sha,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__,
                     "xgboost": xgboost.__version__, "shap": shap.__version__},
    }


def main():
    t0 = time.time()
    for d in (config.PROCESSED_DIR, config.MODELS_DIR, config.REPORTS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    raw_sha = file_sha256(config.RAW_DATA_PATH)
    if raw_sha != config.RAW_DATA_SHA256:
        print("ATTENTION : l'empreinte du fichier original diffère de celle attendue.")
    raw = load_raw()
    audit = audit_raw(raw)
    (config.REPORTS_DIR / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    df, cleaning = clean_data(raw)
    cleaning.to_csv(config.REPORTS_DIR / "cleaning_report.csv", index=False)
    df.to_csv(config.CLEAN_DATA_PATH, index=False)
    print(f"[1/5] Données : {len(raw)} lignes brutes -> {len(df)} lignes nettoyées")

    train_idx, test_idx = make_split(df)
    split = pd.Series("train", index=df.index)
    split.iloc[test_idx] = "test"
    pd.DataFrame({"row_id": df["row_id"], "split": split}).to_csv(config.SPLIT_PATH, index=False)
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    assert not set(df_train["address_id"]) & set(df_test["address_id"]), "fuite d'adresse"
    X_train = df_train[config.MODEL_INPUT_COLUMNS]
    y_train = df_train[config.TARGET]
    groups = df_train["address_id"]
    print(f"[2/5] Découpage par adresse : train={len(df_train)}, test={len(df_test)}")

    rows, searches = [], {}
    for name in MODEL_NAMES:
        t = time.time()
        search = tune_model(name, X_train, y_train, groups)
        searches[name] = search
        rows.append(summarize_search(name, search, time.time() - t))
        print(f"      {MODEL_LABELS[name]:<30} CV MAE = {rows[-1]['cv_mae_mean']:>10,.0f} $"
              f"  ({rows[-1]['search_seconds']} s)")
    cv_table = pd.DataFrame(rows).sort_values("cv_mae_mean").reset_index(drop=True)
    cv_table.to_csv(config.REPORTS_DIR / "cv_results.csv", index=False)
    best_name = cv_table.loc[0, "model"]
    print(f"[3/5] Modèle retenu (CV) : {MODEL_LABELS[best_name]}")

    ablation = location_ablation(best_name, searches[best_name].best_params_,
                                 X_train, y_train, groups)
    ablation.to_csv(config.REPORTS_DIR / "location_ablation.csv", index=False)
    print("[4/5] Étude city / statezip :\n" + ablation.to_string(index=False))

    # Tous les candidats sont sauvegardés (évaluation finale sur le test dans src.evaluate)
    for name, search in searches.items():
        joblib.dump(search.best_estimator_, config.MODELS_DIR / f"candidate_{name}.joblib")
    joblib.dump(searches[best_name].best_estimator_, config.MODEL_PATH)
    background = X_train.sample(n=min(config.SHAP_BACKGROUND_SIZE, len(X_train)),
                                random_state=config.RANDOM_STATE)
    background.to_csv(config.SHAP_BACKGROUND_PATH, index=False)
    meta = build_metadata(best_name, searches[best_name], df_train, df_test, cv_table, raw_sha)
    config.METADATA_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                                    encoding="utf-8")
    print(f"[5/5] Modèle sauvegardé : {config.MODEL_PATH}  ({time.time() - t0:.0f} s au total)")


if __name__ == "__main__":
    main()
