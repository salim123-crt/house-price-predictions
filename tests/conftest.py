import pytest

from src import config
from src.data import clean_data, load_raw
from src.models import build_model


@pytest.fixture(scope="session")
def raw_df():
    return load_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    df, _ = clean_data(raw_df)
    return df


@pytest.fixture
def valid_record():
    return {
        "date": "2014-06-15", "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800,
        "sqft_lot": 6000, "floors": 1.0, "waterfront": 0, "view": 0, "condition": 3,
        "sqft_basement": 400, "yr_built": 1975, "yr_renovated": 0,
        "city": "Seattle", "statezip": "WA 98115",
    }


@pytest.fixture(scope="session")
def small_models(clean_df):
    """Petits modèles entraînés en quelques secondes pour les tests (pas le modèle final).

    Renvoie (modèles, X d'entraînement, X de maisons non vues).
    """
    sample = clean_df.sample(n=900, random_state=0)
    X, y = sample[config.MODEL_INPUT_COLUMNS], sample[config.TARGET]
    models = {
        "ridge": build_model("ridge").fit(X, y),
        "xgboost": build_model("xgboost").set_params(regressor__model__n_estimators=60).fit(X, y),
    }
    X_new = clean_df.drop(sample.index)[config.MODEL_INPUT_COLUMNS].head(30)
    return models, X, X_new


@pytest.fixture(scope="session")
def background(small_models):
    return small_models[1].sample(n=60, random_state=1)
