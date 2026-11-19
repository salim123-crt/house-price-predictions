import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import DataError, file_sha256, load_raw
from src.features import add_features
from src.train import make_split


def test_raw_file_is_intact():
    assert file_sha256(config.RAW_DATA_PATH) == config.RAW_DATA_SHA256


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(DataError, match="introuvable"):
        load_raw(tmp_path / "absent.csv")


def test_missing_column_gives_clear_error(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"price": [1]}).to_csv(p, index=False)
    with pytest.raises(DataError, match="Colonnes manquantes"):
        load_raw(p)


def test_cleaning_rules(raw_df, clean_df):
    assert len(raw_df) == 4600
    assert (clean_df["price"] > 0).all()
    assert not ((clean_df["yr_renovated"] > 0)
                & (clean_df["yr_renovated"] < clean_df["yr_built"])).any()
    assert not (clean_df["bedrooms"] == 0).any()
    ppsf = clean_df["price"] / clean_df["sqft_living"]
    ratio = ppsf / ppsf.groupby(clean_df["statezip"]).transform("median")
    assert ratio.max() < config.PPSF_RATIO_MAX * 1.5  # médianes recalculées après filtre
    assert len(clean_df) == 4543


def test_plausible_expensive_houses_are_kept(clean_df):
    # 7,06 M$ pour 10 040 sqft en bord de l'eau à Bellevue : cher mais cohérent
    assert (clean_df["price"] == 7_062_500).any()
    assert (clean_df["price"] > 3_000_000).sum() >= 4


def test_implausible_price_removed(clean_df):
    assert not (clean_df["price"] == 26_590_000).any()


def test_split_has_no_address_overlap(clean_df):
    tr, te = make_split(clean_df)
    assert not set(clean_df.iloc[tr]["address_id"]) & set(clean_df.iloc[te]["address_id"])
    assert abs(len(te) / len(clean_df) - config.TEST_SIZE) < 0.03


def test_derived_features():
    X = pd.DataFrame({
        "date": pd.to_datetime(["2014-06-01", "2014-06-01"]),
        "bedrooms": [3, 3], "bathrooms": [2, 2], "sqft_living": [2000, 2000],
        "sqft_lot": [5000, 5000], "floors": [1, 1], "waterfront": [0, 0], "view": [0, 0],
        "condition": [3, 3], "sqft_basement": [0, 500], "yr_built": [1980, 1980],
        "yr_renovated": [0, 2005], "city": ["Seattle", "Seattle"],
        "statezip": ["WA 98115", "WA 98115"],
    })
    f = add_features(X)
    assert f["house_age"].tolist() == [34, 34]
    assert f["is_renovated"].tolist() == [0, 1]
    assert f["years_since_renovation"].tolist() == [34, 9]  # 0 = non rénové -> âge
    assert f["has_basement"].tolist() == [0, 1]
    assert list(f.columns) == (config.NUMERIC_FEATURES + config.BINARY_FEATURES
                               + config.CATEGORICAL_FEATURES)
    assert not f.isna().any().any() and np.isfinite(f[config.NUMERIC_FEATURES]).all().all()
