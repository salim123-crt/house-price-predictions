from src.validation import validate_input

META = {
    "period": ["2014-05-02", "2014-07-10"],
    "categories": {"city": ["Seattle", "Kent"], "statezip": ["WA 98115", "WA 98031"]},
    "city_to_statezip": {"Seattle": ["WA 98115"], "Kent": ["WA 98031"]},
    "numeric_ranges_train": {"sqft_living": [370, 10040], "yr_built": [1900, 2014]},
}


def test_valid_input(valid_record):
    errors, warnings = validate_input(valid_record, META)
    assert errors == [] and warnings == []


def test_basement_larger_than_living(valid_record):
    errors, _ = validate_input({**valid_record, "sqft_basement": 1800}, META)
    assert any("sous-sol" in e for e in errors)


def test_renovation_before_construction(valid_record):
    errors, _ = validate_input({**valid_record, "yr_renovated": 1960}, META)
    assert any("rénovation" in e for e in errors)


def test_construction_after_sale(valid_record):
    errors, _ = validate_input({**valid_record, "yr_built": 2020}, META)
    assert errors


def test_date_outside_period(valid_record):
    errors, _ = validate_input({**valid_record, "date": "2024-01-01"}, META)
    assert any("période" in e for e in errors)


def test_missing_and_non_numeric(valid_record):
    errors, _ = validate_input({**valid_record, "sqft_living": None}, META)
    assert "Champs manquants" in errors[0]
    errors, _ = validate_input({**valid_record, "sqft_lot": "grand"}, META)
    assert "nombre" in errors[0]


def test_invalid_discrete_values(valid_record):
    for change in [{"floors": 1.2}, {"bathrooms": 1.3}, {"condition": 6}, {"view": -1}]:
        errors, _ = validate_input({**valid_record, **change}, META)
        assert errors, change


def test_unknown_city_is_warning_not_error(valid_record):
    errors, warnings = validate_input({**valid_record, "city": "Paris",
                                       "statezip": "WA 00000"}, META)
    assert errors == []
    assert any("Paris" in w for w in warnings) and any("WA 00000" in w for w in warnings)


def test_inconsistent_city_zip_warning(valid_record):
    errors, warnings = validate_input({**valid_record, "statezip": "WA 98031"}, META)
    assert errors == [] and any("jamais été observé" in w for w in warnings)


def test_extrapolation_warning(valid_record):
    errors, warnings = validate_input({**valid_record, "sqft_living": 20000,
                                       "sqft_basement": 0}, META)
    assert errors == [] and any("extrapolation" in w for w in warnings)
