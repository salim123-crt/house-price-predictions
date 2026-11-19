"""Tests sans navigateur des pages Streamlit (outil AppTest de Streamlit)."""
import pytest
from streamlit.testing.v1 import AppTest

from src import config

PAGES = ["1_presentation.py", "2_exploration.py", "3_modeles.py", "4_prediction.py"]
needs_model = pytest.mark.skipif(not config.MODEL_PATH.exists(),
                                 reason="lancez python -m src.train && python -m src.evaluate")


def run_page(name):
    return AppTest.from_file(str(config.ROOT_DIR / "app" / "pages" / name),
                             default_timeout=120).run()


@needs_model
@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_error(page):
    assert not run_page(page).exception


@needs_model
def test_prediction_page_shows_price():
    at = run_page("4_prediction.py")
    assert any("Prix estimé" in m.label for m in at.metric)
    assert not at.error


@needs_model
def test_prediction_page_rejects_basement_larger_than_house():
    at = run_page("4_prediction.py")
    at.number_input(key="sqft_living").set_value(1000)
    at.number_input(key="sqft_basement").set_value(1200).run()
    assert any("sous-sol" in e.value for e in at.error)
    assert not any("Prix estimé" in m.label for m in at.metric)
