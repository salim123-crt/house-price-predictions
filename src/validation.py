"""Validation des saisies avant prédiction.

validate_input renvoie deux listes de messages :
- errors   : la saisie est incohérente -> pas de prédiction ;
- warnings : la saisie est valide mais sort de ce que le modèle a vu à l'entraînement
             (extrapolation) -> prédiction possible, à lire avec prudence.
"""
import pandas as pd

from src import config

NUMERIC_INPUTS = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
                  "view", "condition", "sqft_basement", "yr_built", "yr_renovated"]
ALLOWED_FLOORS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def validate_input(record, meta=None):
    errors, warnings = [], []

    # 1. Champs présents et numériques
    missing = [c for c in config.MODEL_INPUT_COLUMNS if record.get(c) in (None, "")]
    if missing:
        return [f"Champs manquants : {', '.join(missing)}"], warnings
    v = {}
    for key in NUMERIC_INPUTS:
        try:
            v[key] = float(record[key])
        except (TypeError, ValueError):
            errors.append(f"« {config.FEATURE_LABELS.get(key, key)} » doit être un nombre.")
    try:
        sale = pd.Timestamp(record["date"])
    except (TypeError, ValueError):
        errors.append("La date de vente n'est pas une date valide.")
    if errors:
        return errors, warnings

    # 2. Cohérence des surfaces (le sous-sol fait partie de la surface habitable)
    if v["sqft_living"] <= 0 or v["sqft_lot"] <= 0:
        errors.append("Les surfaces habitable et du terrain doivent être positives.")
    if not 0 <= v["sqft_basement"] < v["sqft_living"]:
        errors.append("Le sous-sol fait partie de la surface habitable : il doit être positif "
                      "et strictement inférieur à celle-ci.")

    # 3. Valeurs discrètes
    if v["bedrooms"] < 0 or v["bedrooms"] != int(v["bedrooms"]):
        errors.append("Le nombre de chambres doit être un entier positif.")
    if v["bathrooms"] < 0 or v["bathrooms"] * 4 != int(v["bathrooms"] * 4):
        errors.append("Les salles de bain se comptent par quarts (ex. 1, 1.75, 2.5).")
    if v["floors"] not in ALLOWED_FLOORS:
        errors.append(f"Le nombre de niveaux doit être parmi {ALLOWED_FLOORS}.")
    if v["waterfront"] not in (0, 1):
        errors.append("« Bord de l'eau » vaut 0 (non) ou 1 (oui).")
    if v["view"] not in (0, 1, 2, 3, 4):
        errors.append("La qualité de la vue est une note entière de 0 à 4.")
    if v["condition"] not in (1, 2, 3, 4, 5):
        errors.append("L'état général est une note entière de 1 à 5.")

    # 4. Cohérence des dates : construction <= rénovation <= vente
    if not 1800 <= v["yr_built"] <= sale.year:
        errors.append(f"L'année de construction doit être entre 1800 et {sale.year}.")
    if v["yr_renovated"] != 0 and not v["yr_built"] <= v["yr_renovated"] <= sale.year:
        errors.append("L'année de rénovation vaut 0 (jamais rénovée) ou est comprise entre "
                      "l'année de construction et l'année de vente.")

    if meta is None:
        return errors, warnings

    # 5. Domaine connu du modèle (informations de models/model_metadata.json)
    start, end = pd.Timestamp(meta["period"][0]), pd.Timestamp(meta["period"][1])
    if not start <= sale <= end:
        errors.append(f"Le modèle n'a vu que des ventes du {start.date()} au {end.date()} : "
                      "choisissez une date dans cette période.")
    city, zip_code = record["city"], record["statezip"]
    if city not in meta["categories"]["city"]:
        warnings.append(f"Ville « {city} » absente de l'entraînement : traitée comme rare.")
    elif zip_code not in meta["city_to_statezip"].get(city, []):
        warnings.append(f"Le code postal {zip_code} n'a jamais été observé avec la ville {city}.")
    if zip_code not in meta["categories"]["statezip"]:
        warnings.append(f"Code postal « {zip_code} » absent de l'entraînement : traité comme rare.")
    for key, (low, high) in meta["numeric_ranges_train"].items():
        if not low <= v[key] <= high:
            warnings.append(f"{config.FEATURE_LABELS.get(key, key)} = {v[key]:g} sort de "
                            f"l'intervalle vu à l'entraînement [{low:g} ; {high:g}] : "
                            "extrapolation peu fiable.")
    return errors, warnings
