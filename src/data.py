"""Chargement, audit et nettoyage des données.

Principe : le fichier original `data/raw/data.csv` n'est jamais modifié.
Chaque correction est une règle explicite, appliquée ligne par ligne, et
consignée dans un rapport (étape, règle, nombre de lignes, action).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


class DataError(Exception):
    """Erreur liée aux données (fichier absent, colonnes manquantes...)."""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path=config.RAW_DATA_PATH):
    """Charge le CSV original depuis le disque local et vérifie son schéma."""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"Fichier de données introuvable : {path}\n"
            "Placez le fichier original dans data/raw/data.csv."
        )
    df = pd.read_csv(path)
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Colonnes manquantes dans {path.name} : {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df.insert(0, "row_id", np.arange(len(df)))  # position d'origine (traçabilité)
    return df


def address_key(df):
    """Identifiant d'adresse normalisé : sert à grouper un même logement."""
    return (
        df["street"].str.strip().str.lower() + " | "
        + df["city"].str.strip().str.lower() + " | "
        + df["statezip"].str.strip()
    )


def audit_raw(df):
    """Constats factuels sur les données brutes (aucune modification)."""
    price = df["price"]
    pos = df[price > 0]
    ppsf = pos["price"] / pos["sqft_living"]
    block = df["row_id"] >= config.SUSPECT_BLOCK_START_ROW
    addr = address_key(df)
    dup_addr = addr.duplicated(keep=False)
    return {
        "n_lignes": int(len(df)),
        "n_colonnes": int(df.shape[1] - 1),
        "valeurs_manquantes": int(df[config.RAW_COLUMNS].isna().sum().sum()),
        "doublons_exacts": int(df[config.RAW_COLUMNS].duplicated().sum()),
        "periode_debut": str(df["date"].min().date()),
        "periode_fin": str(df["date"].max().date()),
        "pays": sorted(df["country"].unique().tolist()),
        "etats": sorted(df["statezip"].str[:2].unique().tolist()),
        "n_villes": int(df["city"].nunique()),
        "n_codes_postaux": int(df["statezip"].nunique()),
        "n_rues_distinctes": int(df["street"].nunique()),
        "prix_nuls": int((price == 0).sum()),
        "prix_negatifs": int((price < 0).sum()),
        "prix_non_entiers": int((price % 1 != 0).sum()),
        "prix_min_positif": float(pos["price"].min()),
        "prix_median": float(pos["price"].median()),
        "prix_max": float(price.max()),
        "prix_sqft_median": float(ppsf.median()),
        "prix_sqft_max": float(ppsf.max()),
        "chambres_0": int((df["bedrooms"] == 0).sum()),
        "salles_de_bain_0": int((df["bathrooms"] == 0).sum()),
        "renovation_avant_construction": int(
            ((df["yr_renovated"] > 0) & (df["yr_renovated"] < df["yr_built"])).sum()
        ),
        "yr_renovated_0": int((df["yr_renovated"] == 0).sum()),
        "construction_apres_vente": int((df["yr_built"] > df["date"].dt.year).sum()),
        "renovation_apres_vente": int((df["yr_renovated"] > df["date"].dt.year).sum()),
        "surfaces_incoherentes": int(
            (df["sqft_above"] + df["sqft_basement"] != df["sqft_living"]).sum()
        ),
        "lignes_adresse_repetee": int(dup_addr.sum()),
        "adresses_repetees": int(addr[dup_addr].nunique()),
        "bloc_final_lignes": int(block.sum()),
        "bloc_final_prix_nuls": int(((price == 0) & block).sum()),
        "bloc_final_prix_non_entiers": int(((price % 1 != 0) & block).sum()),
    }


def clean_data(df):
    """Applique les règles de nettoyage. Renvoie (données nettoyées, rapport des corrections)."""
    df = df.copy()
    df["address_id"] = address_key(df)
    df["suspect_block"] = df["row_id"] >= config.SUSPECT_BLOCK_START_ROW
    report = []

    # 1. Prix nul : la cible est absente, on ne peut pas apprendre dessus
    bad = df["price"] <= 0
    df = df[~bad]
    report.append(["Prix nul ou négatif", "price <= 0", bad.sum(), "lignes supprimées", len(df)])

    # 2. Prix incohérent avec la surface ET la localisation (erreur de saisie probable).
    #    On compare le prix au sqft à la médiane du code postal : une maison chère mais
    #    cohérente (grande, bien située) n'est pas concernée.
    price_per_sqft = df["price"] / df["sqft_living"]
    zip_median = price_per_sqft.groupby(df["statezip"]).transform("median")
    ratio = price_per_sqft / zip_median
    bad = (ratio > config.PPSF_RATIO_MAX) | (ratio < 1 / config.PPSF_RATIO_MAX)
    df = df[~bad]
    report.append(["Prix incohérent", f"prix/sqft > {config.PPSF_RATIO_MAX:g}x ou < "
                   f"1/{config.PPSF_RATIO_MAX:g} de la médiane du code postal",
                   bad.sum(), "lignes supprimées", len(df)])

    # 3. Rénovation avant la construction : impossible -> traitée comme « pas de rénovation »
    bad = (df["yr_renovated"] > 0) & (df["yr_renovated"] < df["yr_built"])
    df.loc[bad, "yr_renovated"] = 0
    report.append(["Rénovation avant construction", "0 < yr_renovated < yr_built",
                   bad.sum(), "yr_renovated mis à 0", len(df)])

    # 4. Zéro chambre ou zéro salle de bain pour une grande maison : valeur manquante
    for col in ["bedrooms", "bathrooms"]:
        bad = df[col] == 0
        df.loc[bad, col] = np.nan
        report.append([f"{col} = 0", f"{col} == 0", bad.sum(),
                       "mis à NaN (imputé par la médiane dans le pipeline)", len(df)])

    # 5. Bloc final du fichier : conservé, mais signalé pour mesurer l'erreur à part
    report.append(["Bloc final signalé", f"row_id >= {config.SUSPECT_BLOCK_START_ROW}",
                   df["suspect_block"].sum(), "conservées, erreur mesurée séparément", len(df)])

    report = pd.DataFrame(report, columns=["etape", "regle", "lignes_concernees", "action",
                                           "lignes_restantes"])
    return df.reset_index(drop=True), report
