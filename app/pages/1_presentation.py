import streamlit as st

from app.common import read_report, require_artifacts, scope_banner, usd

model, meta, _ = require_artifacts()
audit = read_report("data_audit.json")
cleaning = read_report("cleaning_report.csv")

st.title("Estimer le prix d'une maison et comprendre l'estimation")
scope_banner(meta)

st.markdown(
    """
**Problème.** À partir des caractéristiques d'une maison (surface, nombre de pièces, état,
localisation…), estimer son prix de vente, puis **expliquer** l'estimation : quelles
caractéristiques font monter ou baisser le prix prédit, et de combien.

**Démarche.**
1. Audit et nettoyage documenté des données (fichier original conservé intact).
2. Comparaison de 4 modèles par **validation croisée groupée par adresse** sur le jeu
   d'entraînement ; un **jeu de test indépendant** (20 %) n'est utilisé qu'à la fin.
3. Explication des prédictions du modèle retenu avec **SHAP**.
"""
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Modèle retenu", meta["selected_label"].split(" (")[0])
c2.metric("Maisons (entraînement)", f"{meta['n_train']:,}".replace(",", " "))
c3.metric("Maisons (test)", f"{meta['n_test']:,}".replace(",", " "))
c4.metric("Période", f"{meta['period'][0][5:]} → {meta['period'][1][5:]} (2014)")

st.subheader("Les données")
if audit:
    st.markdown(
        f"""
- **{audit['n_lignes']} ventes**, {audit['n_colonnes']} colonnes, aucune valeur manquante
  déclarée, aucun doublon exact.
- **Période** : du {audit['periode_debut']} au {audit['periode_fin']} (environ 10 semaines).
- **Zone** : {audit['n_villes']} villes et {audit['n_codes_postaux']} codes postaux, tous dans
  l'État de Washington ({', '.join(audit['etats'])}), pays = {', '.join(audit['pays'])}.
- **Prix** : médiane {usd(audit['prix_median'])}, maximum {usd(audit['prix_max'])}
  ({audit['prix_nuls']} prix nuls).
- **Unités** : la devise et les unités de surface ne sont pas documentées dans le fichier ;
  les noms de colonnes (`sqft_*`) et les ordres de grandeur sont cohérents avec des pieds
  carrés et des dollars US.
"""
    )
st.warning(
    "**Provenance non vérifiable** : le dépôt ne documente ni la source exacte ni la licence "
    "de `data.csv`. Sa structure ressemble à des jeux publics de ventes de la région de "
    "Seattle diffusés sur Kaggle, mais cela n'a pas pu être confirmé ici.",
    icon=":material/help:",
)

st.subheader("Nettoyage appliqué")
if cleaning is not None:
    st.dataframe(cleaning.rename(columns={
        "etape": "Étape", "regle": "Règle", "lignes_concernees": "Lignes concernées",
        "action": "Action", "lignes_restantes": "Lignes restantes"}),
        hide_index=True, width="stretch")
st.caption(
    "Les maisons chères mais cohérentes (ex. 7,06 M$ pour 10 040 sqft en bord de lac à "
    "Bellevue) sont conservées : seuls les prix incohérents avec la surface ET la "
    "localisation sont retirés (ex. 26,59 M$ pour 1 180 sqft à Kent)."
)

st.subheader("Pourquoi le notebook initial obtenait R² = 0,03")
st.markdown(
    """
Le notebook d'origine (régression linéaire, 12 variables numériques) affichait un R² de
**0,0323**. En reproduisant exactement son découpage, on constate qu'**une seule vente**
(26,59 M$ pour 1 180 sqft) tombe dans le jeu de test et représente **77 % de l'erreur
quadratique**. Sans les deux prix aberrants, le même modèle atteint R² ≈ 0,45, et ≈ 0,51
sans les prix nuls. Le problème venait donc d'abord des données et d'une métrique
quadratique sensible aux valeurs extrêmes, puis de l'absence de la localisation.
"""
)

st.subheader("Limites")
st.markdown(
    """
- Données **historiques** (2014), **géographiquement limitées** (région de Seattle).
- Pas d'informations sur la qualité intérieure, l'école de secteur, la distance aux
  transports… : une part de la variance reste inexpliquée.
- Les explications SHAP décrivent **le comportement du modèle**, pas des relations
  causales ; elles se partagent entre variables corrélées (surface, chambres, salles de bain).
"""
)
