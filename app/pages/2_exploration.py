import altair as alt
import numpy as np
import streamlit as st

from app.common import require_data, scope_banner

df, split = require_data()
df = df.assign(jeu=split.values, prix_par_sqft=df["price"] / df["sqft_living"])

st.title("Exploration des données nettoyées")
scope_banner()

cities = sorted(df["city"].unique())
sel = st.multiselect("Filtrer par ville (vide = toutes)", cities)
d = df[df["city"].isin(sel)] if sel else df
st.caption(f"{len(d)} ventes affichées sur {len(df)}.")

c1, c2, c3 = st.columns(3)
c1.metric("Prix médian", f"{d['price'].median():,.0f} $".replace(",", " "))
c2.metric("Surface habitable médiane", f"{d['sqft_living'].median():,.0f} sqft".replace(",", " "))
c3.metric("Prix médian au sqft", f"{d['prix_par_sqft'].median():,.0f} $".replace(",", " "))

left, right = st.columns(2)
with left:
    st.subheader("Distribution des prix")
    hist = d.assign(log_prix=np.log10(d["price"]))
    st.altair_chart(
        alt.Chart(hist).mark_bar().encode(
            alt.X("log_prix:Q", bin=alt.Bin(maxbins=50), title="log10(prix en USD)"),
            alt.Y("count()", title="Nombre de ventes")),
        width="stretch")
    st.caption("Distribution très asymétrique : d'où l'apprentissage sur log1p(prix). "
               "5 sur l'axe = 100 000 $, 6 = 1 000 000 $.")
with right:
    st.subheader("Prix et surface habitable")
    st.altair_chart(
        alt.Chart(d).mark_circle(size=12, opacity=0.4).encode(
            alt.X("sqft_living:Q", scale=alt.Scale(type="log"), title="Surface habitable (sqft)"),
            alt.Y("price:Q", scale=alt.Scale(type="log"), title="Prix (USD)"),
            tooltip=["city", "statezip", "price", "sqft_living", "bedrooms"]),
        width="stretch")
    st.caption("En échelles log-log, la relation est proche d'une droite : c'est pourquoi "
               "le modèle linéaire utilise le log des surfaces.")

st.subheader("Prix médian au sqft par ville")
by_city = (d.groupby("city").agg(prix_sqft_median=("prix_par_sqft", "median"),
                                 n=("price", "size")).reset_index())
by_city = by_city[by_city["n"] >= 5]
st.altair_chart(
    alt.Chart(by_city).mark_bar().encode(
        alt.X("prix_sqft_median:Q", title="Prix médian au sqft (USD)"),
        alt.Y("city:N", sort="-x", title=None),
        tooltip=["city", "n", alt.Tooltip("prix_sqft_median:Q", format=",.0f")]),
    width="stretch")
st.caption("Villes comptant au moins 5 ventes. La localisation fait varier le prix au sqft "
           "du simple au triple : elle est indispensable au modèle.")

st.subheader("Prix selon l'état et la vue")
c1, c2 = st.columns(2)
for col, var, title in [(c1, "condition", "État général (1-5)"), (c2, "view", "Vue (0-4)")]:
    with col:
        st.altair_chart(
            alt.Chart(d).mark_boxplot(extent="min-max").encode(
                alt.X(f"{var}:O", title=title),
                alt.Y("price:Q", scale=alt.Scale(type="log"), title="Prix (USD)")),
            width="stretch")

with st.expander("Voir les données nettoyées"):
    st.dataframe(d.drop(columns=["address_id"]), width="stretch", hide_index=True)
