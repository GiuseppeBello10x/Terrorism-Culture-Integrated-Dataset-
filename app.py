import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(layout="wide", page_title="Terrorism & Culture")

# File
DATA_PATH = "DATAV2.csv"
LOGO_PATH = ""

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "num_attacks" in df.columns:
        df["num_attacks"] = pd.to_numeric(df["num_attacks"], errors="coerce").fillna(0).astype(int)
    # eliminare colonne di testo
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df

def get_dimension_columns_by_position(df: pd.DataFrame):
    'Posizione di default delle colonne contenenti le dimensioni culturali'
    cols = list(df.columns)
    dims = cols[6:15] 
    meta = {"gname", "country_txt", "num_attacks", "iso_alpha", "Country", "Country Name", "Country Cluster"}
    dims = [c for c in dims if c not in meta]
    return dims

@st.cache_data
def pre_aggregate(df, dim_cols):
    # attacchi aggregati per paese
    per_country = (
        df.groupby(["iso_alpha", "country_txt", "Country Cluster"], dropna=False, as_index=False)
        .agg({"num_attacks": "sum"})
    )
    # aggregati per gruppi (Top10)
    per_group = (
        df.groupby("gname", dropna=False, as_index=False)
        .agg({"num_attacks": "sum"})
        .sort_values("num_attacks", ascending=False)
    )
    # calcolo medie dimensioni per paese e cluster
    dims_by_country = pd.DataFrame()
    dims_by_cluster = pd.DataFrame()
    if dim_cols:
        df_num = df.copy()
        for c in dim_cols:
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")
        dims_by_country = (
            df_num.groupby(["country_txt", "iso_alpha", "Country Cluster"], dropna=False)[dim_cols]
            .mean()
            .reset_index()
        )
        dims_by_cluster = (
            dims_by_country.groupby("Country Cluster", dropna=False)[dim_cols]
            .mean()
            .reset_index()
        )
    return per_country, per_group, dims_by_country, dims_by_cluster

# caricamento dataset
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"File {DATA_PATH} non trovato. Metti DATAV2.csv nella stessa cartella di app.py o modifica DATA_PATH.")
    st.stop()

# salvo le dimensioni 
present_dims = get_dimension_columns_by_position(df)

# colonne usate
#if present_dims:
#    st.info(f"Colonne dimensioni prese per posizione (usate così come sono): {', '.join(present_dims)}")
#else:
#    st.warning("Non sono state individuate colonne di dimensione tramite la selezione per posizione. Heatmap e radar saranno disabilitati.")


# dati pre-aggregati
per_country, per_group, dims_by_country, dims_by_cluster = pre_aggregate(df, present_dims)

# Titolo e logo
header_col, logo_col = st.columns([9, 1])
header_col.title("Terrorism & Culture — Dashboard")
header_col.markdown("Con questa dashboard si intende mostrare in modo rapido ed efficace le principali informazioni ottenibili dal Terrorism-Culture Integrated Dataset")
if os.path.exists(LOGO_PATH):
    logo_col.image(LOGO_PATH, use_container_width=True)
else:
    logo_col.write("")
st.markdown("---")


# Globe map 
st.subheader("Mappa: attacchi aggregati per paese")
if per_country.empty:
    st.info("Nessun dato disponibile per la mappa.")
else:
    use_iso = per_country["iso_alpha"].notna().any() and per_country["iso_alpha"].str.len().gt(0).any()
    if use_iso:
        fig_map = px.choropleth(
            per_country,
            locations="iso_alpha",
            color="num_attacks",
            hover_name="country_txt",
            color_continuous_scale="Reds",
            title="Attacchi aggregati per paese",
            labels={"num_attacks": "Numero attacchi"},
            projection="natural earth",
        )
    else:
        fig_map = px.choropleth(
            per_country,
            locations="country_txt",
            locationmode="country names",
            color="num_attacks",
            hover_name="country_txt",
            color_continuous_scale="Reds",
            title="Attacchi aggregati per paese — using country names",
            labels={"num_attacks": "Numero attacchi"},
            projection="natural earth",
        )
    fig_map.update_layout(margin={"r":0,"t":35,"l":0,"b":0}, height=560)
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# Top10 peasi e gruppi
st.subheader("Paesi e Gruppi con più attacchi")
left, right = st.columns(2)

with left:
    st.markdown("### Top 10 Paesi più colpiti")
    top_countries = per_country.sort_values("num_attacks", ascending=False).head(10)
    if top_countries.empty:
        st.info("Nessun paese da mostrare.")
    else:
        fig_top_c = px.bar(top_countries, x="country_txt", y="num_attacks", color="Country Cluster",
                           title="Paesi per maggior numero di attacchi subiti",
                           labels={"num_attacks": "Attacchi", "country_txt": "Paese"})
        fig_top_c.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig_top_c, use_container_width=True)

with right:
    st.markdown("### Top 10 Gruppi con più attacchi")
    top_groups = per_group.sort_values("num_attacks", ascending=False).head(10)
    if top_groups.empty:
        st.info("Nessun gruppo da mostrare.")
    else:
        fig_top_g = px.bar(top_groups, x="gname", y="num_attacks",
                           title="Gruppi per maggior numero di attacchi",
                           labels={"num_attacks": "Attacchi", "gname": "Gruppo"})
        fig_top_g.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig_top_g, use_container_width=True)

st.markdown("---")

#  Heatmap: medie delle dimensioni per cluster (values inside) ====
st.subheader("Heatmap: medie delle dimensioni per cluster culturale")
st.markdown("Con questa heatmap si possono confrontare i profili medi delle dimensioni culturali per i diversi cluster di paesi.")
if not present_dims:
    st.info("Non sono state selezionate colonne delle dimensioni. Heatmap non disponibile.")
else:
    if dims_by_cluster.empty:
        st.info("Dati cluster mancanti o aggregazione non disponibile.")
    else:
        # rows = dimensioni, cols = clusters
        hm = dims_by_cluster.set_index("Country Cluster")[present_dims]
        
        fig_hm = px.imshow(
            hm.T.fillna(np.nan),
            x=hm.index.tolist(),
            y=hm.columns.tolist(),
            labels=dict(x="Cluster", y="Dimensione culturale", color="Valore medio"),
            color_continuous_scale="RdBu",
            text_auto=".2f",
            aspect="auto"
        )
        # stile 
        fig_hm.update_traces(textfont=dict(size=11, color="black"))
        fig_hm.update_layout(height=520, margin=dict(t=60, b=40, l=200, r=40))
        fig_hm.update_yaxes(tickangle=0)
        st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")

#  Radar chart 
st.subheader("Radar chart: profilo culturale di un singolo paese")
st.markdown("Con questa visualizzazione si possono controllare i profili delle dimensioni culturali per i diversi paesi presenti nel dataset integrato.")
if not present_dims or dims_by_country.empty:
    st.info("Radar non disponibile: assicurati che il CSV contenga colonne delle dimensioni e righe per paese.")
else:
    country_options = sorted(dims_by_country["country_txt"].dropna().unique().tolist())
    sel_country = st.selectbox("Scegli un paese per il radar (singolo)", options=[""] + country_options, index=0)
    if not sel_country:
        st.info("Seleziona un paese per visualizzare il radar.")
    else:
        row = dims_by_country[dims_by_country["country_txt"] == sel_country]
        if row.empty:
            st.warning("Profilo paese non disponibile nei dati aggregati.")
        else:
            values = row.iloc[0][present_dims].astype(float).tolist()
            values = [0.0 if pd.isna(v) else float(v) for v in values]
            categories = present_dims.copy()
            values_loop = values + [values[0]]
            categories_loop = categories + [categories[0]]
            fig_radar = go.Figure(
                data=[
                    go.Scatterpolar(r=values_loop, theta=categories_loop, fill='toself', name=sel_country,
                                    marker=dict(color="#1f77b4"))
                ],
                layout=go.Layout(
                    title=f"Profilo culturale: {sel_country}",
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=False
                )
            )
            fig_radar.update_layout(margin=dict(t=60, b=40, l=40, r=40), height=520)
            st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
