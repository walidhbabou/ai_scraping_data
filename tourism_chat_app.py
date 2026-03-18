import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

DATASET_CANDIDATES = [
    Path(__file__).with_name("morocco_enriched.json"),
    Path(__file__).with_name("morocco_tourism_dataset_enriched.json"),
    Path(__file__).with_name("morocco.json"),
    Path(__file__).with_name("morocco_tourism_dataset.json"),
]
MAX_RESULTS_DEFAULT = 12

CATEGORY_ALIASES = {
    "restaurant": ["restaurant", "restaurants", "resto", "restos", "eat", "food", "diner", "fast food"],
    "cafe": ["cafe", "cafes", "coffee", "coffee shop", "cafeteria"],
    "attraction": ["attraction", "attractions", "visit", "visiter", "activity", "activities", "museum", "musee", "viewpoint"],
    "park": ["park", "parks", "parc", "parcs", "garden", "gardens", "nature"],
    "monument": ["monument", "monuments", "historic", "history", "memorial", "castle", "heritage"],
}

STOPWORDS = {
    "a", "ai", "and", "au", "aux", "avec", "best", "bon", "bons", "bonjour", "chercher", "cherche",
    "combien", "dans", "de", "des", "do", "est", "for", "hello", "hey", "i", "in", "je", "la", "le",
    "les", "me", "moi", "need", "on", "ou", "pour", "pres", "près", "que", "quoi", "recommend",
    "recommande", "recommandes", "salut", "show", "suis", "sur", "the", "to", "trouve", "un", "une",
    "visit", "visiter", "voir", "want", "where", "y", "lieu", "lieux", "touristique", "touristiques",
}

COUNT_PATTERNS = ["combien", "how many", "nombre", "count", "total"]


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset_path is None:
        return pd.DataFrame()

    with dataset_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    df = pd.DataFrame(payload.get("places", []))
    if df.empty:
        return df

    for column in ["name", "city", "normalized_category", "subtype", "address"]:
        if column not in df.columns:
            df[column] = ""

    df["name"] = df["name"].fillna("")
    df["city"] = df["city"].fillna("")
    df["normalized_category"] = df["normalized_category"].fillna("")
    df["subtype"] = df["subtype"].fillna("")
    df["address"] = df["address"].apply(lambda value: "" if pd.isna(value) else str(value))

    df["name_norm"] = df["name"].map(normalize_text)
    df["city_norm"] = df["city"].map(normalize_text)
    df["category_norm"] = df["normalized_category"].map(normalize_text)
    df["subtype_norm"] = df["subtype"].map(normalize_text)
    df["search_blob"] = (
        df["name_norm"] + " " + df["city_norm"] + " " + df["category_norm"] + " " + df["subtype_norm"]
    ).str.strip()

    return df


def build_city_lookup(df: pd.DataFrame) -> List[Tuple[str, str]]:
    cities = [city for city in df["city"].dropna().astype(str).unique().tolist() if city.strip()]
    pairs = [(normalize_text(city), city) for city in cities]
    return sorted(pairs, key=lambda item: len(item[0]), reverse=True)


def extract_category(query_norm: str) -> Optional[str]:
    for category, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            if normalize_text(alias) in query_norm:
                return category
    return None


def extract_city(query_norm: str, city_lookup: List[Tuple[str, str]]) -> Optional[str]:
    for city_norm, city_label in city_lookup:
        if city_norm and city_norm in query_norm:
            return city_label
    return None


def extract_limit(query: str) -> int:
    match = re.search(r"\b(\d{1,2})\b", query)
    if not match:
        return MAX_RESULTS_DEFAULT
    return max(1, min(int(match.group(1)), 25))


def extract_tokens(query_norm: str, detected_city: Optional[str], detected_category: Optional[str]) -> List[str]:
    tokens = []
    excluded = {normalize_text(detected_city), normalize_text(detected_category)}
    for aliases in CATEGORY_ALIASES.values():
        excluded.update(normalize_text(alias) for alias in aliases)

    for token in query_norm.split():
        if len(token) < 3 or token in STOPWORDS or token in excluded:
            continue
        tokens.append(token)
    return tokens


def format_place(row: pd.Series) -> str:
    location = row["city"] if row["city"] else "ville non renseignée"
    address = row["address"] if row["address"] else "adresse non renseignée"
    return (
        f"- {row['name']} | categorie: {row['normalized_category']} | ville: {location} | "
        f"lat: {row['latitude']}, lon: {row['longitude']} | adresse: {address}"
    )


def answer_query(query: str, df: pd.DataFrame, city_lookup: List[Tuple[str, str]]) -> Tuple[str, pd.DataFrame]:
    if df.empty:
        return "Le dataset est vide. Lance d'abord le pipeline pour générer les lieux.", df

    query_norm = normalize_text(query)
    if not query_norm:
        return "Pose une question comme: restaurants a Marrakech, cafes a Casablanca, ou combien de parcs a Rabat.", df.head(0)

    category = extract_category(query_norm)
    city = extract_city(query_norm, city_lookup)
    limit = extract_limit(query)
    wants_count = any(pattern in query_norm for pattern in COUNT_PATTERNS)

    filtered = df.copy()
    score = pd.Series(0, index=filtered.index, dtype="int64")

    if category:
        category_mask = filtered["normalized_category"].eq(category)
        filtered = filtered[category_mask].copy()
        score = score.loc[filtered.index] + 5

    if city:
        city_norm = normalize_text(city)
        city_mask = filtered["city_norm"].eq(city_norm)
        filtered = filtered[city_mask].copy()
        score = score.loc[filtered.index] + 5

    tokens = extract_tokens(query_norm, city, category)
    for token in tokens:
        if filtered.empty:
            break
        name_match = filtered["name_norm"].str.contains(token, regex=False)
        blob_match = filtered["search_blob"].str.contains(token, regex=False)
        score = score.loc[filtered.index] + name_match.astype(int) * 3 + blob_match.astype(int)

    if filtered.empty and (category or city):
        details = []
        if category:
            details.append(f"categorie '{category}'")
        if city:
            details.append(f"ville '{city}'")
        return f"Aucun lieu trouve pour {' et '.join(details)}.", filtered

    if not filtered.empty:
        filtered = filtered.assign(score=score.loc[filtered.index].fillna(0).astype(int))
        if tokens:
            filtered = filtered[filtered["score"] > 0].copy() if filtered["score"].gt(0).any() else filtered
        filtered = filtered.sort_values(
            by=["score", "city", "name"],
            ascending=[False, True, True],
            kind="stable",
        )

    if wants_count:
        if filtered.empty:
            return "Je n'ai trouve aucun resultat pour cette recherche.", filtered
        summary = f"J'ai trouve {len(filtered)} lieux"
        if category:
            summary += f" dans la categorie {category}"
        if city:
            summary += f" a {city}"
        summary += "."
        return summary, filtered.head(limit)

    if filtered.empty:
        sample = df.head(limit)
        response = "Je n'ai pas compris de filtre precis. Voici quelques lieux pour commencer:"
        return response, sample

    top_results = filtered.head(limit).copy()
    intro_parts = []
    if category:
        intro_parts.append(category)
    if city:
        intro_parts.append(f"a {city}")

    response = "Voici des suggestions"
    if intro_parts:
        response += " " + " ".join(intro_parts)
    response += ":\n" + "\n".join(format_place(row) for _, row in top_results.iterrows())
    return response, top_results


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtres")
    categories = ["Toutes"] + sorted(df["normalized_category"].dropna().unique().tolist())
    cities = ["Toutes"] + sorted(city for city in df["city"].dropna().unique().tolist() if str(city).strip())

    selected_category = st.sidebar.selectbox("Categorie", categories)
    selected_city = st.sidebar.selectbox("Ville", cities)
    max_rows = st.sidebar.slider("Nombre max de resultats", min_value=5, max_value=25, value=12, step=1)

    filtered = df
    if selected_category != "Toutes":
        filtered = filtered[filtered["normalized_category"] == selected_category]
    if selected_city != "Toutes":
        filtered = filtered[filtered["city"] == selected_city]

    st.sidebar.metric("Lieux disponibles", len(filtered))
    st.sidebar.metric("Villes detectees", int(df["city"].astype(str).str.strip().ne("").sum()))
    st.session_state.sidebar_limit = max_rows
    return filtered


def render_results(results: pd.DataFrame) -> None:
    if results.empty:
        st.info("Aucun resultat a afficher.")
        return

    display_columns = ["name", "normalized_category", "city", "latitude", "longitude", "address", "source"]
    table = results[display_columns].rename(
        columns={
            "name": "Nom",
            "normalized_category": "Categorie",
            "city": "Ville",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "address": "Adresse",
            "source": "Source",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    map_df = results[["latitude", "longitude"]].dropna().rename(columns={"latitude": "lat", "longitude": "lon"})
    if not map_df.empty:
        st.map(map_df)


def main() -> None:
    st.set_page_config(page_title="Morocco Tourism Chat", page_icon="🗺️", layout="wide")
    st.title("Morocco Tourism Dataset Tester")
    st.caption("Pose une question en francais ou en anglais pour tester automatiquement les lieux extraits depuis OpenStreetMap.")

    if not any(path.exists() for path in DATASET_CANDIDATES):
        st.error("Le fichier morocco_tourism_dataset.json est introuvable dans ce dossier.")
        return

    df = load_dataset()
    if df.empty:
        st.error("Le dataset charge est vide.")
        return

    city_lookup = build_city_lookup(df)
    filtered_df = render_sidebar(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total lieux", len(df))
    col2.metric("Categories", df["normalized_category"].nunique())
    col3.metric("Villes remplies", int(df["city"].astype(str).str.strip().ne("").sum()))

    st.markdown("Exemples: restaurants a Marrakech, cafes a Casablanca, combien de monuments a Rabat, parc a Agadir")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Je suis pret. Demande-moi des restaurants, cafes, attractions, parcs ou monuments au Maroc.",
            }
        ]
    if "last_results" not in st.session_state:
        st.session_state.last_results = filtered_df.head(st.session_state.get("sidebar_limit", MAX_RESULTS_DEFAULT))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Exemple: trouve-moi 10 restaurants a Marrakech")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response_text, results = answer_query(prompt, filtered_df, city_lookup)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.last_results = results

        with st.chat_message("assistant"):
            st.write(response_text)

    st.subheader("Resultats")
    render_results(st.session_state.last_results.head(st.session_state.get("sidebar_limit", MAX_RESULTS_DEFAULT)))


if __name__ == "__main__":
    main()