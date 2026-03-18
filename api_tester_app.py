import json
import math
import re
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


def call_health(base_url: str) -> Dict[str, Any]:
    response = requests.get(f"{base_url}/health", timeout=20)
    response.raise_for_status()
    return response.json()


def call_ask(base_url: str, question: str, top_k: int) -> Dict[str, Any]:
    payload = {"question": question, "top_k": top_k}
    response = requests.post(f"{base_url}/ask", json=payload, timeout=45)
    response.raise_for_status()
    return response.json()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def is_near_me_query(question: str) -> bool:
    query = normalize_text(question)
    patterns = [
        r"\bproche(s)? de moi\b",
        r"\bpres de moi\b",
        r"\bprès de moi\b",
        r"\bautour de moi\b",
        r"\bnear me\b",
        r"\baround me\b",
    ]
    return any(re.search(pattern, query) for pattern in patterns)


def asks_for_cafe(question: str) -> bool:
    query = normalize_text(question)
    return any(token in query for token in ["cafe", "cafes", "coffee"])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sort_results_by_distance(
    table_df: pd.DataFrame,
    question: str,
    user_lat: float,
    user_lon: float,
) -> pd.DataFrame:
    if table_df.empty:
        return table_df

    near_me = is_near_me_query(question)
    if not near_me:
        return table_df

    df = table_df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    if df.empty:
        return df

    if asks_for_cafe(question):
        cafe_mask = df["category"].fillna("").astype(str).str.contains("cafe", case=False, regex=False)
        if cafe_mask.any():
            df = df[cafe_mask].copy()

    df["distance_km"] = df.apply(
        lambda row: haversine_km(user_lat, user_lon, float(row["latitude"]), float(row["longitude"])),
        axis=1,
    )
    df = df.sort_values(by=["distance_km", "score"], ascending=[True, False], kind="stable")
    return df


def normalize_matches(matches: List[Dict[str, Any]]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame()

    df = pd.DataFrame(matches)
    for col in ["name", "category", "city", "lat_lon", "address", "score", "latitude", "longitude"]:
        if col not in df.columns:
            df[col] = None

    return df[["name", "category", "city", "lat_lon", "latitude", "longitude", "address", "score"]]


def main() -> None:
    st.set_page_config(page_title="API Tester - Morocco Guide", page_icon="🧭", layout="wide")
    st.title("Interface de test API - Guide intelligent")
    st.caption("Teste facilement ton API /health et /ask avec affichage des resultats et des localisations.")

    with st.sidebar:
        st.header("Configuration API")
        base_url = st.text_input("Base URL", value="http://localhost:8000").strip().rstrip("/")
        top_k = st.slider("Top K", min_value=1, max_value=30, value=8, step=1)

        st.divider()
        st.subheader("Ma position (optionnel)")
        user_lat = st.number_input("Latitude", value=33.5731, format="%.6f")
        user_lon = st.number_input("Longitude", value=-7.5898, format="%.6f")
        st.caption("Exemple Casablanca: 33.5731, -7.5898")

        if st.button("Tester Health"):
            try:
                health = call_health(base_url)
                st.success("API accessible")
                st.json(health)
            except Exception as exc:
                st.error(f"Erreur health: {exc}")

    st.markdown(
        "Exemples: restaurants a Marrakech, cafes a Casablanca, cafes proches de moi, combien de monuments a Rabat"
    )

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    question = st.text_input("Question", value="restaurants a Marrakech")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        ask_clicked = st.button("Envoyer Question", type="primary")
    with col_b:
        clear_clicked = st.button("Effacer Historique")

    if clear_clicked:
        st.session_state.history = []
        st.session_state.last_result = None
        st.info("Historique efface.")

    if ask_clicked:
        if not question.strip():
            st.warning("Ecris une question avant d'envoyer.")
        else:
            try:
                result = call_ask(base_url, question.strip(), top_k)
                st.session_state.last_result = result
                st.session_state.history.append({"question": question.strip(), "result": result})
            except Exception as exc:
                st.error(f"Erreur /ask: {exc}")

    if st.session_state.last_result:
        result = st.session_state.last_result

        st.subheader("Reponse")
        st.write(result.get("answer", ""))

        meta_col1, meta_col2, meta_col3 = st.columns(3)
        meta_col1.metric("Count", int(result.get("count", 0)))
        meta_col2.metric("Category", (result.get("filters") or {}).get("category") or "-")
        meta_col3.metric("City", (result.get("filters") or {}).get("city") or "-")

        st.subheader("Resultats")
        matches = result.get("matches", [])
        table_df = normalize_matches(matches)
        table_df = sort_results_by_distance(
            table_df=table_df,
            question=question,
            user_lat=float(user_lat),
            user_lon=float(user_lon),
        )

        if table_df.empty:
            st.info("Aucun resultat retourne.")
        else:
            if is_near_me_query(question):
                st.success(
                    f"Resultats tries par distance depuis ({user_lat:.5f}, {user_lon:.5f})."
                )
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            map_df = table_df[["latitude", "longitude"]].dropna().rename(columns={"latitude": "lat", "longitude": "lon"})
            if not map_df.empty:
                st.map(map_df)

        with st.expander("JSON brut"):
            st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")

    st.subheader("Historique")
    if not st.session_state.history:
        st.write("Aucune question envoyee pour le moment.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history[-10:]), start=1):
            st.markdown(f"{idx}. {item['question']}")


if __name__ == "__main__":
    main()