import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]
OUTPUT_FILE = "morocco_tourism_dataset.json"

# Tuned for large responses while staying stable on public Overpass endpoints.
OVERPASS_TIMEOUT_SECONDS = 240
REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 5
BACKOFF_SECONDS = 2
CATEGORY_DELAY_SECONDS = 2


CATEGORY_QUERIES: Dict[str, List[str]] = {
    "restaurant": [
        'nwr["amenity"="restaurant"](area.searchArea);',
        'nwr["amenity"="fast_food"](area.searchArea);',
    ],
    "cafe": [
        'nwr["amenity"="cafe"](area.searchArea);',
    ],
    "attraction": [
        'nwr["tourism"="attraction"](area.searchArea);',
        'nwr["tourism"="museum"](area.searchArea);',
        'nwr["tourism"="theme_park"](area.searchArea);',
        'nwr["tourism"="zoo"](area.searchArea);',
        'nwr["tourism"="aquarium"](area.searchArea);',
        'nwr["tourism"="gallery"](area.searchArea);',
        'nwr["tourism"="viewpoint"](area.searchArea);',
    ],
    "park": [
        'nwr["leisure"="park"](area.searchArea);',
        'nwr["boundary"="national_park"](area.searchArea);',
        'nwr["leisure"="nature_reserve"](area.searchArea);',
    ],
    "monument": [
        'nwr["historic"="monument"](area.searchArea);',
        'nwr["historic"="memorial"](area.searchArea);',
        'nwr["historic"="archaeological_site"](area.searchArea);',
        'nwr["historic"="castle"](area.searchArea);',
    ],
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_overpass_query(clauses: List[str]) -> str:
    clauses_block = "\n    ".join(clauses)
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT_SECONDS}];
area["ISO3166-1"="MA"]["admin_level"="2"]->.searchArea;
(
    {clauses_block}
);
out tags center qt;
""".strip()


def fetch_overpass_data(query: str, session: requests.Session) -> List[Dict[str, Any]]:
    total_endpoints = len(OVERPASS_URLS)
    for endpoint_index, endpoint in enumerate(OVERPASS_URLS, start=1):
        logging.info("Using Overpass endpoint %s/%s: %s", endpoint_index, total_endpoints, endpoint)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = session.post(
                    endpoint,
                    data=query,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                elements = payload.get("elements", [])
                return elements
            except requests.RequestException as exc:
                wait_time = BACKOFF_SECONDS ** attempt
                logging.warning(
                    "Overpass request failed on endpoint %s attempt %s/%s: %s. Retrying in %ss...",
                    endpoint,
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait_time,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(wait_time)
        logging.warning("Switching to next Overpass endpoint after repeated failures: %s", endpoint)

    logging.error("All Overpass endpoints failed for this query. Returning empty result set.")
    return []


def extract_lat_lon(element: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    if "lat" in element and "lon" in element:
        return element.get("lat"), element.get("lon")

    center = element.get("center", {})
    return center.get("lat"), center.get("lon")


def extract_city(tags: Dict[str, Any]) -> Optional[str]:
    city_keys = [
        "addr:city",
        "is_in:city",
        "addr:town",
        "is_in:town",
        "addr:municipality",
        "addr:county",
    ]
    for key in city_keys:
        value = tags.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def determine_subtype(tags: Dict[str, Any]) -> Optional[str]:
    for key in ("amenity", "tourism", "leisure", "historic", "boundary"):
        value = tags.get(key)
        if isinstance(value, str) and value.strip():
            return f"{key}:{value.strip()}"
    return None


def parse_elements(elements: List[Dict[str, Any]], normalized_category: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for element in elements:
        tags = element.get("tags", {}) or {}
        osm_id = element.get("id")
        osm_type = element.get("type")

        if osm_id is None or not osm_type:
            continue

        lat, lon = extract_lat_lon(element)
        name = tags.get("name")
        city = extract_city(tags)
        subtype = determine_subtype(tags)

        record = {
            "osm_uid": f"{osm_type}/{osm_id}",
            "osm_id": osm_id,
            "osm_type": osm_type,
            "name": name.strip() if isinstance(name, str) else None,
            "normalized_category": normalized_category,
            "subtype": subtype,
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "lat_lon": f"{lat},{lon}" if lat is not None and lon is not None else None,
            "address": tags.get("addr:full") or None,
            "source": "OpenStreetMap",
        }
        records.append(record)

    return records


def process_dataset(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "osm_uid",
                "osm_id",
                "osm_type",
                "name",
                "normalized_category",
                "subtype",
                "city",
                "latitude",
                "longitude",
                "address",
                "source",
            ]
        )

    df = pd.DataFrame.from_records(records)

    logging.info("Initial records collected: %s", len(df))

    # Keep only records that can be mapped on a map and are useful for chatbot retrieval.
    df = df.dropna(subset=["latitude", "longitude", "name"]).copy()

    # Normalize text fields for robust deduplication.
    df["name"] = df["name"].astype(str).str.strip()
    df["city"] = df["city"].fillna("").astype(str).str.strip()
    df["normalized_category"] = df["normalized_category"].astype(str).str.strip().str.lower()
    df["lat_lon"] = df["latitude"].astype(str) + "," + df["longitude"].astype(str)

    before_osm_dedup = len(df)
    df = df.drop_duplicates(subset=["osm_uid"], keep="first")
    logging.info("Removed %s duplicates by osm_uid", before_osm_dedup - len(df))

    # Secondary dedup in case same place appears through multiple tag paths.
    df["name_key"] = df["name"].str.lower()
    df["city_key"] = df["city"].str.lower()
    df["lat_round"] = pd.to_numeric(df["latitude"], errors="coerce").round(5)
    df["lon_round"] = pd.to_numeric(df["longitude"], errors="coerce").round(5)

    before_soft_dedup = len(df)
    df = df.drop_duplicates(
        subset=["name_key", "city_key", "normalized_category", "lat_round", "lon_round"],
        keep="first",
    )
    logging.info("Removed %s near-duplicates by name/category/coordinates", before_soft_dedup - len(df))

    df = df.drop(columns=["name_key", "city_key", "lat_round", "lon_round"])
    df = df.reset_index(drop=True)

    logging.info("Final records after processing: %s", len(df))
    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    clean_df = df.where(pd.notna(df), None)
    payload = {
        "metadata": {
            "country": "Morocco",
            "source": "OpenStreetMap Overpass API",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "record_count": int(len(clean_df)),
            "categories": sorted(clean_df["normalized_category"].dropna().unique().tolist()),
            "columns": clean_df.columns.tolist(),
        },
        "places": clean_df.to_dict(orient="records"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)

    logging.info("Dataset saved to %s", output_path)


def main() -> None:
    setup_logging()
    logging.info("Starting Morocco tourism pipeline...")

    all_records: List[Dict[str, Any]] = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "MoroccoTourismPipeline/1.0 (AI chatbot dataset builder)",
                "Accept": "application/json",
            }
        )

        total_categories = len(CATEGORY_QUERIES)
        for idx, (category, clauses) in enumerate(CATEGORY_QUERIES.items(), start=1):
            logging.info("[%s/%s] Fetching category: %s", idx, total_categories, category)
            query = build_overpass_query(clauses)
            elements = fetch_overpass_data(query, session)
            logging.info("Fetched %s raw OSM elements for %s", len(elements), category)

            category_records = parse_elements(elements, normalized_category=category)
            logging.info("Parsed %s records for %s", len(category_records), category)
            all_records.extend(category_records)

            # Small delay reduces rate-limit pressure on public Overpass instances.
            if idx < total_categories:
                time.sleep(CATEGORY_DELAY_SECONDS)

    logging.info("Total parsed records before processing: %s", len(all_records))
    df = process_dataset(all_records)

    save_dataset(df, OUTPUT_FILE)
    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
