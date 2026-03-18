import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

INPUT_DATASET_CANDIDATES = [
    Path(__file__).with_name("morocco.json"),
    Path(__file__).with_name("morocco_tourism_dataset.json"),
]
CACHE_FILE = Path(__file__).with_name("google_geocode_cache.json")
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
BACKOFF_SECONDS = 2
SAVE_CACHE_EVERY = 50
LOG_PROGRESS_EVERY = 25
REQUEST_DELAY_SECONDS = 1.1


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)


def clean_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    clean_df = df.where(pd.notna(df), None)
    return clean_df.to_dict(orient="records")


def load_cache() -> Dict[str, Dict[str, Any]]:
    if not CACHE_FILE.exists():
        return {}
    payload = load_json(CACHE_FILE)
    if not isinstance(payload, dict):
        return {}
    return payload


def save_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    save_json(CACHE_FILE, cache)


def is_cacheable_enrichment(enrichment: Dict[str, Any]) -> bool:
    return enrichment.get("enrichment_status") in {"OK", "ZERO_RESULTS"}


def purge_invalid_cache_entries(cache: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    cleaned_cache: Dict[str, Dict[str, Any]] = {}
    removed_entries = 0

    for cache_key, enrichment in cache.items():
        if isinstance(enrichment, dict) and is_cacheable_enrichment(enrichment):
            cleaned_cache[cache_key] = enrichment
        else:
            removed_entries += 1

    if removed_entries:
        logging.info("Removed %s non-reusable cache entries", removed_entries)

    return cleaned_cache


def resolve_dataset_paths() -> Tuple[Path, Path]:
    input_path = next((path for path in INPUT_DATASET_CANDIDATES if path.exists()), None)
    if input_path is None:
        candidates = ", ".join(path.name for path in INPUT_DATASET_CANDIDATES)
        raise FileNotFoundError(f"Input dataset not found. Tried: {candidates}")

    if input_path.name == "morocco.json":
        output_path = Path(__file__).with_name("morocco_enriched.json")
    else:
        output_path = Path(__file__).with_name("morocco_tourism_dataset_enriched.json")

    return input_path, output_path


def build_cache_key(latitude: Any, longitude: Any) -> Optional[str]:
    try:
        lat_value = round(float(latitude), 6)
        lon_value = round(float(longitude), 6)
    except (TypeError, ValueError):
        return None
    return f"{lat_value},{lon_value}"


def extract_city_from_components(components: List[Dict[str, Any]]) -> Optional[str]:
    priority_types = [
        "locality",
        "postal_town",
        "administrative_area_level_3",
        "administrative_area_level_2",
        "administrative_area_level_1",
    ]
    for component_type in priority_types:
        for component in components:
            types = component.get("types", [])
            if component_type in types:
                name = component.get("long_name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
    return None


def fetch_reverse_geocode(
    session: requests.Session,
    latitude: Any,
    longitude: Any,
) -> Dict[str, Any]:
    params = {
        "lat": latitude,
        "lon": longitude,
        "language": "fr",
        "format": "jsonv2",
        "addressdetails": 1,
        "zoom": 18,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(NOMINATIM_REVERSE_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                details = payload.get("error")
                logging.warning(
                    "Nominatim returned error for %s,%s: %s",
                    latitude,
                    longitude,
                    details,
                )
                return {"enrichment_status": "ZERO_RESULTS"}

            address = payload.get("address", {}) if isinstance(payload, dict) else {}
            if not payload or not address:
                return {"enrichment_status": "ZERO_RESULTS"}

            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality")
                or address.get("county")
            )

            return {
                "enrichment_status": "OK",
                "osm_place_id": payload.get("place_id"),
                "osm_display_name": payload.get("display_name"),
                "osm_class": payload.get("class"),
                "osm_type": payload.get("type"),
                "osm_city": city,
            }
        except requests.RequestException as exc:
            wait_time = BACKOFF_SECONDS ** attempt
            logging.warning(
                "Nominatim request failed on attempt %s/%s for %s,%s: %s. Retrying in %ss...",
                attempt,
                MAX_RETRIES,
                latitude,
                longitude,
                exc,
                wait_time,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait_time)

    return {"enrichment_status": "FAILED"}


def enrich_dataset() -> None:
    input_dataset, output_dataset = resolve_dataset_paths()

    payload = load_json(input_dataset)
    df = pd.DataFrame(payload.get("places", []))
    if df.empty:
        raise RuntimeError("Input dataset has no places to enrich.")

    for column in ["city", "address", "latitude", "longitude"]:
        if column not in df.columns:
            df[column] = None

    df["city"] = df["city"].fillna("")
    df["address"] = df["address"].fillna("")
    df["lat_lon"] = (
        df["latitude"].astype(str).str.strip() + "," + df["longitude"].astype(str).str.strip()
    )
    df["osm_place_id"] = df.get("osm_place_id")
    df["osm_display_name"] = df.get("osm_display_name")
    df["osm_class"] = df.get("osm_class")
    df["osm_type"] = df.get("osm_type")
    df["enrichment_status"] = df.get("enrichment_status")
    df["enrichment_source"] = df.get("enrichment_source")

    candidate_mask = (
        df["latitude"].notna()
        & df["longitude"].notna()
        & (
            df["city"].astype(str).str.strip().eq("")
            | df["address"].astype(str).str.strip().eq("")
        )
    )

    candidate_indices = df.index[candidate_mask].tolist()
    logging.info("Input dataset: %s", input_dataset.name)
    logging.info("Rows eligible for OSM enrichment: %s", len(candidate_indices))

    cache = purge_invalid_cache_entries(load_cache())
    processed_count = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": "MoroccoTourismOSMEnrichment/1.0 (contact: local-script)"})

        for row_number, idx in enumerate(candidate_indices, start=1):
            row = df.loc[idx]
            cache_key = build_cache_key(row["latitude"], row["longitude"])
            if not cache_key:
                continue

            enrichment = cache.get(cache_key)
            if enrichment is None:
                try:
                    enrichment = fetch_reverse_geocode(session, row["latitude"], row["longitude"])
                except Exception as exc:  # Defensive fallback to keep long jobs running.
                    logging.exception(
                        "Unexpected error during reverse geocode for %s,%s: %s",
                        row["latitude"],
                        row["longitude"],
                        exc,
                    )
                    enrichment = {"enrichment_status": "FAILED"}
                if is_cacheable_enrichment(enrichment):
                    cache[cache_key] = enrichment
                processed_count += 1

            osm_city = enrichment.get("osm_city")
            display_name = enrichment.get("osm_display_name")

            if not str(row["city"]).strip() and osm_city:
                df.at[idx, "city"] = osm_city
            if not str(row["address"]).strip() and display_name:
                df.at[idx, "address"] = display_name

            df.at[idx, "osm_place_id"] = enrichment.get("osm_place_id")
            df.at[idx, "osm_display_name"] = enrichment.get("osm_display_name")
            df.at[idx, "osm_class"] = enrichment.get("osm_class")
            df.at[idx, "osm_type"] = enrichment.get("osm_type")
            df.at[idx, "enrichment_status"] = enrichment.get("enrichment_status")
            df.at[idx, "enrichment_source"] = "OSM Nominatim"

            if row_number % LOG_PROGRESS_EVERY == 0:
                logging.info("Processed %s/%s candidate rows", row_number, len(candidate_indices))
            if processed_count and processed_count % SAVE_CACHE_EVERY == 0:
                save_cache(cache)
                logging.info("Saved cache after %s API calls", processed_count)

            time.sleep(REQUEST_DELAY_SECONDS)

    save_cache(cache)

    output_payload = {
        "metadata": {
            **payload.get("metadata", {}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "record_count": int(len(df)),
            "input_file": input_dataset.name,
            "output_file": output_dataset.name,
            "enriched_with_osm_nominatim": True,
            "osm_enrichment": {
                "api": "OSM Nominatim Reverse",
                "candidate_rows": int(len(candidate_indices)),
                "api_calls": int(processed_count),
                "cache_file": CACHE_FILE.name,
            },
        },
        "places": clean_records(df),
    }
    save_json(output_dataset, output_payload)
    logging.info("Enriched dataset saved to %s", output_dataset.name)


def main() -> None:
    setup_logging()
    logging.info("Starting OSM Nominatim enrichment (no API key)...")
    enrich_dataset()
    logging.info("OSM Nominatim enrichment completed successfully.")


if __name__ == "__main__":
    main()