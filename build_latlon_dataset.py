import json
import math
from pathlib import Path

INPUT_PATH = Path("morocco_tourism_dataset.json")
OUTPUT_PATH = Path("morocco_with_latlon.json")


def sanitize_json(value):
    if isinstance(value, dict):
        return {k: sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    payload = sanitize_json(payload)
    places = payload.get("places", [])
    for record in places:
        lat = record.get("latitude")
        lon = record.get("longitude")
        record["lat_lon"] = f"{lat},{lon}" if lat is not None and lon is not None else None

    metadata = payload.setdefault("metadata", {})
    columns = metadata.get("columns", [])
    if "lat_lon" not in columns:
        metadata["columns"] = columns + ["lat_lon"]

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
