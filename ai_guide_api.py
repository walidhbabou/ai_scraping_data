import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATASET_CANDIDATES = [
    Path("morocco_with_latlon.json"),
    Path("morocco_enriched.json"),
    Path("morocco_tourism_dataset_enriched.json"),
    Path("morocco.json"),
    Path("morocco_tourism_dataset.json"),
]

CATEGORY_ALIASES = {
    "restaurant": ["restaurant", "resto", "eat", "food", "fast food"],
    "cafe": ["cafe", "coffee", "coffee shop"],
    "attraction": ["attraction", "museum", "viewpoint", "activity"],
    "park": ["park", "parc", "nature", "garden"],
    "monument": ["monument", "historic", "memorial", "castle", "heritage"],
}

COUNT_PATTERNS = ["combien", "how many", "count", "nombre", "total"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="Question utilisateur sur les lieux au Maroc")
    top_k: int = Field(default=10, ge=1, le=30)


class GuideModel:
    def __init__(self) -> None:
        self.dataset_path: Optional[Path] = None
        self.df: pd.DataFrame = pd.DataFrame()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
        self.matrix = None
        self.city_lookup: List[Tuple[str, str]] = []

    @staticmethod
    def normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def load(self) -> None:
        dataset_path = next((path for path in DATASET_CANDIDATES if path.exists()), None)
        if dataset_path is None:
            tried = ", ".join(path.name for path in DATASET_CANDIDATES)
            raise FileNotFoundError(f"No dataset file found. Tried: {tried}")

        with dataset_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        df = pd.DataFrame(payload.get("places", []))
        if df.empty:
            raise RuntimeError("Dataset is empty.")

        for col in ["name", "city", "normalized_category", "subtype", "address", "lat_lon", "latitude", "longitude"]:
            if col not in df.columns:
                df[col] = ""

        df["name"] = df["name"].fillna("")
        df["city"] = df["city"].fillna("")
        df["normalized_category"] = df["normalized_category"].fillna("")
        df["subtype"] = df["subtype"].fillna("")
        df["address"] = df["address"].fillna("")

        if "lat_lon" not in df.columns or df["lat_lon"].astype(str).str.strip().eq("").all():
            df["lat_lon"] = (
                df["latitude"].astype(str).str.strip() + "," + df["longitude"].astype(str).str.strip()
            )

        df["name_norm"] = df["name"].map(self.normalize_text)
        df["city_norm"] = df["city"].map(self.normalize_text)
        df["category_norm"] = df["normalized_category"].map(self.normalize_text)
        df["subtype_norm"] = df["subtype"].map(self.normalize_text)
        df["address_norm"] = df["address"].map(self.normalize_text)

        df["search_text"] = (
            df["name_norm"]
            + " "
            + df["city_norm"]
            + " "
            + df["category_norm"]
            + " "
            + df["subtype_norm"]
            + " "
            + df["address_norm"]
        ).str.strip()

        self.city_lookup = sorted(
            [(self.normalize_text(city), city) for city in df["city"].astype(str).unique().tolist() if str(city).strip()],
            key=lambda item: len(item[0]),
            reverse=True,
        )

        self.matrix = self.vectorizer.fit_transform(df["search_text"].tolist())
        self.df = df
        self.dataset_path = dataset_path

    def detect_category(self, query_norm: str) -> Optional[str]:
        for category, aliases in CATEGORY_ALIASES.items():
            for alias in aliases:
                if self.normalize_text(alias) in query_norm:
                    return category
        return None

    def detect_city(self, query_norm: str) -> Optional[str]:
        for city_norm, city_label in self.city_lookup:
            if city_norm and city_norm in query_norm:
                return city_label
        return None

    def ask(self, question: str, top_k: int) -> Dict[str, Any]:
        if self.df.empty or self.matrix is None:
            raise RuntimeError("Model is not loaded.")

        query_norm = self.normalize_text(question)
        if not query_norm:
            return {
                "answer": "Question vide. Exemple: restaurants a Marrakech",
                "count": 0,
                "matches": [],
            }

        category = self.detect_category(query_norm)
        city = self.detect_city(query_norm)
        wants_count = any(pattern in query_norm for pattern in COUNT_PATTERNS)

        candidate_df = self.df
        if category:
            candidate_df = candidate_df[candidate_df["normalized_category"].str.lower() == category]
        if city:
            city_norm = self.normalize_text(city)
            candidate_df = candidate_df[candidate_df["city_norm"] == city_norm]

        if candidate_df.empty:
            details = []
            if category:
                details.append(f"categorie={category}")
            if city:
                details.append(f"ville={city}")
            suffix = ", ".join(details) if details else "filtres"
            return {
                "answer": f"Aucun resultat pour {suffix}.",
                "count": 0,
                "matches": [],
            }

        query_vector = self.vectorizer.transform([query_norm])
        candidate_matrix = self.matrix[candidate_df.index]
        sims = cosine_similarity(query_vector, candidate_matrix).flatten()

        candidate_df = candidate_df.copy()
        candidate_df["score"] = sims
        candidate_df = candidate_df.sort_values(by=["score", "name"], ascending=[False, True], kind="stable")

        if wants_count:
            answer = f"J'ai trouve {len(candidate_df)} lieux"
            if category:
                answer += f" dans la categorie {category}"
            if city:
                answer += f" a {city}"
            answer += "."
        else:
            answer = "Voici les meilleures suggestions pour ta question."

        top = candidate_df.head(top_k)
        matches = []
        for _, row in top.iterrows():
            matches.append(
                {
                    "name": row.get("name") or "",
                    "category": row.get("normalized_category") or "",
                    "city": row.get("city") or "",
                    "lat_lon": row.get("lat_lon") or "",
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                    "address": row.get("address") or "",
                    "score": float(row.get("score", 0.0)),
                }
            )

        return {
            "answer": answer,
            "count": int(len(candidate_df)),
            "dataset_file": self.dataset_path.name if self.dataset_path else None,
            "filters": {
                "category": category,
                "city": city,
            },
            "matches": matches,
        }


app = FastAPI(title="Morocco AI Guide API", version="1.0.0")
guide_model = GuideModel()


@app.on_event("startup")
def startup_event() -> None:
    guide_model.load()


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": "Morocco AI Guide API is running.",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "ask": "/ask",
        },
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "dataset": guide_model.dataset_path.name if guide_model.dataset_path else None,
        "records": int(len(guide_model.df)) if not guide_model.df.empty else 0,
    }


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    try:
        return guide_model.ask(req.question, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
