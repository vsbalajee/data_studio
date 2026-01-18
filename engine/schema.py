from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnInference:
    column: str
    dtype: str
    inferred_type: str
    semantic_role: str
    confidence: float
    samples: List[str]


ROLE_PATTERNS = {
    "id": ("_id", " id", "id_", "identifier", "uuid"),
    "date": ("date", "datetime", "timestamp", "time"),
    "amount": ("amount", "amt", "price", "cost", "total", "revenue", "sales"),
    "status": ("status", "state", "stage"),
    "category": ("type", "category", "segment", "group", "class"),
    "email": ("email", "e-mail"),
    "phone": ("phone", "mobile", "contact"),
    "name": ("name", "title"),
    "country": ("country", "nation"),
    "city": ("city",),
}


def _sample_values(series: pd.Series, limit: int = 5) -> List[str]:
    values = series.dropna().astype(str).unique().tolist()
    return values[:limit]


def _numeric_stats(series: pd.Series) -> Dict[str, float]:
    cleaned = pd.to_numeric(series, errors="coerce")
    non_null = cleaned.dropna()
    if non_null.empty:
        return {"ratio": 0.0, "unique_ratio": 0.0}
    ratio = len(non_null) / max(len(series), 1)
    unique_ratio = non_null.nunique() / max(len(non_null), 1)
    return {"ratio": ratio, "unique_ratio": unique_ratio}


def _date_stats(series: pd.Series) -> Dict[str, float]:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    non_null = parsed.dropna()
    if non_null.empty:
        return {"ratio": 0.0}
    ratio = len(non_null) / max(len(series), 1)
    return {"ratio": ratio}


def _guess_role_from_name(name: str) -> Optional[str]:
    lowered = name.lower()
    for role, patterns in ROLE_PATTERNS.items():
        if any(p in lowered for p in patterns):
            return role
    return None


def infer_schema(df: pd.DataFrame) -> List[ColumnInference]:
    results: List[ColumnInference] = []
    total_rows = len(df)
    for column in df.columns:
        series = df[column]
        dtype = str(series.dtype)
        name_role = _guess_role_from_name(column)

        numeric_stats = _numeric_stats(series)
        date_stats = _date_stats(series)

        inferred_type = "text"
        confidence = 0.3

        if pd.api.types.is_numeric_dtype(series):
            inferred_type = "number"
            confidence = 0.9
        elif date_stats["ratio"] >= 0.8:
            inferred_type = "date"
            confidence = min(0.95, 0.5 + date_stats["ratio"] / 2)
        elif numeric_stats["ratio"] >= 0.9:
            inferred_type = "number"
            confidence = min(0.85, 0.4 + numeric_stats["ratio"] / 2)
        elif pd.api.types.is_bool_dtype(series):
            inferred_type = "boolean"
            confidence = 0.85
        else:
            inferred_type = "text"
            confidence = 0.6

        semantic_role = name_role or "attribute"

        if name_role == "id":
            unique_ratio = series.nunique(dropna=True) / max(total_rows, 1)
            if unique_ratio >= 0.95:
                confidence = min(0.98, confidence + 0.1)
            inferred_type = "id"

        results.append(
            ColumnInference(
                column=str(column),
                dtype=dtype,
                inferred_type=inferred_type,
                semantic_role=semantic_role,
                confidence=round(confidence, 2),
                samples=_sample_values(series),
            )
        )

    return results
