from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class RowReport:
    title: str
    description: str
    data: pd.DataFrame
    confidence: float


def highest_missing_rows(df: pd.DataFrame, limit: int = 10) -> RowReport:
    missing_counts = df.isna().sum(axis=1)
    data = df.copy()
    data["missing_count"] = missing_counts
    data = data.sort_values("missing_count", ascending=False).head(limit)
    return RowReport(
        title="Rows with Most Missing Values",
        description="Rows that have the highest number of missing cells.",
        data=data,
        confidence=0.6,
    )


def extreme_numeric_rows(df: pd.DataFrame, limit: int = 10) -> RowReport:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        data = pd.DataFrame({"note": ["No numeric columns found."]})
        return RowReport(
            title="Extreme Numeric Rows",
            description="Rows with extreme numeric values.",
            data=data,
            confidence=0.6,
        )
    zscores = (numeric - numeric.mean()) / numeric.std(ddof=0)
    score = zscores.abs().sum(axis=1)
    data = df.copy()
    data["extreme_score"] = score
    data = data.sort_values("extreme_score", ascending=False).head(limit)
    return RowReport(
        title="Extreme Numeric Rows",
        description="Rows with the strongest numeric extremes.",
        data=data,
        confidence=0.65,
    )


def rare_pattern_rows(df: pd.DataFrame, limit: int = 10) -> RowReport:
    categorical = df.select_dtypes(include="object")
    if categorical.empty:
        data = pd.DataFrame({"note": ["No categorical columns found."]})
        return RowReport(
            title="Rare Pattern Rows",
            description="Rows with rare categorical combinations.",
            data=data,
            confidence=0.6,
        )
    combined = categorical.astype(str).agg(" | ".join, axis=1)
    counts = combined.value_counts()
    rarity = combined.map(counts).astype(float)
    data = df.copy()
    data["rarity_score"] = 1 / rarity
    data = data.sort_values("rarity_score", ascending=False).head(limit)
    return RowReport(
        title="Rare Pattern Rows",
        description="Rows with rare category combinations.",
        data=data,
        confidence=0.6,
    )


def generate_row_reports(df: pd.DataFrame) -> List[RowReport]:
    return [
        highest_missing_rows(df),
        extreme_numeric_rows(df),
        rare_pattern_rows(df),
    ]
