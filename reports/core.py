from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class CoreReport:
    title: str
    description: str
    data: pd.DataFrame
    confidence: float


def dataset_overview(df: pd.DataFrame) -> CoreReport:
    data = pd.DataFrame(
        {
            "rows": [len(df)],
            "columns": [len(df.columns)],
        }
    )
    return CoreReport(
        title="Dataset Overview",
        description="A quick snapshot of row and column counts.",
        data=data,
        confidence=0.95,
    )


def column_profile(df: pd.DataFrame) -> CoreReport:
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_rate": round(series.isna().mean(), 3),
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return CoreReport(
        title="Column Profile Summary",
        description="Type, missing count, and unique values per column.",
        data=pd.DataFrame(rows),
        confidence=0.9,
    )


def missing_values(df: pd.DataFrame) -> CoreReport:
    missing = df.isna().sum().to_frame("missing").reset_index()
    missing = missing.rename(columns={"index": "column"})
    missing["missing_rate"] = (missing["missing"] / max(len(df), 1)).round(3)
    return CoreReport(
        title="Missing Values",
        description="Missing values per column with rates.",
        data=missing,
        confidence=0.9,
    )


def duplicate_rows(df: pd.DataFrame) -> CoreReport:
    dupes = int(df.duplicated().sum())
    data = pd.DataFrame({"duplicate_rows": [dupes]})
    return CoreReport(
        title="Duplicate Rows",
        description="Total number of duplicate rows in the dataset.",
        data=data,
        confidence=0.9,
    )


def top_categories(df: pd.DataFrame, limit: int = 10) -> List[CoreReport]:
    reports: List[CoreReport] = []
    for col in df.columns:
        if df[col].dtype == object or df[col].nunique(dropna=True) <= 50:
            counts = (
                df[col]
                .astype(str)
                .value_counts(dropna=False)
                .head(limit)
                .reset_index()
            )
            counts.columns = [col, "count"]
            reports.append(
                CoreReport(
                    title=f"Top Categories - {col}",
                    description="Most common values and their counts.",
                    data=counts,
                    confidence=0.85,
                )
            )
    return reports


def numeric_summary(df: pd.DataFrame) -> CoreReport:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        data = pd.DataFrame({"note": ["No numeric columns found."]})
    else:
        data = numeric.describe().transpose().reset_index().rename(columns={"index": "column"})
    return CoreReport(
        title="Numeric Summary",
        description="Basic stats (min, max, mean, etc.) for numeric columns.",
        data=data,
        confidence=0.9,
    )


def generate_core_reports(df: pd.DataFrame) -> List[CoreReport]:
    reports: List[CoreReport] = [
        dataset_overview(df),
        column_profile(df),
        missing_values(df),
        duplicate_rows(df),
        numeric_summary(df),
    ]
    reports.extend(top_categories(df))
    return reports
