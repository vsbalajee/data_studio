from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    title: str
    description: str
    data: pd.DataFrame
    confidence: float


def invalid_format_report(df: pd.DataFrame) -> List[QualityReport]:
    reports: List[QualityReport] = []
    for col in df.columns:
        if "email" in col.lower():
            series = df[col].astype(str)
            invalid = ~series.str.match(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", na=False)
            data = pd.DataFrame(
                {
                    "invalid_count": [int(invalid.sum())],
                    "invalid_rate": [round(invalid.mean(), 3)],
                }
            )
            reports.append(
                QualityReport(
                    title=f"Invalid Emails - {col}",
                    description="Rows that do not look like valid emails.",
                    data=data,
                    confidence=0.6,
                )
            )
        if "phone" in col.lower():
            series = df[col].astype(str)
            invalid = ~series.str.match(r"^[0-9\\-\\+\\(\\)\\s]{6,}$", na=False)
            data = pd.DataFrame(
                {
                    "invalid_count": [int(invalid.sum())],
                    "invalid_rate": [round(invalid.mean(), 3)],
                }
            )
            reports.append(
                QualityReport(
                    title=f"Invalid Phones - {col}",
                    description="Rows that do not look like valid phone numbers.",
                    data=data,
                    confidence=0.6,
                )
            )
    return reports


def outlier_summary(df: pd.DataFrame) -> QualityReport:
    numeric = df.select_dtypes(include="number")
    rows = []
    for col in numeric.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        rows.append(
            {
                "column": col,
                "outliers": int(outliers),
                "outlier_rate": round(outliers / max(len(series), 1), 3),
            }
        )
    if not rows:
        data = pd.DataFrame({"note": ["No numeric columns for outliers."]})
    else:
        data = pd.DataFrame(rows)
    return QualityReport(
        title="Outlier Summary",
        description="IQR-based outlier counts per numeric column.",
        data=data,
        confidence=0.65,
    )


def robust_outlier_summary(df: pd.DataFrame) -> QualityReport:
    numeric = df.select_dtypes(include="number")
    rows = []
    for col in numeric.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            continue
        modified_z = 0.6745 * (series - median) / mad
        outliers = (modified_z.abs() > 3.5).sum()
        rows.append(
            {
                "column": col,
                "mad_outliers": int(outliers),
                "mad_outlier_rate": round(outliers / max(len(series), 1), 3),
            }
        )
    if not rows:
        data = pd.DataFrame({"note": ["No numeric columns for MAD outliers."]})
    else:
        data = pd.DataFrame(rows)
    return QualityReport(
        title="Robust Outliers (MAD)",
        description="Median absolute deviation based outlier counts.",
        data=data,
        confidence=0.7,
    )


def label_variants(df: pd.DataFrame) -> QualityReport:
    rows = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        series = df[col].dropna().astype(str)
        lowered = series.str.strip().str.lower()
        variants = (
            pd.DataFrame({"raw": series, "norm": lowered})
            .groupby("norm")["raw"]
            .nunique()
            .reset_index()
        )
        inconsistent = variants[variants["raw"] > 1]
        if not inconsistent.empty:
            rows.append(
                {
                    "column": col,
                    "inconsistent_labels": int(inconsistent.shape[0]),
                }
            )
    if not rows:
        data = pd.DataFrame({"note": ["No label inconsistencies detected."]})
    else:
        data = pd.DataFrame(rows)
    return QualityReport(
        title="Label Variants",
        description="Columns with inconsistent text variants (case/spacing).",
        data=data,
        confidence=0.7,
    )


def drift_summary(df: pd.DataFrame) -> QualityReport:
    numeric = df.select_dtypes(include="number")
    if numeric.empty or len(df) < 10:
        data = pd.DataFrame({"note": ["Not enough data for drift check."]})
    else:
        midpoint = len(df) // 2
        first = numeric.iloc[:midpoint]
        second = numeric.iloc[midpoint:]
        rows = []
        for col in numeric.columns:
            first_mean = first[col].mean()
            second_mean = second[col].mean()
            if pd.isna(first_mean) or pd.isna(second_mean):
                continue
            change = second_mean - first_mean
            rows.append(
                {
                    "column": col,
                    "first_mean": round(first_mean, 3),
                    "second_mean": round(second_mean, 3),
                    "change": round(change, 3),
                }
            )
        data = pd.DataFrame(rows) if rows else pd.DataFrame({"note": ["No drift found."]})
    return QualityReport(
        title="Numeric Drift",
        description="Mean shift between first and second halves of the data.",
        data=data,
        confidence=0.6,
    )


def category_drift(df: pd.DataFrame) -> QualityReport:
    categorical = df.select_dtypes(include="object")
    if categorical.empty or len(df) < 10:
        data = pd.DataFrame({"note": ["Not enough data for category drift."]})
    else:
        midpoint = len(df) // 2
        rows = []
        for col in categorical.columns:
            first = df[col].iloc[:midpoint].astype(str)
            second = df[col].iloc[midpoint:].astype(str)
            first_dist = first.value_counts(normalize=True)
            second_dist = second.value_counts(normalize=True)
            shared = set(first_dist.index) | set(second_dist.index)
            drift = sum(abs(first_dist.get(k, 0) - second_dist.get(k, 0)) for k in shared)
            rows.append({"column": col, "distribution_shift": round(drift, 3)})
        data = pd.DataFrame(rows)
    return QualityReport(
        title="Category Drift",
        description="Distribution shift between first and second halves.",
        data=data,
        confidence=0.6,
    )


def time_spike_check(df: pd.DataFrame) -> QualityReport:
    date_cols = []
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            date_cols.append(col)
    if not date_cols:
        data = pd.DataFrame({"note": ["No strong date columns found."]})
    else:
        rows = []
        for col in date_cols:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            counts = parsed.dt.to_period("M").value_counts().sort_index()
            if len(counts) < 3:
                continue
            mean = counts.mean()
            std = counts.std(ddof=0)
            spikes = (counts > mean + 2 * std).sum() if std > 0 else 0
            drops = (counts < mean - 2 * std).sum() if std > 0 else 0
            rows.append(
                {
                    "date_column": col,
                    "spike_periods": int(spikes),
                    "drop_periods": int(drops),
                }
            )
        data = pd.DataFrame(rows) if rows else pd.DataFrame({"note": ["Not enough time data."]})
    return QualityReport(
        title="Time Series Spikes",
        description="Detect spikes/drops in monthly record counts.",
        data=data,
        confidence=0.6,
    )


def generate_quality_reports(df: pd.DataFrame) -> List[QualityReport]:
    reports: List[QualityReport] = []
    reports.append(outlier_summary(df))
    reports.append(robust_outlier_summary(df))
    reports.append(label_variants(df))
    reports.append(drift_summary(df))
    reports.append(category_drift(df))
    reports.append(time_spike_check(df))
    reports.extend(invalid_format_report(df))
    return reports
