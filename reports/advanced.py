from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class AdvancedReport:
    title: str
    description: str
    data: pd.DataFrame
    level: str
    confidence: float


def correlation_matrix(df: pd.DataFrame) -> AdvancedReport:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        data = pd.DataFrame({"note": ["Not enough numeric columns for correlation."]})
    else:
        data = numeric.corr().round(3)
    return AdvancedReport(
        title="Correlation Matrix",
        description="Correlation between numeric columns.",
        data=data,
        level="full",
        confidence=0.7,
    )


def pareto_report(df: pd.DataFrame, limit: int = 10) -> AdvancedReport:
    target_col = None
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique(dropna=True) <= 50:
            target_col = col
            break
    if not target_col:
        data = pd.DataFrame({"note": ["No suitable category column found."]})
    else:
        counts = df[target_col].astype(str).value_counts(dropna=False).reset_index()
        counts.columns = [target_col, "count"]
        counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(2)
        counts["cumulative_percent"] = counts["percent"].cumsum().round(2)
        data = counts.head(limit)
    return AdvancedReport(
        title="Pareto Summary",
        description="Top categories with cumulative contribution.",
        data=data,
        level="fast",
        confidence=0.75,
    )


def distribution_buckets(df: pd.DataFrame, bins: int = 10) -> List[AdvancedReport]:
    reports: List[AdvancedReport] = []
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        series = df[col].dropna()
        if series.empty:
            data = pd.DataFrame({"note": ["No numeric data found."]})
        else:
            hist, edges = np.histogram(series, bins=bins)
            data = pd.DataFrame(
                {
                    "bin_start": edges[:-1],
                    "bin_end": edges[1:],
                    "count": hist,
                }
            )
        reports.append(
            AdvancedReport(
                title=f"Distribution - {col}",
                description="Bucketed distribution for numeric values.",
                data=data,
                level="fast",
                confidence=0.7,
            )
        )
    return reports


def outlier_buckets(df: pd.DataFrame) -> List[AdvancedReport]:
    reports: List[AdvancedReport] = []
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        series = df[col].dropna()
        if series.empty:
            data = pd.DataFrame({"note": ["No numeric data found."]})
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            data = pd.DataFrame(
                {
                    "lower_bound": [lower],
                    "upper_bound": [upper],
                    "outliers": [int(((series < lower) | (series > upper)).sum())],
                }
            )
        reports.append(
            AdvancedReport(
                title=f"Outlier Buckets - {col}",
                description="IQR-based outlier counts.",
                data=data,
                level="full",
                confidence=0.65,
            )
        )
    return reports


def skewness_report(df: pd.DataFrame) -> AdvancedReport:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        data = pd.DataFrame({"note": ["No numeric columns found."]})
    else:
        data = numeric.skew(numeric_only=True).to_frame("skewness").reset_index()
        data = data.rename(columns={"index": "column"})
    return AdvancedReport(
        title="Skewness",
        description="Skewness for numeric columns.",
        data=data,
        level="fast",
        confidence=0.7,
    )


def mean_vs_median(df: pd.DataFrame) -> AdvancedReport:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        data = pd.DataFrame({"note": ["No numeric columns found."]})
    else:
        data = pd.DataFrame(
            {
                "column": numeric.columns,
                "mean": numeric.mean().values,
                "median": numeric.median().values,
            }
        )
    return AdvancedReport(
        title="Mean vs Median",
        description="Compare mean and median for numeric columns.",
        data=data,
        level="fast",
        confidence=0.75,
    )


def generate_advanced_reports(df: pd.DataFrame, mode: str) -> List[AdvancedReport]:
    reports: List[AdvancedReport] = []
    reports.append(pareto_report(df))
    reports.append(skewness_report(df))
    reports.append(mean_vs_median(df))
    reports.extend(distribution_buckets(df))
    if mode == "full":
        reports.append(correlation_matrix(df))
        reports.extend(outlier_buckets(df))
    return reports
