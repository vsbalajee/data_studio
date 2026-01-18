from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class TimeSeriesReport:
    title: str
    description: str
    data: pd.DataFrame
    confidence: float


def _find_date_columns(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(col)
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            candidates.append(col)
    return candidates


def _ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def time_trend_report(
    df: pd.DataFrame,
    date_col: str,
    freq: str,
    value_col: Optional[str] = None,
) -> TimeSeriesReport:
    series = _ensure_datetime(df[date_col]).dropna()
    if series.empty:
        data = pd.DataFrame({"note": ["No valid dates found."]})
    else:
        grouped = pd.DataFrame({date_col: series})
        if value_col and value_col in df.columns:
            grouped[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            grouped = grouped.dropna(subset=[value_col])
            data = (
                grouped.set_index(date_col)
                .resample(freq)[value_col]
                .sum()
                .reset_index()
            )
        else:
            data = grouped.set_index(date_col).resample(freq).size().reset_index(name="count")
    return TimeSeriesReport(
        title=f"Trend ({freq}) - {date_col}",
        description="Counts or totals grouped by time.",
        data=data,
        confidence=0.8,
    )


def period_over_period(df: pd.DataFrame, date_col: str, freq: str) -> TimeSeriesReport:
    series = _ensure_datetime(df[date_col]).dropna()
    if series.empty:
        data = pd.DataFrame({"note": ["No valid dates found."]})
    else:
        grouped = pd.DataFrame({date_col: series})
        counts = grouped.set_index(date_col).resample(freq).size().reset_index(name="count")
        counts["previous"] = counts["count"].shift(1)
        counts["change"] = counts["count"] - counts["previous"]
        counts["change_pct"] = (counts["change"] / counts["previous"].replace(0, pd.NA)).round(3)
        data = counts
    return TimeSeriesReport(
        title=f"Period-over-Period ({freq}) - {date_col}",
        description="Current vs previous period changes.",
        data=data,
        confidence=0.75,
    )


def generate_time_reports(df: pd.DataFrame) -> List[TimeSeriesReport]:
    reports: List[TimeSeriesReport] = []
    date_columns = _find_date_columns(df)
    for date_col in date_columns:
        try:
            reports.append(time_trend_report(df, date_col, "D"))
            reports.append(time_trend_report(df, date_col, "M"))
            reports.append(period_over_period(df, date_col, "M"))
        except Exception as exc:
            reports.append(
                TimeSeriesReport(
                    title=f"Time Series Error - {date_col}",
                    description=f"Unable to build time reports: {exc}",
                    data=pd.DataFrame({"note": ["Time series processing failed."]}),
                    confidence=0.3,
                )
            )
    return reports
