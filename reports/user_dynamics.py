from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class UserDynamicReport:
    title: str
    description: str
    data: pd.DataFrame


def apply_filters(
    df: pd.DataFrame,
    categorical_filters: Dict[str, List[str]],
    numeric_filters: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> pd.DataFrame:
    filtered = df.copy()
    for col, values in categorical_filters.items():
        if col in filtered.columns and values:
            filtered = filtered[filtered[col].astype(str).isin(values)]
    for col, bounds in numeric_filters.items():
        if col in filtered.columns:
            series = pd.to_numeric(filtered[col], errors="coerce")
            min_val, max_val = bounds
            if min_val is not None:
                filtered = filtered[series >= min_val]
            if max_val is not None:
                filtered = filtered[series <= max_val]
    return filtered


def build_dynamic_report(
    df: pd.DataFrame,
    dimensions: List[str],
    metric: Optional[str],
    agg: str,
    top_n: Optional[int],
    percent_of_total: bool,
    categorical_filters: Dict[str, List[str]],
    numeric_filters: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> UserDynamicReport:
    filtered = apply_filters(df, categorical_filters, numeric_filters)
    if not dimensions:
        raise ValueError("Select at least one dimension column.")
    if metric and metric in filtered.columns:
        metric_series = pd.to_numeric(filtered[metric], errors="coerce")
        filtered = filtered.assign(_metric=metric_series)
        if agg == "sum":
            grouped = filtered.groupby(dimensions)["_metric"].sum().reset_index()
        elif agg == "avg":
            grouped = filtered.groupby(dimensions)["_metric"].mean().reset_index()
        elif agg == "min":
            grouped = filtered.groupby(dimensions)["_metric"].min().reset_index()
        elif agg == "max":
            grouped = filtered.groupby(dimensions)["_metric"].max().reset_index()
        else:
            grouped = filtered.groupby(dimensions)["_metric"].count().reset_index()
        grouped = grouped.rename(columns={"_metric": f"{agg}_{metric}"})
        value_col = f"{agg}_{metric}"
    else:
        grouped = filtered.groupby(dimensions).size().reset_index(name="count")
        value_col = "count"

    if percent_of_total and not grouped.empty:
        total = grouped[value_col].sum()
        if total:
            grouped["percent_of_total"] = (grouped[value_col] / total * 100).round(2)

    if top_n and top_n > 0:
        grouped = grouped.sort_values(value_col, ascending=False).head(top_n)

    title = "User Dynamic Report"
    description = "Custom report based on selected columns and filters."
    return UserDynamicReport(title=title, description=description, data=grouped)


def build_lookup_report(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    return_cols: List[str],
) -> UserDynamicReport:
    if left_key not in left_df.columns or right_key not in right_df.columns:
        raise ValueError("Lookup keys must exist in both datasets.")
    if not return_cols:
        raise ValueError("Select at least one return column.")
    missing_cols = [c for c in return_cols if c not in right_df.columns]
    if missing_cols:
        raise ValueError(f"Missing return columns: {', '.join(missing_cols)}")
    left_key_series = left_df[left_key].astype(str)
    right_key_series = right_df[right_key].astype(str)
    left_temp = left_df.copy()
    right_temp = right_df[[right_key] + return_cols].copy()
    left_temp["_lookup_key"] = left_key_series
    right_temp["_lookup_key"] = right_key_series
    merged = left_temp.merge(
        right_temp,
        on="_lookup_key",
        how="left",
        suffixes=("", "_lookup"),
    ).drop(columns=["_lookup_key"])
    title = "Lookup Report"
    description = "Join-based lookup returning selected columns."
    return UserDynamicReport(title=title, description=description, data=merged)


def _series_or_const(df: pd.DataFrame, token: str):
    if token in df.columns:
        return df[token]
    try:
        return float(token)
    except (TypeError, ValueError):
        return token


def _apply_filter_rule(series: pd.Series, operator: str, value: str) -> pd.Series:
    if operator == "=":
        return series.astype(str) == value
    if operator == "!=":
        return series.astype(str) != value
    if operator == ">":
        return pd.to_numeric(series, errors="coerce") > float(value)
    if operator == ">=":
        return pd.to_numeric(series, errors="coerce") >= float(value)
    if operator == "<":
        return pd.to_numeric(series, errors="coerce") < float(value)
    if operator == "<=":
        return pd.to_numeric(series, errors="coerce") <= float(value)
    if operator == "contains":
        return series.astype(str).str.contains(value, na=False)
    if operator == "starts_with":
        return series.astype(str).str.startswith(value, na=False)
    if operator == "ends_with":
        return series.astype(str).str.endswith(value, na=False)
    if operator == "is_null":
        return series.isna()
    if operator == "not_null":
        return series.notna()
    return pd.Series([True] * len(series), index=series.index)


def _apply_filters(df: pd.DataFrame, rules: List[dict], mode: str) -> pd.DataFrame:
    if not rules:
        return df
    masks = []
    for rule in rules:
        col = rule.get("column")
        op = rule.get("operator")
        val = rule.get("value", "")
        if not col or col not in df.columns:
            continue
        masks.append(_apply_filter_rule(df[col], op, str(val)))
    if not masks:
        return df
    if mode == "OR":
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
    else:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
    return df[combined]


def build_sql_features_report(
    df: pd.DataFrame,
    select_columns: List[str],
    aliases: Dict[str, str],
    filters: List[dict],
    filter_mode: str,
    calculated_fields: List[dict],
    case_fields: List[dict],
    date_extracts: List[dict],
    group_by: List[str],
    aggregations: List[dict],
    having: List[dict],
    having_mode: str,
    distinct: bool,
    order_by: Optional[Tuple[str, bool]],
    limit: Optional[int],
    null_fills: Dict[str, str],
) -> UserDynamicReport:
    working = df.copy()

    for col, value in null_fills.items():
        if col in working.columns and value != "":
            working[col] = working[col].fillna(value)

    for calc in calculated_fields:
        name = calc.get("name")
        left = calc.get("left")
        right = calc.get("right")
        op = calc.get("operator")
        if not name or not left or not op:
            continue
        left_series = _series_or_const(working, left)
        right_series = _series_or_const(working, right) if right is not None else None
        if op == "+":
            working[name] = left_series + right_series
        elif op == "-":
            working[name] = left_series - right_series
        elif op == "*":
            working[name] = left_series * right_series
        elif op == "/":
            working[name] = left_series / right_series
        elif op == "concat":
            working[name] = working[left].astype(str) + working[right].astype(str)

    for case in case_fields:
        name = case.get("name")
        col = case.get("column")
        op = case.get("operator")
        value = case.get("value")
        then_val = case.get("then")
        else_val = case.get("else")
        if not name or not col or col not in working.columns:
            continue
        cond = _apply_filter_rule(working[col], op, str(value))
        working[name] = pd.Series([else_val] * len(working), index=working.index)
        working.loc[cond, name] = then_val

    for extract in date_extracts:
        name = extract.get("name")
        col = extract.get("column")
        part = extract.get("part")
        if not name or not col or col not in working.columns:
            continue
        series = pd.to_datetime(working[col], errors="coerce")
        if part == "year":
            working[name] = series.dt.year
        elif part == "month":
            working[name] = series.dt.month
        elif part == "day":
            working[name] = series.dt.day

    working = _apply_filters(working, filters, filter_mode)

    if aggregations:
        agg_map = {}
        for agg in aggregations:
            col = agg.get("column")
            func = agg.get("agg")
            if not col or col not in working.columns:
                continue
            if func == "count_distinct":
                agg_map[col] = "nunique"
            elif func == "count":
                agg_map[col] = "count"
            else:
                agg_map[col] = func
        if group_by:
            aggregated = working.groupby(group_by).agg(agg_map).reset_index()
        else:
            aggregated = working.agg(agg_map).to_frame().T
        working = aggregated

    working = _apply_filters(working, having, having_mode)

    if select_columns:
        keep_cols = [c for c in select_columns if c in working.columns]
        working = working[keep_cols]

    if aliases:
        rename_map = {k: v for k, v in aliases.items() if v}
        working = working.rename(columns=rename_map)

    if distinct:
        working = working.drop_duplicates()

    if order_by and order_by[0] in working.columns:
        working = working.sort_values(order_by[0], ascending=order_by[1])

    if limit and limit > 0:
        working = working.head(limit)

    return UserDynamicReport(
        title="SQL Features Report",
        description="SQL-style report generated from user selections.",
        data=working,
    )
