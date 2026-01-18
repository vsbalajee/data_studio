from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from engine.schema import ColumnInference


@dataclass
class KeyCandidate:
    column: str
    uniqueness: float
    null_rate: float
    confidence: float


@dataclass
class RelationshipCandidate:
    left_column: str
    right_column: str
    overlap: float
    confidence: float


@dataclass
class CompositeKeyCandidate:
    columns: Tuple[str, str]
    uniqueness: float
    null_rate: float
    confidence: float


@dataclass
class RelationshipValidation:
    left_column: str
    right_column: str
    overlap: float
    orphan_rate: float
    left_unique: bool
    right_unique: bool
    cardinality: str
    confidence: float


def find_key_candidates(df: pd.DataFrame) -> List[KeyCandidate]:
    candidates: List[KeyCandidate] = []
    total = max(len(df), 1)
    for column in df.columns:
        series = df[column]
        null_rate = series.isna().mean()
        unique = series.nunique(dropna=True)
        uniqueness = unique / total
        confidence = 0.0
        if uniqueness >= 0.98 and null_rate <= 0.02:
            confidence = 0.9
        elif uniqueness >= 0.95 and null_rate <= 0.05:
            confidence = 0.7
        if confidence > 0.0:
            candidates.append(
                KeyCandidate(
                    column=str(column),
                    uniqueness=round(uniqueness, 3),
                    null_rate=round(null_rate, 3),
                    confidence=confidence,
                )
            )
    return candidates


def find_composite_key_candidates(
    df: pd.DataFrame, max_pairs: int = 20
) -> List[CompositeKeyCandidate]:
    candidates: List[CompositeKeyCandidate] = []
    total = max(len(df), 1)
    columns = list(df.columns)
    pairs_checked = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if pairs_checked >= max_pairs:
                return candidates
            col_a, col_b = columns[i], columns[j]
            pair = df[[col_a, col_b]]
            null_rate = pair.isna().any(axis=1).mean()
            unique = pair.dropna().drop_duplicates().shape[0]
            uniqueness = unique / total
            confidence = 0.0
            if uniqueness >= 0.98 and null_rate <= 0.02:
                confidence = 0.8
            elif uniqueness >= 0.95 and null_rate <= 0.05:
                confidence = 0.6
            if confidence > 0.0:
                candidates.append(
                    CompositeKeyCandidate(
                        columns=(str(col_a), str(col_b)),
                        uniqueness=round(uniqueness, 3),
                        null_rate=round(null_rate, 3),
                        confidence=confidence,
                    )
                )
            pairs_checked += 1
    return candidates


def find_triple_key_candidates(
    df: pd.DataFrame, max_triples: int = 10
) -> List[CompositeKeyCandidate]:
    candidates: List[CompositeKeyCandidate] = []
    total = max(len(df), 1)
    columns = list(df.columns)
    triples_checked = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            for k in range(j + 1, len(columns)):
                if triples_checked >= max_triples:
                    return candidates
                col_a, col_b, col_c = columns[i], columns[j], columns[k]
                trio = df[[col_a, col_b, col_c]]
                null_rate = trio.isna().any(axis=1).mean()
                unique = trio.dropna().drop_duplicates().shape[0]
                uniqueness = unique / total
                confidence = 0.0
                if uniqueness >= 0.98 and null_rate <= 0.02:
                    confidence = 0.75
                elif uniqueness >= 0.95 and null_rate <= 0.05:
                    confidence = 0.55
                if confidence > 0.0:
                    candidates.append(
                        CompositeKeyCandidate(
                            columns=(str(col_a), str(col_b), str(col_c)),
                            uniqueness=round(uniqueness, 3),
                            null_rate=round(null_rate, 3),
                            confidence=confidence,
                        )
                    )
                triples_checked += 1
    return candidates


def _possible_fk_pairs(
    inferences: Sequence[ColumnInference],
) -> List[Tuple[str, str]]:
    ids = [inf.column for inf in inferences if inf.semantic_role == "id"]
    pairs: List[Tuple[str, str]] = []
    for left in ids:
        for right in ids:
            if left != right:
                pairs.append((left, right))
    return pairs


def _value_overlap(left: pd.Series, right: pd.Series) -> float:
    left_vals = set(left.dropna().astype(str).unique())
    right_vals = set(right.dropna().astype(str).unique())
    if not left_vals or not right_vals:
        return 0.0
    overlap = len(left_vals & right_vals) / max(len(left_vals), 1)
    return overlap


def find_relationships(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_inferences: Sequence[ColumnInference],
    right_inferences: Sequence[ColumnInference],
) -> List[RelationshipCandidate]:
    candidates: List[RelationshipCandidate] = []
    for left_col, right_col in _possible_fk_pairs(left_inferences):
        if left_col not in left_df.columns or right_col not in right_df.columns:
            continue
        overlap = _value_overlap(left_df[left_col], right_df[right_col])
        confidence = 0.0
        if overlap >= 0.8:
            confidence = 0.85
        elif overlap >= 0.6:
            confidence = 0.7
        elif overlap >= 0.4:
            confidence = 0.5
        if confidence > 0.0:
            candidates.append(
                RelationshipCandidate(
                    left_column=left_col,
                    right_column=right_col,
                    overlap=round(overlap, 2),
                    confidence=confidence,
                )
            )
    return candidates


def validate_relationships(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_inferences: Sequence[ColumnInference],
    right_inferences: Sequence[ColumnInference],
) -> List[RelationshipValidation]:
    validations: List[RelationshipValidation] = []
    total_left = max(len(left_df), 1)
    total_right = max(len(right_df), 1)
    for left_col, right_col in _possible_fk_pairs(left_inferences):
        if left_col not in left_df.columns or right_col not in right_df.columns:
            continue
        left_series = left_df[left_col]
        right_series = right_df[right_col]
        overlap = _value_overlap(left_series, right_series)
        left_unique = left_series.nunique(dropna=True) / total_left >= 0.98
        right_unique = right_series.nunique(dropna=True) / total_right >= 0.98
        right_values = set(right_series.dropna().astype(str).unique())
        left_values = set(left_series.dropna().astype(str).unique())
        if left_values:
            orphan_rate = 1 - (len(left_values & right_values) / len(left_values))
        else:
            orphan_rate = 1.0
        if left_unique and right_unique:
            cardinality = "1-1"
        elif left_unique and not right_unique:
            cardinality = "1-N"
        elif not left_unique and right_unique:
            cardinality = "N-1"
        else:
            cardinality = "N-N"
        confidence = 0.0
        if overlap >= 0.8 and orphan_rate <= 0.1:
            confidence = 0.85
        elif overlap >= 0.6 and orphan_rate <= 0.2:
            confidence = 0.7
        elif overlap >= 0.4 and orphan_rate <= 0.3:
            confidence = 0.5
        if confidence > 0.0:
            validations.append(
                RelationshipValidation(
                    left_column=left_col,
                    right_column=right_col,
                    overlap=round(overlap, 2),
                    orphan_rate=round(orphan_rate, 2),
                    left_unique=left_unique,
                    right_unique=right_unique,
                    cardinality=cardinality,
                    confidence=confidence,
                )
            )
    return validations


def sample_orphans(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: str,
    limit: int = 5,
) -> List[str]:
    left_values = left_df[left_key].dropna().astype(str)
    right_values = set(right_df[right_key].dropna().astype(str))
    orphans = left_values[~left_values.isin(right_values)]
    return orphans.unique().tolist()[:limit]


def infer_relationships(
    dfs: Dict[str, pd.DataFrame],
    schemas: Dict[str, Sequence[ColumnInference]],
) -> Dict[str, List[RelationshipCandidate]]:
    results: Dict[str, List[RelationshipCandidate]] = {}
    names = list(dfs.keys())
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left_df = dfs[left_name]
            right_df = dfs[right_name]
            left_inf = schemas[left_name]
            right_inf = schemas[right_name]
            rels = find_relationships(left_df, right_df, left_inf, right_inf)
            if rels:
                key = f"{left_name} -> {right_name}"
                results[key] = rels
    return results


def validate_relationship_graph(
    dfs: Dict[str, pd.DataFrame],
    schemas: Dict[str, Sequence[ColumnInference]],
) -> Dict[str, List[RelationshipValidation]]:
    results: Dict[str, List[RelationshipValidation]] = {}
    names = list(dfs.keys())
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            left_df = dfs[left_name]
            right_df = dfs[right_name]
            left_inf = schemas[left_name]
            right_inf = schemas[right_name]
            rels = validate_relationships(left_df, right_df, left_inf, right_inf)
            if rels:
                key = f"{left_name} -> {right_name}"
                results[key] = rels
    return results
