from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from engine.relationships import RelationshipCandidate


@dataclass
class RelationshipReport:
    title: str
    description: str
    data: pd.DataFrame
    confidence: float


def _pick_best_relationship(
    relationships: Dict[str, List[RelationshipCandidate]],
) -> Optional[Tuple[str, RelationshipCandidate]]:
    best: Optional[Tuple[str, RelationshipCandidate]] = None
    for pair, rels in relationships.items():
        for rel in rels:
            if not best or rel.confidence > best[1].confidence:
                best = (pair, rel)
    return best


def relationship_rollup(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    relationship: RelationshipCandidate,
    metric_col: Optional[str] = None,
) -> RelationshipReport:
    left_key = relationship.left_column
    right_key = relationship.right_column
    if left_key not in left_df.columns or right_key not in right_df.columns:
        data = pd.DataFrame({"note": ["Join keys not found in data."]})
        return RelationshipReport(
            title="Relationship Rollup",
            description="Join keys missing.",
            data=data,
            confidence=0.3,
        )

    merged = left_df.merge(
        right_df,
        left_on=left_key,
        right_on=right_key,
        how="inner",
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        data = pd.DataFrame({"note": ["No joined rows found."]})
        return RelationshipReport(
            title="Relationship Rollup",
            description="No matching keys after join.",
            data=data,
            confidence=0.3,
        )

    if metric_col and metric_col in merged.columns:
        agg = merged.groupby(left_key)[metric_col].sum().reset_index()
        agg = agg.rename(columns={metric_col: "total"})
        title = f"Rollup by {left_key} (sum {metric_col})"
    else:
        agg = merged.groupby(left_key).size().reset_index(name="count")
        title = f"Rollup by {left_key} (count)"

    return RelationshipReport(
        title=title,
        description="Joined rollup across related tables.",
        data=agg,
        confidence=0.75,
    )


def generate_relationship_reports(
    dfs: Dict[str, pd.DataFrame],
    relationships: Dict[str, List[RelationshipCandidate]],
) -> List[RelationshipReport]:
    reports: List[RelationshipReport] = []
    best = _pick_best_relationship(relationships)
    if not best:
        return reports

    pair_label, relationship = best
    left_name, right_name = [name.strip() for name in pair_label.split("->")]
    left_df = dfs.get(left_name)
    right_df = dfs.get(right_name)
    if left_df is None or right_df is None:
        return reports

    metric_col = None
    for col in right_df.select_dtypes(include="number").columns:
        metric_col = col
        break

    reports.append(relationship_rollup(left_df, right_df, relationship, metric_col))
    return reports
