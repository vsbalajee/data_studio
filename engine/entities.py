from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from engine.schema import ColumnInference


@dataclass
class EntityGroup:
    name: str
    columns: List[str]
    confidence: float


def _extract_prefix(column: str) -> str:
    lowered = column.lower()
    for sep in ("_", " "):
        if sep in lowered:
            return lowered.split(sep)[0]
    return lowered


def cluster_entities(inferences: Sequence[ColumnInference]) -> List[EntityGroup]:
    buckets: Dict[str, List[str]] = {}
    role_hints: Dict[str, List[str]] = {}
    for inf in inferences:
        prefix = _extract_prefix(inf.column)
        buckets.setdefault(prefix, []).append(inf.column)
        role_hints.setdefault(prefix, []).append(inf.semantic_role)

    entities: List[EntityGroup] = []
    for prefix, columns in buckets.items():
        roles = role_hints.get(prefix, [])
        confidence = 0.5
        if any(role == "id" for role in roles) and len(columns) >= 2:
            confidence = 0.8
        elif len(columns) >= 3:
            confidence = 0.7
        entities.append(
            EntityGroup(
                name=prefix.title(),
                columns=columns,
                confidence=confidence,
            )
        )
    return entities
