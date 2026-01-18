from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ReportCatalogItem:
    title: str
    description: str
    report_type: str
    confidence: float


def build_catalog(report_type: str, reports: Iterable) -> List[ReportCatalogItem]:
    items: List[ReportCatalogItem] = []
    for report in reports:
        confidence = getattr(report, "confidence", 0.5)
        items.append(
            ReportCatalogItem(
                title=report.title,
                description=report.description,
                report_type=report_type,
                confidence=confidence,
            )
        )
    return items
