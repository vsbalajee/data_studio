from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile
from typing import Dict, Iterable, List
import json

import pandas as pd


@dataclass
class ExportItem:
    title: str
    description: str
    data: pd.DataFrame
    file_name: str


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name.lower())


def build_export_items(reports: Iterable, prefix: str) -> List[ExportItem]:
    items: List[ExportItem] = []
    for report in reports:
        safe_title = _safe_name(report.title)
        file_name = f"{prefix}_{safe_title}.csv"
        items.append(
            ExportItem(
                title=report.title,
                description=report.description,
                data=report.data,
                file_name=file_name,
            )
        )
    return items


def write_exports(
    output_dir: Path,
    dataset_name: str,
    items: List[ExportItem],
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for item in items:
        out_path = output_dir / item.file_name
        item.data.to_csv(out_path, index=False)
        written.append(out_path)
    index_rows = [
        {"file": item.file_name, "title": item.title, "description": item.description}
        for item in items
    ]
    index_path = output_dir / f"{_safe_name(dataset_name)}_index.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    written.append(index_path)
    return written


def write_readme(output_dir: Path, dataset_name: str, items: List[ExportItem]) -> Path:
    lines = [
        f"# Report Pack - {dataset_name}",
        "",
        "This bundle contains CSV report outputs for the selected dataset.",
        "",
        "## Reports",
    ]
    for item in items:
        lines.append(f"- {item.file_name}: {item.title} - {item.description}")
    readme_path = output_dir / f"{_safe_name(dataset_name)}_README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path


def write_metadata(
    output_dir: Path,
    dataset_name: str,
    report_groups: Dict[str, List[ExportItem]],
) -> Path:
    metadata = {
        "dataset": dataset_name,
        "report_groups": {
            group: [item.file_name for item in items] for group, items in report_groups.items()
        },
    }
    meta_path = output_dir / f"{_safe_name(dataset_name)}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def write_dependency_index(
    output_dir: Path, dataset_name: str, report_groups: Dict[str, List[ExportItem]]
) -> Path:
    rows = []
    for group, items in report_groups.items():
        for item in items:
            rows.append({"report": item.file_name, "group": group})
    dep_path = output_dir / f"{_safe_name(dataset_name)}_dependencies.csv"
    pd.DataFrame(rows).to_csv(dep_path, index=False)
    return dep_path


def zip_reports(output_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zipf.write(path, path.relative_to(output_dir))
    return zip_path


def package_reports(
    base_dir: Path,
    dataset_name: str,
    report_groups: Dict[str, List[ExportItem]],
) -> Path:
    export_dir = base_dir / _safe_name(dataset_name)
    export_dir.mkdir(parents=True, exist_ok=True)
    all_items: List[ExportItem] = []
    for group_items in report_groups.values():
        all_items.extend(group_items)
    write_exports(export_dir, dataset_name, all_items)
    write_readme(export_dir, dataset_name, all_items)
    write_metadata(export_dir, dataset_name, report_groups)
    write_dependency_index(export_dir, dataset_name, report_groups)
    zip_path = base_dir / f"{_safe_name(dataset_name)}_reports.zip"
    return zip_reports(export_dir, zip_path)
