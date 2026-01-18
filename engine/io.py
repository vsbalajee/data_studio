from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import uuid
from typing import Iterable, List, Optional, Tuple

import pandas as pd
try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


@dataclass
class FileIngestResult:
    name: str
    path: Path
    size_mb: float
    rows: int
    cols: int
    encoding: Optional[str]
    file_type: str
    warnings: List[str]
    df: pd.DataFrame
    lazy_frame: Optional[object]


def bytes_to_mb(byte_count: int) -> float:
    return round(byte_count / (1024 * 1024), 2)


def detect_encoding(sample: bytes) -> Optional[str]:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return None


def validate_headers(columns: Iterable[object]) -> List[str]:
    warnings: List[str] = []
    normalized = [str(c).strip() for c in columns]
    if any(not name for name in normalized):
        raise ValueError("Missing header names detected. Please ensure all columns have names.")
    seen = set()
    dupes = set()
    for name in normalized:
        key = name.lower()
        if key in seen:
            dupes.add(name)
        seen.add(key)
    if dupes:
        dupes_list = ", ".join(sorted(dupes))
        raise ValueError(f"Duplicate header names detected: {dupes_list}")
    if any(name != str(col) for name, col in zip(normalized, columns)):
        warnings.append("Some headers were trimmed for validation purposes.")
    return warnings


def save_uploaded_file(upload, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = os.path.basename(upload.name)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    out_path = upload_dir / unique_name
    out_path.write_bytes(upload.getvalue())
    return out_path


def read_csv(path: Path, encoding: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    if not encoding:
        sample = path.read_bytes()[:4096]
        encoding = detect_encoding(sample)
    df = pd.read_csv(
        path,
        encoding=encoding or "utf-8",
        sep=None,
        engine="python",
        low_memory=False,
    )
    return df, encoding


def read_csv_lazy(path: Path, encoding: Optional[str]) -> Tuple[Optional[object], Optional[str]]:
    if pl is None:
        return None, encoding
    if not encoding:
        sample = path.read_bytes()[:4096]
        encoding = detect_encoding(sample)
    lf = pl.scan_csv(
        path,
        encoding=encoding or "utf-8",
        infer_schema_length=1000,
        ignore_errors=True,
    )
    return lf, encoding


def read_xlsx_lazy(path: Path) -> Optional[object]:
    if pl is None:
        return None
    df = pd.read_excel(path, engine="openpyxl")
    return pl.from_pandas(df).lazy()


def read_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def ingest_files(
    uploads: Iterable[object],
    upload_dir: Path,
    max_size_mb: float,
) -> List[FileIngestResult]:
    results: List[FileIngestResult] = []
    for upload in uploads:
        size_mb = bytes_to_mb(upload.size)
        if size_mb > max_size_mb:
            raise ValueError(
                f"File '{upload.name}' is {size_mb} MB. Limit is {max_size_mb} MB."
            )

        path = save_uploaded_file(upload, upload_dir)
        file_lower = upload.name.lower()
        encoding: Optional[str] = None
        lazy_frame: Optional[pl.LazyFrame] = None
        if file_lower.endswith(".csv"):
            df, encoding = read_csv(path, encoding=None)
            lazy_frame, _ = read_csv_lazy(path, encoding=encoding)
            file_type = "csv"
        elif file_lower.endswith(".xlsx"):
            df = read_xlsx(path)
            lazy_frame = read_xlsx_lazy(path)
            file_type = "xlsx"
        else:
            raise ValueError(f"Unsupported file type: {upload.name}")

        warnings = validate_headers(df.columns)
        results.append(
            FileIngestResult(
                name=upload.name,
                path=path,
                size_mb=size_mb,
                rows=len(df),
                cols=len(df.columns),
                encoding=encoding,
                file_type=file_type,
                warnings=warnings,
                df=df,
                lazy_frame=lazy_frame,
            )
        )
    return results
