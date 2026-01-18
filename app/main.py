from pathlib import Path
import json
import re
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.io import (
    FileIngestResult,
    ingest_files,
    read_csv,
    read_xlsx,
    validate_headers,
)
from engine.schema import infer_schema
from engine.entities import cluster_entities
from engine.relationships import (
    find_composite_key_candidates,
    find_key_candidates,
    find_triple_key_candidates,
    infer_relationships,
    sample_orphans,
    validate_relationship_graph,
)
from engine.semantic import infer_semantics
from reports.core import generate_core_reports
from reports.advanced import generate_advanced_reports
from reports.time_series import generate_time_reports
from reports.relationships import generate_relationship_reports
from reports.quality import generate_quality_reports
from reports.rows import generate_row_reports
from exports.package import build_export_items, package_reports
from reports.templates import build_template_reports
from reports.catalog import build_catalog
from reports.user_dynamics import (
    build_dynamic_report,
    build_lookup_report,
    build_sql_features_report,
)


APP_TITLE = "Data Report Studio"
UPLOAD_DIR = Path("data_uploads")
MAX_SIZE_MB = 200.0
EXPORT_DIR = Path("exports")
UPLOAD_MANIFEST = UPLOAD_DIR / "last_uploads.json"


st.set_page_config(page_title=APP_TITLE, layout="wide")
top_cols = st.columns([8, 2])
with top_cols[0]:
    st.title(APP_TITLE)
    st.caption("Upload data, we will scan it and prepare a report pack.")
with top_cols[1]:
    st.markdown(
        """
        <style>
          .export-cta a {
            display: inline-block;
            text-decoration: none;
            border: 1px solid #ffa94d;
            padding: 0.45rem 0.8rem;
            border-radius: 14px;
            background: linear-gradient(90deg,#fff1e6,#ffe8cc);
            color: #c05600;
            font-size: 0.85rem;
            font-weight: 600;
          }
        </style>
        <div class="export-cta" style="text-align:right; margin-top:0.35rem;">
          <a href="#export-package">Export and Package</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="export-cta" style="text-align:right; margin-top:0.4rem;">
          <a href="?view=report_list">Report List</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_header(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)

def render_banking_rules():
    render_step_header(
        "Banking Rules",
        "Default banking rules used for reports. Adjust if needed.",
    )
    dpd_bins_text = st.text_input(
        "DPD buckets (comma-separated, include 0 and 120+ upper bound)",
        value=st.session_state.get("banking_dpd_bins", "0,30,60,90,120,999999"),
        key="banking_dpd_bins",
    )
    balance_bands_text = st.text_area(
        "Balance bands (label:min:max per line)",
        value=st.session_state.get(
            "banking_balance_bands",
            "0:0:0\n1-10k:1:10000\n10k-50k:10001:50000\n50k-100k:50001:100000\n100k-500k:100001:500000\n500k-1m:500001:1000000\n1m+:1000001:999999999",
        ),
        key="banking_balance_bands",
    )
    savings_types = st.text_input(
        "CASA Savings types (comma-separated)",
        value=st.session_state.get("banking_casa_savings", "savings,sa,sv"),
        key="banking_casa_savings",
    )
    current_types = st.text_input(
        "CASA Current types (comma-separated)",
        value=st.session_state.get("banking_casa_current", "current,ca,cc"),
        key="banking_casa_current",
    )
    large_txn_threshold_text = st.text_input(
        "Large transaction threshold",
        value=st.session_state.get("banking_large_txn_threshold", "100000"),
        key="banking_large_txn_threshold",
    )
    round_amount_base_text = st.text_input(
        "Round amount base",
        value=st.session_state.get("banking_round_amount_base", "1000"),
        key="banking_round_amount_base",
    )
    dpd_bins = [int(x.strip()) for x in dpd_bins_text.split(",") if x.strip().isdigit()]
    dpd_labels = []
    if dpd_bins and dpd_bins[0] == 0:
        dpd_labels.append("0")
        for idx in range(1, len(dpd_bins) - 1):
            dpd_labels.append(f"{dpd_bins[idx-1]+1}-{dpd_bins[idx]}")
        dpd_labels.append(f"{dpd_bins[-2]+1}+")
    balance_bands = []
    for line in balance_bands_text.splitlines():
        parts = [p.strip() for p in line.split(":")]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            balance_bands.append({"label": parts[0], "min": int(parts[1]), "max": int(parts[2])})
    rules = {
        "dpd_bins": [-1] + dpd_bins if dpd_bins and dpd_bins[0] == 0 else None,
        "dpd_labels": dpd_labels if dpd_labels else None,
        "balance_bands": balance_bands if balance_bands else None,
        "casa_savings_values": [v.strip() for v in savings_types.split(",") if v.strip()],
        "casa_current_values": [v.strip() for v in current_types.split(",") if v.strip()],
        "large_txn_threshold": int(large_txn_threshold_text) if large_txn_threshold_text.isdigit() else 100000,
        "round_amount_base": int(round_amount_base_text) if round_amount_base_text.isdigit() else 1000,
    }
    st.session_state["banking_rules"] = rules
    return rules


@st.cache_data(show_spinner=False)
def cached_infer_schema(df):
    return infer_schema(df)


@st.cache_data(show_spinner=False)
def cached_core_reports(df):
    return generate_core_reports(df)


@st.cache_data(show_spinner=False)
def cached_advanced_reports(df, mode):
    return generate_advanced_reports(df, mode)


@st.cache_data(show_spinner=False)
def cached_time_reports(df):
    return generate_time_reports(df)


@st.cache_data(show_spinner=False)
def cached_quality_reports(df):
    return generate_quality_reports(df)


@st.cache_data(show_spinner=False)
def cached_row_reports(df):
    return generate_row_reports(df)


@st.cache_data(show_spinner=False)
def cached_key_candidates(df):
    return find_key_candidates(df)


@st.cache_data(show_spinner=False)
def cached_composite_key_candidates(df):
    return find_composite_key_candidates(df)


@st.cache_data(show_spinner=False)
def cached_relationship_validation(dfs, schema_map):
    return validate_relationship_graph(dfs, schema_map)


@st.cache_data(show_spinner=False)
def cached_triple_key_candidates(df):
    return find_triple_key_candidates(df)


@st.cache_data(show_spinner=False)
def cached_semantic_inference(df):
    return infer_semantics(df)


@st.cache_data(show_spinner=False)
def cached_semantic_inference_by_industry(df, industry):
    return infer_semantics(df, industry=industry)


@st.cache_data(show_spinner=False)
def cached_semantic_inference_by_domain(df, industry, sub_domain):
    return infer_semantics(df, industry=industry, sub_domain=sub_domain)


@st.cache_data(show_spinner=False)
def cached_template_reports(df, semantics, rules):
    return build_template_reports(df, semantics, rules)


@st.cache_data(show_spinner=False)
def cached_catalog(report_type, reports):
    return build_catalog(report_type, reports)


def make_anchor(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "report"


def report_category(report, group):
    if group == "core" and report.title == "Dataset Overview":
        return "Executive"
    if group == "core":
        return "Basic"
    if group == "templates":
        return "Simple"
    if group == "advanced":
        return "Intermediate"
    if group == "time":
        return "Advanced"
    if group == "relationships":
        return "Expert"
    if group in ("quality", "rows"):
        return "Forensics"
    return "Basic"


def build_report_index(report_groups, dataset_name, min_confidence, search_text):
    rows = []
    serial = 1
    for group, reports in report_groups.items():
        for report in reports:
            category = report_category(report, group)
            confidence = getattr(report, "confidence", 0.5)
            title = report.title
            if search_text and search_text.lower() not in title.lower():
                continue
            if confidence < min_confidence:
                continue
            if group == "relationships":
                anchor = make_anchor(f"relationships-{title}")
            else:
                anchor = make_anchor(f"{dataset_name}-{group}-{title}")
            rows.append(
                {
                    "s_no": serial,
                    "title": title,
                    "anchor": anchor,
                    "category": category,
                    "confidence": confidence,
                    "why": report.description,
                }
            )
            serial += 1
    return rows


def category_order():
    return [
        "Executive",
        "Basic",
        "Simple",
        "Intermediate",
        "Advanced",
        "Expert",
        "Forensics",
    ]


def build_report_list_catalog() -> dict:
    return {
        "Executive": [
            "Dataset Overview",
        ],
        "Basic": [
            "Column Profile Summary",
            "Missing Values",
            "Duplicate Rows",
            "Numeric Summary",
            "Top Categories",
        ],
        "Simple": [
            "Category Count",
            "Amount by Category",
            "Quantity by Category",
            "Average Rate Metrics",
            "Channel Mix",
            "Delinquency Buckets (DPD)",
            "Portfolio at Risk (PAR30/60/90)",
            "NPA Proxy (DPD 90+)",
            "Loan Vintage / Seasoning Table",
            "Collections Roll Rate Matrix",
            "Collections Cure Rate",
            "Balance Summary",
            "Balance Aging (Monthly Average)",
            "Loan Portfolio Summary",
            "Fee and Penalty Analysis",
            "Interest Income vs Outstanding",
            "Risk Grade Distribution",
            "Expected Loss Summary (PD * LGD * EAD)",
            "Deposit Account Type Mix",
            "Deposit Balance by Account Type",
            "CASA Mix",
            "Deposit Balance Bands",
            "Average Daily Balance",
            "Branch Volume Summary",
            "Branch Outstanding Summary",
            "Relationship Manager Portfolio",
            "Relationship Manager Outstanding",
            "Credit Utilization Summary",
            "Credit Utilization Bands",
            "Large Transaction Summary",
            "Round Amount Frequency",
            "Transaction Velocity (Daily)",
        ],
        "Intermediate": [
            "Pareto Summary",
            "Distribution Buckets",
            "Skewness",
            "Mean vs Median",
            "Correlation Matrix (full mode)",
            "Outlier Buckets (full mode)",
        ],
        "Advanced": [
            "Daily Trend Counts",
            "Monthly Trend Counts",
            "Period-over-Period Changes",
        ],
        "Expert": [
            "Join-based Rollups",
        ],
        "Forensics": [
            "IQR Outlier Summary",
            "MAD Outlier Summary",
            "Label Variants",
            "Numeric Drift",
            "Category Drift",
            "Time Series Spikes",
            "Invalid Emails",
            "Invalid Phones",
            "Rows with Most Missing Values",
            "Extreme Numeric Rows",
            "Rare Pattern Rows",
        ],
    }


def render_report_list():
    report_catalog = build_report_list_catalog()
    render_step_header("Report List", "All report categories and templates.")
    st.markdown('<a id="report-list"></a>', unsafe_allow_html=True)
    st.info("No AI is involved in report generation.")
    st.info("Works well on 8GB laptops for typical banking datasets.")
    st.info("Reports are subject to data quality and available columns.")
    st.info("Deterministic output: same data produces the same results.")
    st.info("Multiple sheets prompt for joins before reports.")
    st.info("Counts below show maximum available report types; actual counts vary by data and columns.")
    summary_rows = []
    total_reports = 0
    for idx, category in enumerate(category_order(), start=1):
        count = len(report_catalog.get(category, []))
        total_reports += count
        summary_rows.append(
            f"<tr><td>{idx}</td><td>{category}</td><td>{count}</td></tr>"
        )
    summary_rows.append(
        f"<tr><td></td><td><strong>Total</strong></td><td><strong>{total_reports}</strong></td></tr>"
    )
    st.markdown(
        "<table>"
        "<tr><th>Sl. No.</th><th>Report Category</th><th>No. of Reports (max)</th></tr>"
        + "".join(summary_rows)
        + "</table>",
        unsafe_allow_html=True,
    )
    extra_rows = [
        ("User Dynamics", "Variable (user-defined)"),
        ("Lookup", "Variable (user-defined)"),
        ("SQL Features", "Variable (user-defined)"),
    ]
    st.markdown(
        "<table>"
        "<tr><th>Sl. No.</th><th>Report Category</th><th>No. of Reports (max)</th></tr>"
        + "".join(
            [
                f"<tr><td>{idx}</td><td>{name}</td><td>{count}</td></tr>"
                for idx, (name, count) in enumerate(extra_rows, start=len(summary_rows) + 1)
            ]
        )
        + "</table>",
        unsafe_allow_html=True,
    )
    for category in category_order():
        st.markdown(f"**{category}**")
        items = report_catalog.get(category, [])
        if items:
            st.markdown("\n".join([f"- {item}" for item in items]))
        else:
            st.markdown("- No reports defined.")
    st.markdown("**User Dynamics**")
    st.markdown("- User-defined (dynamic, depends on selected fields)")
    st.markdown("**Lookup**")
    st.markdown("- User-defined (depends on datasets and join keys)")
    st.markdown("**SQL Features**")
    st.markdown("- User-defined (depends on query selections)")
    st.markdown(
        '<a href="?view=main">Back to Main Page</a>',
        unsafe_allow_html=True,
    )


def normalize_col(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def suggest_join_columns(results):
    if len(results) < 2:
        return [], []
    col_maps = []
    for res in results:
        col_maps.append({normalize_col(c): c for c in res.df.columns})
    common_norm = set(col_maps[0].keys())
    for cmap in col_maps[1:]:
        common_norm &= set(cmap.keys())
    common = sorted({col_maps[0][n] for n in common_norm})
    # fallback: near matches (normalized overlap)
    all_norm = {}
    for cmap in col_maps:
        for n, orig in cmap.items():
            all_norm.setdefault(n, []).append(orig)
    near = sorted({names[0] for n, names in all_norm.items() if len(names) >= 2})
    return common, near


def join_preview(results, join_col, how):
    if any(join_col not in res.df.columns for res in results):
        return results[0].df.head(0)
    merged = results[0].df
    for res in results[1:]:
        merged = merged.merge(res.df, on=join_col, how=how, suffixes=("", "_r"))
    return merged.head(5)


def join_confidence(left_df, right_df, join_col):
    if join_col not in left_df.columns or join_col not in right_df.columns:
        return 0.0, 1.0
    left_vals = set(left_df[join_col].dropna().astype(str).unique())
    right_vals = set(right_df[join_col].dropna().astype(str).unique())
    if not left_vals or not right_vals:
        return 0.0, 1.0
    overlap = len(left_vals & right_vals) / max(len(left_vals), 1)
    orphan_rate = 1 - overlap
    return round(overlap, 2), round(orphan_rate, 2)

def get_selected_report_key():
    params = st.query_params
    if params.get("view") == "abstract":
        return None
    report_key = params.get("report")
    return report_key

def set_report_query(report_key=None):
    if report_key:
        st.query_params["report"] = report_key
    else:
        st.query_params.clear()

def get_view():
    return st.query_params.get("view")


def save_upload_manifest(paths):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_MANIFEST.write_text(json.dumps(paths, indent=2), encoding="utf-8")


def load_upload_manifest():
    if UPLOAD_MANIFEST.exists():
        try:
            return json.loads(UPLOAD_MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []

if get_view() == "report_list":
    render_report_list()
    st.stop()

render_step_header(
    "Step 1: Upload data",
    "Supported files: CSV or XLSX. You can upload multiple files together.",
)
industry_choice = st.selectbox(
    "Industry/Vertical",
    ["General", "Banking", "Healthcare", "Finance", "Retail", "Manufacturing"],
    index=0,
)
st.session_state["industry"] = industry_choice
file_mode = st.radio(
    "Upload mode",
    ["Single file", "Multiple files"],
    horizontal=True,
)
uploads = st.file_uploader(
    "Upload data files",
    type=["csv", "xlsx"],
    accept_multiple_files=(file_mode == "Multiple files"),
)

results = []
errors = []

if uploads:
    upload_items = uploads if isinstance(uploads, list) else [uploads]
    st.info("Reading files and checking headers...")
    progress = st.progress(0)
    step = 1.0 / max(len(upload_items), 1)
    for idx, upload in enumerate(upload_items, start=1):
        try:
            result = ingest_files([upload], UPLOAD_DIR, MAX_SIZE_MB)[0]
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - show user-friendly error
            errors.append(f"{upload.name}: {exc}")
        progress.progress(min(1.0, idx * step))

    if results:
        st.session_state["uploaded_data"] = results
        st.session_state["uploaded_paths"] = [str(r.path) for r in results]
        save_upload_manifest(st.session_state["uploaded_paths"])
elif st.session_state.get("uploaded_paths") or get_selected_report_key() or st.query_params.get("view") == "abstract":
    cached_paths = st.session_state.get("uploaded_paths") or load_upload_manifest()
    for path_str in cached_paths:
        path = Path(path_str)
        if not path.exists():
            errors.append(f"Missing file on disk: {path}")
            continue
        name = path.name
        size_mb = round(path.stat().st_size / (1024 * 1024), 2)
        if name.lower().endswith(".csv"):
            df, encoding = read_csv(path, encoding=None)
            file_type = "csv"
        elif name.lower().endswith(".xlsx"):
            df = read_xlsx(path)
            encoding = None
            file_type = "xlsx"
        else:
            errors.append(f"Unsupported file type: {name}")
            continue
        warnings = validate_headers(df.columns)
        results.append(
            FileIngestResult(
                name=name,
                path=path,
                size_mb=size_mb,
                rows=len(df),
                cols=len(df.columns),
                encoding=encoding,
                file_type=file_type,
                warnings=warnings,
                df=df,
                lazy_frame=None,
            )
        )

if errors:
    st.error("Some files could not be read.")
    for err in errors:
        st.write(f"- {err}")

if not results:
    st.warning("No files uploaded yet.")
    if st.button("Load last upload"):
        st.session_state["uploaded_paths"] = load_upload_manifest()
        st.rerun()
    st.stop()

if file_mode == "Single file":
    render_step_header("Upload Summary", "Click a file to preview data.")
    for result in results:
        st.markdown(f"**{result.name}**")
        st.write(
            f"Rows: {result.rows} | Columns: {result.cols} | Size: {result.size_mb} MB"
        )
        if result.encoding:
            st.write(f"Detected encoding: {result.encoding}")
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)
        with st.expander("Preview data"):
            st.dataframe(result.df.head(50), use_container_width=True)
else:
    render_step_header("Uploaded Files", "File names only.")
    for result in results:
        st.markdown(f"- {result.name}")

industry_value = st.session_state.get("industry") if st.session_state.get("industry") != "General" else None
sub_domain_value = None
banking_rules = st.session_state.get("banking_rules", {})
if file_mode == "Single file":
    render_step_header(
        "Step 2: Understand column types",
        "We suggest data types and roles. You can adjust if needed.",
    )
    schema_overrides = st.session_state.get("schema_overrides", {})
    updated_overrides = {}
    for result in results:
        st.markdown(f"**{result.name}**")
        inferences = cached_infer_schema(result.df)
        for inf in inferences:
            key = f"{result.name}:{inf.column}"
            default_type = schema_overrides.get(key, inf.inferred_type)
            default_role = schema_overrides.get(key + ":role", inf.semantic_role)
            cols = st.columns([3, 2, 2, 2, 3])
            cols[0].write(inf.column)
            cols[1].write(inf.inferred_type)
            cols[2].write(f"{inf.confidence:.2f}")
            new_type = cols[3].selectbox(
                "Type",
                ["text", "number", "date", "boolean", "id"],
                index=["text", "number", "date", "boolean", "id"].index(default_type),
                key=f"type:{key}",
            )
            new_role = cols[4].selectbox(
                "Role",
                ["attribute", "id", "date", "amount", "status", "category", "email", "phone", "name", "country", "city"],
                index=[
                    "attribute",
                    "id",
                    "date",
                    "amount",
                    "status",
                    "category",
                    "email",
                    "phone",
                    "name",
                    "country",
                    "city",
                ].index(default_role),
                key=f"role:{key}",
            )
            updated_overrides[key] = new_type
            updated_overrides[key + ":role"] = new_role

    st.session_state["schema_overrides"] = updated_overrides
    st.success("Schema suggestions ready. You can adjust selections anytime.")

    render_step_header(
        "Step 3: Group columns into entities",
        "We group related columns that look like the same entity.",
    )
    for result in results:
        st.markdown(f"**{result.name}**")
        inferences = infer_schema(result.df)
        entities = cluster_entities(inferences)
        state_key = f"entities:{result.name}"
        if state_key not in st.session_state:
            st.session_state[state_key] = [
                {"name": e.name, "columns": e.columns} for e in entities
            ]
        updated_entities = []
        for idx, entity in enumerate(st.session_state[state_key], start=1):
            col1, col2 = st.columns([2, 5])
            name = col1.text_input(
                f"Entity {idx} name", value=entity["name"], key=f"entity_name:{result.name}:{idx}"
            )
            cols = col2.multiselect(
                f"Entity {idx} columns",
                options=list(result.df.columns),
                default=entity["columns"],
                key=f"entity_cols:{result.name}:{idx}",
            )
            updated_entities.append({"name": name, "columns": cols})
        if st.button(f"Add entity for {result.name}", key=f"add_entity:{result.name}"):
            updated_entities.append({"name": "New Entity", "columns": []})
        st.session_state[state_key] = updated_entities

    render_step_header(
        "Step 3b: Semantic modeling",
        "We infer business meaning, metric types, and units.",
    )
    if industry_value == "Banking":
        sub_domain = st.selectbox(
            "Banking sub-domain (optional)",
            ["None", "Retail Banking", "Lending", "Cards", "Payments"],
            index=0,
            key="sub_domain_select",
        )
        sub_domain_value = None if sub_domain == "None" else sub_domain
    for result in results:
        st.markdown(f"**{result.name}**")
        semantic = cached_semantic_inference_by_domain(
            result.df, industry_value, sub_domain_value
        )
        for inf in semantic:
            st.write(
                f"{inf.column} | role {inf.semantic_role} | "
                f"entity {inf.entity or 'n/a'} | metric {inf.metric_type} | "
                f"unit {inf.unit or 'n/a'} | confidence {inf.confidence:.2f}"
            )
    banking_rules = render_banking_rules()

if file_mode == "Multiple files" and len(results) > 1:
    banking_rules = render_banking_rules()

schema_map = {res.name: cached_infer_schema(res.df) for res in results}
if len(results) > 1:
    dfs = {res.name: res.df for res in results}
    relationships = infer_relationships(dfs, schema_map)
else:
    relationships = {}

default_flags = {
    "run_advanced": True,
    "run_time": True,
    "run_quality": True,
    "run_rows": True,
    "run_relationship_reports": True,
    "advanced_mode": "fast",
}
for key, value in default_flags.items():
    if key not in st.session_state:
        st.session_state[key] = value

join_confirmed = st.session_state.get("join_confirmed", False)
joined_df = None
joined_name = "Joined Dataset"
if len(results) > 1 and file_mode == "Multiple files":
    render_step_header("Join Suggestions", "Confirm joins before reports.")
    common_cols, near_cols = suggest_join_columns(results)
    all_cols = set(results[0].df.columns)
    for res in results[1:]:
        all_cols &= set(res.df.columns)
    candidate_cols = sorted(all_cols) if all_cols else []
    if candidate_cols:
        if common_cols:
            st.caption("Exact column matches detected across files.")
        else:
            st.warning(
                "No exact column matches found. Showing best candidates. "
                "If these are incorrect, reupload corrected files."
            )
        default_join = st.session_state.get("join_key")
        default_index = (
            candidate_cols.index(default_join)
            if default_join in candidate_cols
            else 0
        )
        join_col = st.selectbox("Join column", options=candidate_cols, index=default_index)
        join_type = st.selectbox(
            "Join type",
            ["inner", "left"],
            index=0 if st.session_state.get("join_type") not in ["inner", "left"] else ["inner", "left"].index(st.session_state.get("join_type")),
        )
        overlap, orphan_rate = join_confidence(results[0].df, results[1].df, join_col)
        st.write(f"Join confidence: {overlap:.2f} | Orphan rate: {orphan_rate:.2f}")
        st.markdown("**Preview (first 5 rows)**")
        st.dataframe(join_preview(results, join_col, join_type), use_container_width=True)
        join_confirmed = st.checkbox(
            "Confirm join and build unified reports",
            value=join_confirmed,
        )
        if join_confirmed:
            joined_df = results[0].df
            for res in results[1:]:
                joined_df = joined_df.merge(
                    res.df, on=join_col, how=join_type, suffixes=("", "_r")
                )
            st.session_state["join_key"] = join_col
            st.session_state["join_type"] = join_type
            st.session_state["join_confirmed"] = True
        else:
            st.session_state["join_confirmed"] = False
    else:
        st.warning(
            "No join column exists in every file. "
            "Please review headers and reupload corrected files."
        )
        st.session_state["join_confirmed"] = False
        join_confirmed = False
        if near_cols:
            st.caption("Nearest joiner column suggestions (not in every file):")
            suggestions = []
            for cand in near_cols:
                presence = []
                for res in results:
                    present = "yes" if cand in res.df.columns else "no"
                    presence.append(f"{res.name}: {present}")
                suggestions.append({"candidate": cand, "presence": ", ".join(presence)})
            st.dataframe(suggestions, use_container_width=True)
    if not join_confirmed:
        st.info("Confirm joins to generate the unified report pack.")
        st.stop()

render_step_header("Report Abstract", "Quick links to available reports.")
st.markdown('<a id="report-abstract"></a>', unsafe_allow_html=True)
category_filter = st.selectbox(
    "View category",
    ["All"] + category_order(),
    index=0,
)
search_text = st.text_input("Search reports", value="")
high_conf_only = st.checkbox("Show only high-confidence reports", value=False)
min_confidence = 0.75 if high_conf_only else 0.0
selected_report_key = get_selected_report_key()
if selected_report_key:
    render_step_header("Report View", "Selected report details.")
    st.markdown(
        '<a href="?view=abstract#report-abstract">Back to Report Abstract</a>',
        unsafe_allow_html=True,
    )
    report_sources = [(joined_name, joined_df)] if joined_df is not None else [(r.name, r.df) for r in results]
    for name, df in report_sources:
        report_groups = {}
        report_groups["core"] = cached_core_reports(df)
        report_groups["templates"] = cached_template_reports(
            df,
            cached_semantic_inference_by_domain(
                df, industry_value, sub_domain_value
            ),
            banking_rules,
        )
        if st.session_state.get("run_advanced", False):
            report_groups["advanced"] = cached_advanced_reports(
                df, st.session_state.get("advanced_mode", "fast")
            )
        if st.session_state.get("run_time", False):
            report_groups["time"] = cached_time_reports(df)
        if st.session_state.get("run_quality", False):
            report_groups["quality"] = cached_quality_reports(df)
        if st.session_state.get("run_rows", False):
            report_groups["rows"] = cached_row_reports(df)
        if st.session_state.get("run_relationship_reports", False):
            report_groups["relationships"] = generate_relationship_reports(
                {res.name: res.df for res in results}, relationships
            )
        for group, reports in report_groups.items():
            for report in reports:
                report_anchor = make_anchor(
                    f"{name}-{group}-{report.title}"
                ) if group != "relationships" else make_anchor(
                    f"relationships-{report.title}"
                )
                if report_anchor == selected_report_key:
                    st.markdown(f"**{report.title}**")
                    st.caption(report.description)
                    st.dataframe(report.data, use_container_width=True)
                    st.stop()
report_sources = [(joined_name, joined_df)] if joined_df is not None else [(r.name, r.df) for r in results]
for name, df in report_sources:
    st.markdown(f"**{name}**")
    report_groups = {}
    report_groups["core"] = cached_core_reports(df)
    report_groups["templates"] = cached_template_reports(
        df,
        cached_semantic_inference_by_domain(
            df, industry_value, sub_domain_value
        ),
        banking_rules,
    )
    if st.session_state.get("run_advanced", False):
        report_groups["advanced"] = cached_advanced_reports(
            df, st.session_state.get("advanced_mode", "fast")
        )
    if st.session_state.get("run_time", False):
        report_groups["time"] = cached_time_reports(df)
    if st.session_state.get("run_quality", False):
        report_groups["quality"] = cached_quality_reports(df)
    if st.session_state.get("run_rows", False):
        report_groups["rows"] = cached_row_reports(df)
    if st.session_state.get("run_relationship_reports", False):
        report_groups["relationships"] = generate_relationship_reports(
            {res.name: res.df for res in results}, relationships
        )
    index_rows = build_report_index(
        report_groups, name, min_confidence, search_text
    )
    if index_rows:
        category_counts = {cat: 0 for cat in category_order()}
        for row in index_rows:
            category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
        st.write(
            "Report counts: "
            + ", ".join([f"{k}: {v}" for k, v in category_counts.items()])
        )
        for category in category_order():
            if category_filter != "All" and category_filter != category:
                continue
            st.markdown(f"**{category}**")
            rows_html = []
            rows_html.append("<table>")
            rows_html.append(
                "<tr><th>S.No</th><th>Report</th><th>Confidence</th><th>Why</th></tr>"
            )
            category_rows = [r for r in index_rows if r["category"] == category]
            for row in category_rows:
                rows_html.append(
                    "<tr>"
                    f"<td>{row['s_no']}</td>"
                f"<td><a href=\"?report={row['anchor']}#report-abstract\">{row['title']}</a></td>"
                    f"<td>{row['confidence']:.2f}</td>"
                    f"<td>{row['why']}</td>"
                    "</tr>"
                )
            if not category_rows:
                rows_html.append(
                    "<tr><td colspan=\"4\">No reports in this category.</td></tr>"
                )
            rows_html.append("</table>")
            st.markdown("".join(rows_html), unsafe_allow_html=True)

render_step_header("User Dynamics", "Build custom reports from selected columns.")
selected_file = st.selectbox(
    "Choose dataset",
    [r.name for r in results] + (["Joined Dataset"] if joined_df is not None else []),
    index=0,
    key="user_dyn_file",
)
active_df = joined_df if (joined_df is not None and selected_file == "Joined Dataset") else next(r.df for r in results if r.name == selected_file)
all_columns = list(active_df.columns)
numeric_columns = list(active_df.select_dtypes(include="number").columns)

dimensions = st.multiselect(
    "Dimensions (group by)",
    options=all_columns,
    key="user_dyn_dims",
)
metric = st.selectbox(
    "Metric (optional)",
    options=["(count)"] + numeric_columns,
    index=0,
    key="user_dyn_metric",
)
agg = st.selectbox(
    "Aggregation",
    options=["count", "sum", "avg", "min", "max"],
    index=0,
    key="user_dyn_agg",
)
top_n = st.number_input("Top N (optional)", min_value=0, value=0, step=1, key="user_dyn_top")
percent_of_total = st.checkbox("Include percent of total", value=False, key="user_dyn_pct")

st.markdown("**Filters**")
cat_filter_col = st.selectbox(
    "Categorical filter column",
    options=["(none)"] + all_columns,
    index=0,
    key="user_dyn_cat_col",
)
cat_filter_values = []
if cat_filter_col != "(none)":
    unique_vals = active_df[cat_filter_col].dropna().astype(str).unique().tolist()
    cat_filter_values = st.multiselect("Filter values", options=unique_vals, key="user_dyn_cat_vals")

num_filter_col = st.selectbox(
    "Numeric filter column",
    options=["(none)"] + numeric_columns,
    index=0,
    key="user_dyn_num_col",
)
num_min = None
num_max = None
if num_filter_col != "(none)":
    num_min = st.number_input("Min value", value=0.0, step=1.0, key="user_dyn_num_min")
    num_max = st.number_input("Max value", value=0.0, step=1.0, key="user_dyn_num_max")

if st.button("Generate user dynamic report", key="user_dyn_run"):
    metric_value = None if metric == "(count)" else metric
    categorical_filters = {}
    if cat_filter_col != "(none)":
        categorical_filters[cat_filter_col] = cat_filter_values
    numeric_filters = {}
    if num_filter_col != "(none)":
        numeric_filters[num_filter_col] = (num_min, num_max)
    report = build_dynamic_report(
        active_df,
        dimensions=dimensions,
        metric=metric_value,
        agg=agg,
        top_n=top_n if top_n > 0 else None,
        percent_of_total=percent_of_total,
        categorical_filters=categorical_filters,
        numeric_filters=numeric_filters,
    )
    st.markdown(f"**{report.title}**")
    st.caption(report.description)
    st.dataframe(report.data, use_container_width=True)

st.markdown("**Lookup Report**")
left_file = st.selectbox(
    "Left dataset",
    [r.name for r in results],
    index=0,
    key="lookup_left_file",
)
right_file = st.selectbox(
    "Right dataset",
    [r.name for r in results],
    index=0,
    key="lookup_right_file",
)
left_df = next(r.df for r in results if r.name == left_file)
right_df = next(r.df for r in results if r.name == right_file)
left_key = st.selectbox(
    "Left key column",
    options=list(left_df.columns),
    index=list(left_df.columns).index(st.session_state.get("join_key", left_df.columns[0])) if st.session_state.get("join_key") in list(left_df.columns) else 0,
    key="lookup_left_key",
)
right_key = st.selectbox(
    "Right key column",
    options=list(right_df.columns),
    index=list(right_df.columns).index(st.session_state.get("join_key", right_df.columns[0])) if st.session_state.get("join_key") in list(right_df.columns) else 0,
    key="lookup_right_key",
)
return_cols = st.multiselect(
    "Return columns (from right dataset)",
    options=list(right_df.columns),
    key="lookup_return_cols",
)
if st.button("Run lookup", key="lookup_run"):
    report = build_lookup_report(
        left_df=left_df,
        right_df=right_df,
        left_key=left_key,
        right_key=right_key,
        return_cols=return_cols,
    )
    st.markdown(f"**{report.title}**")
    st.caption(report.description)
    st.dataframe(report.data, use_container_width=True)

render_step_header("SQL Features", "SQL-style report builder.")
sql_file = st.selectbox(
    "SQL dataset",
    [r.name for r in results] + (["Joined Dataset"] if joined_df is not None else []),
    index=0,
    key="sql_file",
)
sql_df = (
    joined_df
    if (joined_df is not None and sql_file == "Joined Dataset")
    else next(r.df for r in results if r.name == sql_file)
)
sql_columns = list(sql_df.columns)

select_columns = st.multiselect(
    "Select columns",
    options=sql_columns,
    key="sql_select_columns",
)

alias_rows = [{"column": col, "alias": ""} for col in select_columns]
alias_editor = st.data_editor(
    alias_rows,
    key="sql_aliases",
    hide_index=True,
    column_config={
        "column": st.column_config.TextColumn("Column"),
        "alias": st.column_config.TextColumn("Alias"),
    },
)
alias_map = {
    row["column"]: row.get("alias", "")
    for row in alias_editor
    if row.get("column")
}

st.markdown("**Filters (WHERE)**")
if "sql_filters" not in st.session_state:
    st.session_state["sql_filters"] = []
if st.button("Add filter", key="sql_add_filter"):
    st.session_state["sql_filters"].append(
        {"column": "", "operator": "=", "value": ""}
    )
filter_mode = st.radio(
    "Filter mode",
    ["AND", "OR"],
    horizontal=True,
    key="sql_filter_mode",
)
filters = []
for idx, rule in enumerate(st.session_state["sql_filters"]):
    cols = st.columns([3, 2, 3, 1])
    col_value = cols[0].selectbox(
        "Column",
        options=[""] + sql_columns,
        index=0 if rule["column"] == "" else ([""] + sql_columns).index(rule["column"]),
        key=f"sql_filter_col_{idx}",
    )
    op_value = cols[1].selectbox(
        "Operator",
        options=["=", "!=", ">", ">=", "<", "<=", "contains", "starts_with", "ends_with", "is_null", "not_null"],
        index=0,
        key=f"sql_filter_op_{idx}",
    )
    val_value = cols[2].text_input(
        "Value",
        value=str(rule.get("value", "")),
        key=f"sql_filter_val_{idx}",
    )
    if cols[3].button("Remove", key=f"sql_filter_remove_{idx}"):
        st.session_state["sql_filters"].pop(idx)
        st.rerun()
    if col_value:
        filters.append({"column": col_value, "operator": op_value, "value": val_value})

st.markdown("**Calculated Fields**")
if "sql_calculated" not in st.session_state:
    st.session_state["sql_calculated"] = []
if st.button("Add calculated field", key="sql_add_calc"):
    st.session_state["sql_calculated"].append(
        {"name": "", "left": "", "operator": "+", "right": ""}
    )
calculated_fields = []
for idx, rule in enumerate(st.session_state["sql_calculated"]):
    cols = st.columns([2, 3, 2, 3, 1])
    name_value = cols[0].text_input(
        "Name",
        value=rule.get("name", ""),
        key=f"sql_calc_name_{idx}",
    )
    left_value = cols[1].selectbox(
        "Left",
        options=[""] + sql_columns,
        index=0 if rule.get("left") == "" else ([""] + sql_columns).index(rule.get("left")),
        key=f"sql_calc_left_{idx}",
    )
    op_value = cols[2].selectbox(
        "Op",
        options=["+", "-", "*", "/", "concat"],
        index=0,
        key=f"sql_calc_op_{idx}",
    )
    right_value = cols[3].text_input(
        "Right (column or constant)",
        value=rule.get("right", ""),
        key=f"sql_calc_right_{idx}",
    )
    if cols[4].button("Remove", key=f"sql_calc_remove_{idx}"):
        st.session_state["sql_calculated"].pop(idx)
        st.rerun()
    if name_value and left_value:
        calculated_fields.append(
            {
                "name": name_value,
                "left": left_value,
                "operator": op_value,
                "right": right_value,
            }
        )

st.markdown("**CASE / Conditional Fields**")
if "sql_case_fields" not in st.session_state:
    st.session_state["sql_case_fields"] = []
if st.button("Add CASE field", key="sql_add_case"):
    st.session_state["sql_case_fields"].append(
        {"name": "", "column": "", "operator": "=", "value": "", "then": "", "else": ""}
    )
case_fields = []
for idx, rule in enumerate(st.session_state["sql_case_fields"]):
    cols = st.columns([2, 3, 2, 2, 2, 2, 1])
    name_value = cols[0].text_input(
        "Name",
        value=rule.get("name", ""),
        key=f"sql_case_name_{idx}",
    )
    col_value = cols[1].selectbox(
        "Column",
        options=[""] + sql_columns,
        index=0 if rule.get("column") == "" else ([""] + sql_columns).index(rule.get("column")),
        key=f"sql_case_col_{idx}",
    )
    op_value = cols[2].selectbox(
        "Operator",
        options=["=", "!=", ">", ">=", "<", "<=", "contains", "starts_with", "ends_with"],
        index=0,
        key=f"sql_case_op_{idx}",
    )
    val_value = cols[3].text_input(
        "Value",
        value=rule.get("value", ""),
        key=f"sql_case_val_{idx}",
    )
    then_value = cols[4].text_input(
        "Then",
        value=rule.get("then", ""),
        key=f"sql_case_then_{idx}",
    )
    else_value = cols[5].text_input(
        "Else",
        value=rule.get("else", ""),
        key=f"sql_case_else_{idx}",
    )
    if cols[6].button("Remove", key=f"sql_case_remove_{idx}"):
        st.session_state["sql_case_fields"].pop(idx)
        st.rerun()
    if name_value and col_value:
        case_fields.append(
            {
                "name": name_value,
                "column": col_value,
                "operator": op_value,
                "value": val_value,
                "then": then_value,
                "else": else_value,
            }
        )

st.markdown("**Date Extracts**")
if "sql_date_extracts" not in st.session_state:
    st.session_state["sql_date_extracts"] = []
if st.button("Add date extract", key="sql_add_date_extract"):
    st.session_state["sql_date_extracts"].append(
        {"name": "", "column": "", "part": "year"}
    )
date_extracts = []
for idx, rule in enumerate(st.session_state["sql_date_extracts"]):
    cols = st.columns([2, 3, 2, 1])
    name_value = cols[0].text_input(
        "Name",
        value=rule.get("name", ""),
        key=f"sql_date_name_{idx}",
    )
    col_value = cols[1].selectbox(
        "Date column",
        options=[""] + sql_columns,
        index=0 if rule.get("column") == "" else ([""] + sql_columns).index(rule.get("column")),
        key=f"sql_date_col_{idx}",
    )
    part_value = cols[2].selectbox(
        "Part",
        options=["year", "month", "day"],
        index=0,
        key=f"sql_date_part_{idx}",
    )
    if cols[3].button("Remove", key=f"sql_date_remove_{idx}"):
        st.session_state["sql_date_extracts"].pop(idx)
        st.rerun()
    if name_value and col_value:
        date_extracts.append(
            {"name": name_value, "column": col_value, "part": part_value}
        )

st.markdown("**Group By & Aggregations**")
group_by_cols = st.multiselect(
    "Group by columns",
    options=sql_columns,
    key="sql_group_by",
)
if "sql_aggs" not in st.session_state:
    st.session_state["sql_aggs"] = []
if st.button("Add aggregation", key="sql_add_agg"):
    st.session_state["sql_aggs"].append({"column": "", "agg": "sum"})
aggregations = []
for idx, rule in enumerate(st.session_state["sql_aggs"]):
    cols = st.columns([4, 3, 1])
    col_value = cols[0].selectbox(
        "Agg column",
        options=[""] + sql_columns,
        index=0 if rule.get("column") == "" else ([""] + sql_columns).index(rule.get("column")),
        key=f"sql_agg_col_{idx}",
    )
    agg_value = cols[1].selectbox(
        "Agg",
        options=["sum", "avg", "min", "max", "count", "count_distinct"],
        index=0,
        key=f"sql_agg_fn_{idx}",
    )
    if cols[2].button("Remove", key=f"sql_agg_remove_{idx}"):
        st.session_state["sql_aggs"].pop(idx)
        st.rerun()
    if col_value:
        aggregations.append({"column": col_value, "agg": agg_value})

st.markdown("**Having (filters on aggregates)**")
if "sql_having" not in st.session_state:
    st.session_state["sql_having"] = []
if st.button("Add having filter", key="sql_add_having"):
    st.session_state["sql_having"].append(
        {"column": "", "operator": ">", "value": ""}
    )
having_mode = st.radio(
    "Having mode",
    ["AND", "OR"],
    horizontal=True,
    key="sql_having_mode",
)
having_filters = []
for idx, rule in enumerate(st.session_state["sql_having"]):
    cols = st.columns([3, 2, 3, 1])
    col_value = cols[0].text_input(
        "Aggregate column",
        value=rule.get("column", ""),
        key=f"sql_having_col_{idx}",
    )
    op_value = cols[1].selectbox(
        "Operator",
        options=["=", "!=", ">", ">=", "<", "<="],
        index=0,
        key=f"sql_having_op_{idx}",
    )
    val_value = cols[2].text_input(
        "Value",
        value=str(rule.get("value", "")),
        key=f"sql_having_val_{idx}",
    )
    if cols[3].button("Remove", key=f"sql_having_remove_{idx}"):
        st.session_state["sql_having"].pop(idx)
        st.rerun()
    if col_value:
        having_filters.append({"column": col_value, "operator": op_value, "value": val_value})

st.markdown("**Sort, Distinct, Limit**")
distinct = st.checkbox("Distinct rows", value=False, key="sql_distinct")
order_col = st.selectbox(
    "Order by column",
    options=[""] + sql_columns,
    index=0,
    key="sql_order_col",
)
order_asc = st.checkbox("Ascending order", value=True, key="sql_order_asc")
limit = st.number_input("Limit", min_value=0, value=0, step=1, key="sql_limit")

st.markdown("**Null handling**")
null_fill_col = st.selectbox(
    "Fill nulls for column",
    options=[""] + sql_columns,
    index=0,
    key="sql_null_fill_col",
)
null_fill_value = st.text_input(
    "Fill value",
    value="",
    key="sql_null_fill_value",
)
null_fills = {}
if null_fill_col:
    null_fills[null_fill_col] = null_fill_value

if st.button("Run SQL Features", key="sql_run"):
    report = build_sql_features_report(
        sql_df,
        select_columns=select_columns,
        aliases=alias_map,
        filters=filters,
        filter_mode=filter_mode,
        calculated_fields=calculated_fields,
        case_fields=case_fields,
        date_extracts=date_extracts,
        group_by=group_by_cols,
        aggregations=aggregations,
        having=having_filters,
        having_mode=having_mode,
        distinct=distinct,
        order_by=(order_col, order_asc) if order_col else None,
        limit=limit if limit > 0 else None,
        null_fills=null_fills,
    )
    st.markdown(f"**{report.title}**")
    st.caption(report.description)
    st.dataframe(report.data, use_container_width=True)

with st.expander("Export and Package", expanded=False):
    st.markdown('<a id="export-package"></a>', unsafe_allow_html=True)
    st.caption("Export and Package.")
    export_ready = st.checkbox(
        "Prepare export bundle", value=False, key="export_ready"
    )
    if export_ready:
        for result in results:
            st.markdown(f"**{result.name}**")
            report_groups = {}
            core_reports = cached_core_reports(result.df)
            report_groups["core"] = build_export_items(core_reports, "core")
            if st.session_state.get("run_advanced", False):
                advanced_reports = cached_advanced_reports(
                    result.df, st.session_state.get("advanced_mode", "fast")
                )
                report_groups["advanced"] = build_export_items(advanced_reports, "advanced")
            if st.session_state.get("run_time", False):
                time_reports = cached_time_reports(result.df)
                report_groups["time"] = build_export_items(time_reports, "time")
            if st.session_state.get("run_quality", False):
                quality_reports = cached_quality_reports(result.df)
                report_groups["quality"] = build_export_items(quality_reports, "quality")
            if st.session_state.get("run_rows", False):
                row_reports = cached_row_reports(result.df)
                report_groups["rows"] = build_export_items(row_reports, "rows")

            zip_path = package_reports(EXPORT_DIR, result.name, report_groups)
            with open(zip_path, "rb") as f:
                st.download_button(
                    "Download report bundle",
                    data=f.read(),
                    file_name=zip_path.name,
                    mime="application/zip",
                )
