# Data Report Studio - Product Document (Banking Focus)

This document is the authoritative source of truth for the current product behavior. It is designed so a new chat can become fully context-aware without reading code.

---

## 1) Product Purpose

Data Report Studio is a Streamlit-based, deterministic report engine for CSV/XLSX datasets. It generates non-visual, tabular intelligence reports, optimized for the Banking vertical. It does not use LLMs/AI. Same input yields the same output.

---

## 2) Tech Stack

- Python, Streamlit
- pandas, numpy, scipy, statsmodels
- pyarrow, openpyxl
- polars is optional (lazy parsing if installed)

`requirements.txt` lists dependencies.

---

## 3) Folder Structure

- `app/`: Streamlit UI (`app/main.py`)
- `engine/`: schema inference, entities, relationships, semantics
- `reports/`: report generators
- `exports/`: export bundle logic
- `data_uploads/`: persisted uploads + manifest
- `docs/`: documentation

---

## 4) Core App Flow

### 4.1 Entry & Navigation
- Top-right CTA: **Export and Package** jumps to export section.
- Top-right CTA: **Report List** opens a standalone report catalog page.
- Industry/Vertical selector (Banking-focused but supports other labels).
- Upload mode: **Single file** or **Multiple files**.

### 4.1a Report List (Standalone Page)
- Global, data-independent catalog of all report categories and report types (max coverage).
- Shows category index with maximum counts.
- Includes notes: no AI, 8GB laptop friendly, data quality dependent, deterministic output, multi-file join prompt.

### 4.2 Single-File Mode
1. Upload file (CSV/XLSX).
2. Upload summary + preview.
3. Schema inference (type/role) with overrides.
4. Entity clustering (editable).
5. Semantic modeling (Banking pack optional).
6. Banking Rules (editable defaults; see section 8).
7. Report Abstract (category grouped with links).
8. Report View (single report + back link).
9. User Dynamics (custom report builder).
10. SQL Features (SQL-style report builder).
11. Export & Package.

### 4.3 Multiple-File Mode (Join-First)
1. Upload multiple files (CSV/XLSX).
2. File names only (no previews/summaries).
3. Banking Rules editor (same as single-file).
4. Join Suggestions:
   - If a column exists in **every** file, it is selectable.
   - Otherwise, show nearest joiner suggestions with per-file presence.
5. Reports only generate after join confirmation.
6. All reports use the unified joined dataset.

---

## 5) Report Categories

Reports are categorized for navigation:

1. Executive  
2. Basic  
3. Simple  
4. Intermediate  
5. Advanced  
6. Expert  
7. Forensics  

Category mapping in UI:
- Core → Executive/Basic
- Templates (Banking) → Simple
- Advanced → Intermediate
- Time Series → Advanced
- Relationships → Expert
- Quality + Row-level → Forensics

---

## 6) Report Families

### 6.1 Core Reports (`reports/core.py`) – Basic/Executive
- Dataset Overview (Executive)
- Column Profile Summary
- Missing Values
- Duplicate Rows
- Numeric Summary
- Top Categories

### 6.2 Advanced Reports (`reports/advanced.py`) – Intermediate
- Pareto Summary
- Distribution Buckets
- Skewness
- Mean vs Median
- Correlation Matrix (full mode)
- Outlier Buckets (full mode)

### 6.3 Time-Series (`reports/time_series.py`) – Advanced
- Daily & monthly trends
- Period-over-period changes

### 6.4 Relationships (`reports/relationships.py`) – Expert
- Join-based rollups when relationships exist

### 6.5 Data Quality (`reports/quality.py`) – Forensics
- IQR outliers
- MAD outliers
- Label variants
- Numeric drift
- Category drift
- Time-series spikes/drops
- Invalid email/phone

### 6.6 Row-Level (`reports/rows.py`) – Forensics
- Rows with most missing values
- Extreme numeric rows
- Rare pattern rows

### 6.7 Templates / Banking Intelligence (`reports/templates.py`) – Simple

Base template reports:
- Category Count
- Amount by Category
- Quantity by Category
- Average Rate Metrics
- Channel Mix

Banking templates:
- Delinquency Buckets (DPD)
- Portfolio at Risk (PAR30/60/90)
- NPA Proxy (DPD 90+)
- Loan Vintage / Seasoning Table
- Collections Roll Rate Matrix
- Collections Cure Rate
- Balance Summary
- Balance Aging (Monthly Average)
- Loan Portfolio Summary (principal/interest/EMI totals)
- Fee and Penalty Analysis
- Interest Income vs Outstanding
- Risk Grade Distribution
- Expected Loss Summary (PD * LGD * EAD)
- Deposit Account Type Mix
- Deposit Balance by Account Type
- CASA Mix
- Deposit Balance Bands
- Average Daily Balance
- Branch Volume Summary
- Branch Outstanding Summary
- Relationship Manager Portfolio
- Relationship Manager Outstanding
- Credit Utilization Summary
- Credit Utilization Bands
- Large Transaction Summary
- Round Amount Frequency
- Transaction Velocity (Daily)

Trigger rules are column-driven (based on presence of key columns like `dpd`, `balance`, `interest`, `pd/lgd/ead`, `branch`, `rm`, `limit`, etc.).

---

## 6.8 Report List (Catalog)
- Category index with max counts
- Full list of categories and report names
- Includes User Dynamics, Lookup, SQL Features as variable categories

## 7) User Dynamics

### Custom Report Builder
User builds ad-hoc tables:
- Dimensions (group by)
- Metric + aggregation (count/sum/avg/min/max)
- Top-N
- Percent of total
- Categorical and numeric filters

### Lookup Report
User selects left/right datasets, join keys, return columns. Output is a join-based lookup table.

---

## 8) SQL Features (User Dynamics)

SQL-style builder (no charts):
- Select columns + aliases
- WHERE filters with AND/OR and operators
- Calculated fields
- CASE/conditional fields
- Date extracts (year/month/day)
- GROUP BY + multi-aggregation
- HAVING filters
- DISTINCT
- ORDER BY + LIMIT
- Null fill

---

## 9) Banking Rules (Editable Defaults)

The following are editable in the UI and drive banking reports:

- **DPD buckets** (default: `0,30,60,90,120,999999`)
- **Balance bands** (default 0 to 1m+)
- **CASA mapping**
  - Savings values: `savings,sa,sv`
  - Current values: `current,ca,cc`
- **Large transaction threshold** (default: `100000`)
- **Round amount base** (default: `1000`)

Rules apply to both single and multiple file flows. In multi-file, the rules appear before join confirmation.

## 10) Report List Notes (UI)
- No AI involved in report generation
- Works on 8GB laptops for typical datasets
- Reports depend on data quality and available columns
- Deterministic outputs
- Multi-file uploads prompt for joins before reports

---

## 11) Export & Packaging

Export generates a zipped report bundle per dataset:
- CSV report outputs
- README with report list
- Metadata JSON
- Dependency index CSV

---

## 12) Persistent State

- Uploaded files stored in `data_uploads/`
- Manifest: `data_uploads/last_uploads.json`
- Report links use query params for navigation

---

## 13) How to Run

```
venv\\Scripts\\python -m streamlit run app\\main.py
```

---

## 14) Current Status

All requested banking packs are implemented:
- Credit risk (PD/LGD/EAD, risk grade distribution, expected loss)
- Branch/relationship rollups
- Fee & revenue (interest vs outstanding)
- Credit utilization (limits vs usage)
- Transaction monitoring (velocity, large-value, round amounts)

This document fully describes the current product for a fresh session.
