# User Flow - Data Report Studio

This document explains the end-to-end user flow from file upload to reports and exports.

## 1) Open the App
- The app opens to the main screen with the title "Data Report Studio".
- User can select Industry/Vertical (General, Banking, Healthcare, Finance, Retail, Manufacturing).

## 2) Upload Files (Step 1)
- User chooses upload mode: Single file or Multiple files.
- User uploads CSV/XLSX files.
- App validates size, encoding, and headers.
- If any file fails, a clear error is shown.

## 3) Single-File Flow
- **Upload Summary**: file stats and preview (expandable).
- **Step 2: Understand column types**: type/role suggestions with overrides.
- **Step 3: Group columns into entities**: editable entity groupings.
- **Step 3b: Semantic modeling**:
  - Optional Banking pack with sub-domain (Retail Banking, Lending, Cards, Payments).
  - Semantic role/entity/metric/unit inference.
- **Report Abstract**: category-grouped report index with links.
- **Report View**: selected report only, with Back to Report Abstract link.
- **User Dynamics**: ad-hoc reports (group by, aggregates, filters).
- **SQL Features**: SQL-style report builder (filters, calculated fields, group-by, having, etc.).
- **Export & Packaging**: optional ZIP export of report outputs.

## 4) Multiple-File Flow (Join First)
- **Uploaded Files**: file names only (no previews or summaries).
- **Join Suggestions**:
  - If a join column exists in every file, it is selectable.
  - If no exact match, show nearest joiner suggestions with per-file presence.
  - User selects join column + join type (inner/left) and confirms.
- **Before confirmation**: no reports are generated.
- **After confirmation**:
  - A unified dataset is built.
  - Report Abstract and Report View operate on the unified dataset.
  - User Dynamics + SQL Features can use the unified dataset.

## 5) Report Categories
- Executive, Basic, Simple, Intermediate, Advanced, Expert, Forensics.
- Categories are shown in the report abstract even if empty.

## 6) Banking Template Reports (when Banking pack is used)
- Channel Mix
- Delinquency Buckets (DPD)
- Balance Summary
- Balance Aging (Monthly Average)
- Loan Portfolio Summary
- Fee and Penalty Analysis
- Category counts and amount/quantity rollups when applicable
