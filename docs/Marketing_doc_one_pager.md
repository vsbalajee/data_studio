# Data Report Studio - One‑Pager (Banking Intelligence Focus)

## What It Is
Data Report Studio is a deterministic, no‑LLM analytics engine that transforms CSV/XLSX data into banking‑grade intelligence reports. It delivers Power BI‑level insight **without visuals**, focusing on fast, explainable, tabular outputs. There is **no AI involved**, and data remains local for secure, confidential analysis.

## Why It’s Intelligent
- Banking‑aware semantics and rule‑driven logic  
- Consistent, explainable results (same input → same output)  
- Join‑first multi‑file intelligence for unified analysis  
- Editable banking rules (DPD, CASA, balance bands, thresholds)  
- SQL‑style builder for advanced, no‑code queries  
- Users can interactively build dynamic reports after upload  

## Core Banking Intelligence (Examples)
- Delinquency buckets, PAR30/60/90, NPA proxy  
- Vintage/seasoning tables  
- Roll‑rates & cure rates  
- Risk grade distribution & expected loss (PD/LGD/EAD)  
- Branch/RM portfolio rollups  
- Interest income vs outstanding  
- Deposit analytics (CASA mix, balance bands, ADB)  
- Credit utilization bands  
- Transaction monitoring (large values, round amounts, velocity)  

## Report List (Categories and Reports)

Report list is a global, data‑independent view that shows the maximum report coverage available in the product.

Key notes:
- No AI is involved in report generation.  
- Works well on 8GB laptops for typical banking datasets.  
- Reports are subject to data quality and available columns.  
- Deterministic output: same data produces the same results.  
- Multiple sheets prompt for joins before reports.  

### Report Index (Max Coverage)

Executive, Basic, Simple, Intermediate, Advanced, Expert, Forensics  
User Dynamics (variable)  
Lookup (variable)  
SQL Features (variable)  

### Executive
- Dataset Overview

### Basic
- Column Profile Summary  
- Missing Values  
- Duplicate Rows  
- Numeric Summary  
- Top Categories  

### Simple (Banking Intelligence)
- Category Count  
- Amount by Category  
- Quantity by Category  
- Average Rate Metrics  
- Channel Mix  
- Delinquency Buckets (DPD)  
- Portfolio at Risk (PAR30/60/90)  
- NPA Proxy (DPD 90+)  
- Loan Vintage / Seasoning Table  
- Collections Roll Rate Matrix  
- Collections Cure Rate  
- Balance Summary  
- Balance Aging (Monthly Average)  
- Loan Portfolio Summary  
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

### Intermediate
- Pareto Summary  
- Distribution Buckets  
- Skewness  
- Mean vs Median  
- Correlation Matrix (full mode)  
- Outlier Buckets (full mode)  

### Advanced
- Daily Trend Counts  
- Monthly Trend Counts  
- Period‑over‑Period Changes  

### Expert
- Join‑based Rollups  

### Forensics
- IQR Outlier Summary  
- MAD Outlier Summary  
- Label Variants  
- Numeric Drift  
- Category Drift  
- Time Series Spikes  
- Invalid Emails  
- Invalid Phones  
- Rows with Most Missing Values  
- Extreme Numeric Rows  
- Rare Pattern Rows  

### User Dynamics
- User‑defined (dynamic; depends on selected fields)  

### Lookup
- User‑defined (depends on datasets and join keys)  

### SQL Features
- User‑defined (depends on query selections)  

## Intelligence Rating (No Visuals)
~**75–80% of Power BI intelligence** for banking reports.  
The remaining gap is primarily full DAX and the visualization stack.

## Summary Value
Banking‑first analytics, deterministic insights, and no‑code flexibility — ready for audit‑friendly, production reporting.
