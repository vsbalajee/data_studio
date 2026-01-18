# Report Pack - 7cde8b5a7e4d47bca81b112cc009b297_customers.xlsx

This bundle contains CSV report outputs for the selected dataset.

## Reports
- core_dataset_overview.csv: Dataset Overview - A quick snapshot of row and column counts.
- core_column_profile_summary.csv: Column Profile Summary - Type, missing count, and unique values per column.
- core_missing_values.csv: Missing Values - Missing values per column with rates.
- core_duplicate_rows.csv: Duplicate Rows - Total number of duplicate rows in the dataset.
- core_numeric_summary.csv: Numeric Summary - Basic stats (min, max, mean, etc.) for numeric columns.
- core_top_categories_-_customer_id.csv: Top Categories - customer_id - Most common values and their counts.
- core_top_categories_-_customer_name.csv: Top Categories - customer_name - Most common values and their counts.
- core_top_categories_-_city.csv: Top Categories - city - Most common values and their counts.
- core_top_categories_-_kyc_status.csv: Top Categories - kyc_status - Most common values and their counts.
- core_top_categories_-_email.csv: Top Categories - email - Most common values and their counts.
- advanced_pareto_summary.csv: Pareto Summary - Top categories with cumulative contribution.
- advanced_skewness.csv: Skewness - Skewness for numeric columns.
- advanced_mean_vs_median.csv: Mean vs Median - Compare mean and median for numeric columns.
- quality_outlier_summary.csv: Outlier Summary - IQR-based outlier counts per numeric column.
- quality_robust_outliers__mad_.csv: Robust Outliers (MAD) - Median absolute deviation based outlier counts.
- quality_label_variants.csv: Label Variants - Columns with inconsistent text variants (case/spacing).
- quality_numeric_drift.csv: Numeric Drift - Mean shift between first and second halves of the data.
- quality_category_drift.csv: Category Drift - Distribution shift between first and second halves.
- quality_time_series_spikes.csv: Time Series Spikes - Detect spikes/drops in monthly record counts.
- quality_invalid_emails_-_email.csv: Invalid Emails - email - Rows that do not look like valid emails.
- rows_rows_with_most_missing_values.csv: Rows with Most Missing Values - Rows that have the highest number of missing cells.
- rows_extreme_numeric_rows.csv: Extreme Numeric Rows - Rows with extreme numeric values.
- rows_rare_pattern_rows.csv: Rare Pattern Rows - Rows with rare category combinations.