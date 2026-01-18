from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from engine.semantic import SemanticInference


@dataclass
class TemplateReport:
    title: str
    description: str
    data: pd.DataFrame
    confidence: float


def _first_match(columns: List[str], keyword: str) -> Optional[str]:
    for col in columns:
        if keyword in col.lower():
            return col
    return None


def _find_category(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique(dropna=True) <= 50:
            return col
    return None


def _first_match_any(columns: List[str], keywords: List[str]) -> Optional[str]:
    for keyword in keywords:
        col = _first_match(columns, keyword)
        if col:
            return col
    return None


def build_template_reports(
    df: pd.DataFrame,
    semantics: List[SemanticInference],
    rules: Optional[dict] = None,
) -> List[TemplateReport]:
    reports: List[TemplateReport] = []
    rules = rules or {}
    columns = list(df.columns)
    category_col = _find_category(df)
    amount_col = _first_match(columns, "amount") or _first_match(columns, "sales")
    quantity_col = _first_match(columns, "quantity") or _first_match(columns, "qty")
    balance_col = _first_match(columns, "balance")
    interest_col = _first_match(columns, "interest") or _first_match(columns, "interest_income")
    emi_col = _first_match(columns, "emi")
    principal_col = _first_match(columns, "principal")
    fee_col = _first_match(columns, "fee") or _first_match(columns, "charge")
    channel_col = _first_match(columns, "channel")
    dpd_col = _first_match(columns, "dpd") or _first_match(columns, "delinquency")
    date_col = _first_match(columns, "date")
    id_col = _first_match_any(
        columns,
        ["loan_id", "account_no", "account_id", "customer_id", "id"],
    )
    loan_date_col = _first_match_any(
        columns,
        ["disbursal", "disbursement", "opening", "origination", "loan_date"],
    )
    outstanding_col = balance_col or principal_col or amount_col
    pd_col = _first_match_any(columns, ["pd", "probability_of_default"])
    lgd_col = _first_match_any(columns, ["lgd", "loss_given_default"])
    ead_col = _first_match_any(columns, ["ead", "exposure_at_default", "exposure"])
    risk_grade_col = _first_match_any(columns, ["risk_grade", "rating", "grade", "risk_band"])
    limit_col = _first_match_any(columns, ["limit", "credit_limit", "sanction_limit"])
    utilized_col = _first_match_any(columns, ["utilized", "utilisation", "used", "outstanding"])
    transaction_amount_col = _first_match_any(columns, ["transaction_amount", "amount", "txn_amount"])
    transaction_id_col = _first_match_any(columns, ["transaction_id", "txn_id", "utr", "reference"])

    if category_col:
        counts = df[category_col].astype(str).value_counts().reset_index()
        counts.columns = [category_col, "count"]
        reports.append(
            TemplateReport(
                title=f"{category_col} Count",
                description="Counts grouped by category.",
                data=counts,
                confidence=0.75,
            )
        )

    if amount_col and category_col:
        totals = (
            df.groupby(category_col)[amount_col]
            .sum(numeric_only=True)
            .reset_index()
            .rename(columns={amount_col: "total_amount"})
        )
        reports.append(
            TemplateReport(
                title=f"{amount_col} by {category_col}",
                description="Total amount grouped by category.",
                data=totals,
                confidence=0.8,
            )
        )

    if quantity_col and category_col:
        totals = (
            df.groupby(category_col)[quantity_col]
            .sum(numeric_only=True)
            .reset_index()
            .rename(columns={quantity_col: "total_quantity"})
        )
        reports.append(
            TemplateReport(
                title=f"{quantity_col} by {category_col}",
                description="Total quantity grouped by category.",
                data=totals,
                confidence=0.8,
            )
        )

    if channel_col:
        counts = df[channel_col].astype(str).value_counts().reset_index()
        counts.columns = [channel_col, "count"]
        reports.append(
            TemplateReport(
                title="Channel Mix",
                description="Volume by transaction channel.",
                data=counts,
                confidence=0.75,
            )
        )

    if dpd_col:
        series = pd.to_numeric(df[dpd_col], errors="coerce")
        dpd_bins = rules.get("dpd_bins") or [-1, 0, 30, 60, 90, 120, 999999]
        dpd_labels = rules.get("dpd_labels") or ["0", "1-30", "31-60", "61-90", "91-120", "120+"]
        if len(dpd_bins) == len(dpd_labels) + 1:
            bucketed = pd.cut(series, bins=dpd_bins, labels=dpd_labels)
        else:
            bucketed = pd.cut(series, bins=[-1, 0, 30, 60, 90, 120, 999999], labels=["0", "1-30", "31-60", "61-90", "91-120", "120+"])
        counts = bucketed.value_counts().sort_index().reset_index()
        counts.columns = ["dpd_bucket", "count"]
        reports.append(
            TemplateReport(
                title="Delinquency Buckets (DPD)",
                description="Distribution of days past due.",
                data=counts,
                confidence=0.7,
            )
        )

    if dpd_col and outstanding_col:
        dpd_series = pd.to_numeric(df[dpd_col], errors="coerce")
        exposure = pd.to_numeric(df[outstanding_col], errors="coerce")
        total_exposure = exposure.sum()
        par_rows = []
        for threshold in (30, 60, 90):
            mask = dpd_series >= threshold
            par_rows.append(
                {
                    "par_bucket": f"PAR{threshold}+",
                    "accounts": int(mask.fillna(False).sum()),
                    "exposure": float(exposure[mask].sum()),
                    "exposure_pct": round(float(exposure[mask].sum()) / total_exposure * 100, 2)
                    if total_exposure
                    else 0.0,
                }
            )
        reports.append(
            TemplateReport(
                title="Portfolio at Risk (PAR) Summary",
                description="Exposure and account counts for PAR30/60/90 buckets.",
                data=pd.DataFrame(par_rows),
                confidence=0.75,
            )
        )
        npa_mask = dpd_series >= 90
        npa_data = pd.DataFrame(
            {
                "npa_accounts": [int(npa_mask.fillna(False).sum())],
                "npa_exposure": [float(exposure[npa_mask].sum())],
                "npa_exposure_pct": [
                    round(float(exposure[npa_mask].sum()) / total_exposure * 100, 2)
                    if total_exposure
                    else 0.0
                ],
            }
        )
        reports.append(
            TemplateReport(
                title="NPA Proxy (DPD 90+)",
                description="Proxy NPA exposure using DPD >= 90.",
                data=npa_data,
                confidence=0.7,
            )
        )

    if balance_col:
        data = pd.DataFrame(
            {
                "average_balance": [pd.to_numeric(df[balance_col], errors="coerce").mean()],
                "max_balance": [pd.to_numeric(df[balance_col], errors="coerce").max()],
                "min_balance": [pd.to_numeric(df[balance_col], errors="coerce").min()],
            }
        )
        reports.append(
            TemplateReport(
                title="Balance Summary",
                description="Average, min, max balance.",
                data=data,
                confidence=0.75,
            )
        )

    if balance_col and date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        temp = pd.DataFrame({date_col: parsed, balance_col: pd.to_numeric(df[balance_col], errors="coerce")})
        temp = temp.dropna()
        if not temp.empty:
            monthly = temp.set_index(date_col).resample("M")[balance_col].mean().reset_index()
            reports.append(
                TemplateReport(
                    title="Balance Aging (Monthly Average)",
                    description="Average balance by month.",
                    data=monthly,
                    confidence=0.7,
                )
            )

    if principal_col or interest_col or emi_col:
        data = pd.DataFrame(
            {
                "total_principal": [pd.to_numeric(df.get(principal_col, pd.Series()), errors="coerce").sum()],
                "total_interest": [pd.to_numeric(df.get(interest_col, pd.Series()), errors="coerce").sum()],
                "total_emi": [pd.to_numeric(df.get(emi_col, pd.Series()), errors="coerce").sum()],
            }
        )
        reports.append(
            TemplateReport(
                title="Loan Portfolio Summary",
                description="Totals for principal, interest, and EMI.",
                data=data,
                confidence=0.7,
            )
        )

    if loan_date_col and outstanding_col:
        parsed = pd.to_datetime(df[loan_date_col], errors="coerce")
        temp = pd.DataFrame(
            {
                "cohort_month": parsed.dt.to_period("M").astype(str),
                "outstanding": pd.to_numeric(df[outstanding_col], errors="coerce"),
            }
        ).dropna()
        if not temp.empty:
            vintage = temp.groupby("cohort_month").agg(
                accounts=("outstanding", "count"),
                total_outstanding=("outstanding", "sum"),
                avg_outstanding=("outstanding", "mean"),
            ).reset_index()
            if dpd_col:
                delinquent = pd.to_numeric(df[dpd_col], errors="coerce") >= 30
                temp["delinquent"] = delinquent.values
                delinquent_rate = (
                    temp.groupby("cohort_month")["delinquent"]
                    .mean()
                    .reset_index()
                    .rename(columns={"delinquent": "delinquent_rate"})
                )
                vintage = vintage.merge(delinquent_rate, on="cohort_month", how="left")
                vintage["delinquent_rate"] = (vintage["delinquent_rate"] * 100).round(2)
            reports.append(
                TemplateReport(
                    title="Loan Vintage / Seasoning Table",
                    description="Cohort month rollup of outstanding and delinquency.",
                    data=vintage,
                    confidence=0.7,
                )
            )

    if dpd_col and date_col and id_col:
        temp = pd.DataFrame(
            {
                "id": df[id_col],
                "dpd": pd.to_numeric(df[dpd_col], errors="coerce"),
                "period": pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M"),
            }
        ).dropna(subset=["id", "period"])
        if not temp.empty:
            bins = [-1, 0, 30, 60, 90, 120, 999999]
            labels = ["0", "1-30", "31-60", "61-90", "91-120", "120+"]
            temp["bucket"] = pd.cut(temp["dpd"], bins=bins, labels=labels)
            temp = temp.sort_values(["id", "period"])
            temp["prev_bucket"] = temp.groupby("id")["bucket"].shift(1)
            transitions = (
                temp.dropna(subset=["prev_bucket", "bucket"])
                .groupby(["prev_bucket", "bucket"])
                .size()
                .reset_index(name="count")
            )
            reports.append(
                TemplateReport(
                    title="Collections Roll Rate Matrix",
                    description="Transition counts between delinquency buckets by period.",
                    data=transitions,
                    confidence=0.65,
                )
            )
            cure_mask = (temp["prev_bucket"].isin(["31-60", "61-90", "91-120", "120+"])) & (
                temp["bucket"].isin(["0", "1-30"])
            )
            cure_rate = (
                temp.dropna(subset=["prev_bucket", "bucket"])
                .groupby("prev_bucket")["bucket"]
                .apply(lambda s: round(s.isin(["0", "1-30"]).mean() * 100, 2))
                .reset_index()
                .rename(columns={"bucket": "cure_rate_pct"})
            )
            reports.append(
                TemplateReport(
                    title="Collections Cure Rate",
                    description="Percentage of delinquent accounts that moved to current/1-30.",
                    data=cure_rate,
                    confidence=0.65,
                )
            )

    if fee_col:
        data = pd.DataFrame(
            {
                "total_fees": [pd.to_numeric(df[fee_col], errors="coerce").sum()],
                "average_fee": [pd.to_numeric(df[fee_col], errors="coerce").mean()],
            }
        )
        reports.append(
            TemplateReport(
                title="Fee and Penalty Analysis",
                description="Total and average fees/charges.",
                data=data,
                confidence=0.75,
            )
        )

    if interest_col and outstanding_col:
        interest = pd.to_numeric(df[interest_col], errors="coerce")
        outstanding = pd.to_numeric(df[outstanding_col], errors="coerce")
        data = pd.DataFrame(
            {
                "total_interest_income": [interest.sum()],
                "total_outstanding": [outstanding.sum()],
                "interest_yield_pct": [
                    round((interest.sum() / outstanding.sum()) * 100, 2)
                    if outstanding.sum()
                    else 0.0
                ],
            }
        )
        reports.append(
            TemplateReport(
                title="Interest Income vs Outstanding",
                description="Total interest income, outstanding, and implied yield.",
                data=data,
                confidence=0.7,
            )
        )

    if limit_col and utilized_col:
        limit_series = pd.to_numeric(df[limit_col], errors="coerce")
        utilized_series = pd.to_numeric(df[utilized_col], errors="coerce")
        utilization = (utilized_series / limit_series).replace([pd.NA, pd.NaT], 0)
        data = pd.DataFrame(
            {
                "total_limit": [limit_series.sum()],
                "total_utilized": [utilized_series.sum()],
                "avg_utilization_pct": [round((utilization.mean() * 100), 2)],
            }
        )
        reports.append(
            TemplateReport(
                title="Credit Utilization Summary",
                description="Total limits, utilized, and average utilization.",
                data=data,
                confidence=0.7,
            )
        )
        bands = pd.cut(
            utilization,
            bins=[-0.01, 0.25, 0.5, 0.75, 1.0, 999],
            labels=["0-25%", "25-50%", "50-75%", "75-100%", "100%+"],
        )
        band_counts = bands.value_counts().sort_index().reset_index()
        band_counts.columns = ["utilization_band", "count"]
        reports.append(
            TemplateReport(
                title="Credit Utilization Bands",
                description="Distribution of utilization ratios.",
                data=band_counts,
                confidence=0.65,
            )
        )

    if transaction_amount_col:
        txn_series = pd.to_numeric(df[transaction_amount_col], errors="coerce")
        threshold = rules.get("large_txn_threshold", 100000)
        large_txn = txn_series >= threshold
        large_data = pd.DataFrame(
            {
                "large_txn_threshold": [threshold],
                "large_txn_count": [int(large_txn.fillna(False).sum())],
                "large_txn_total": [float(txn_series[large_txn].sum())],
            }
        )
        reports.append(
            TemplateReport(
                title="Large Transaction Summary",
                description="Counts and totals above a large-transaction threshold.",
                data=large_data,
                confidence=0.7,
            )
        )
        round_base = rules.get("round_amount_base", 1000)
        round_txn = (txn_series % round_base == 0)
        round_data = pd.DataFrame(
            {
                "round_amount_count": [int(round_txn.fillna(False).sum())],
                "round_amount_pct": [
                    round(int(round_txn.fillna(False).sum()) / max(len(txn_series.dropna()), 1) * 100, 2)
                ],
            }
        )
        reports.append(
            TemplateReport(
                title="Round Amount Frequency",
                description=f"Frequency of round-amount transactions (multiple of {round_base}).",
                data=round_data,
                confidence=0.65,
            )
        )

    if transaction_amount_col and date_col and transaction_id_col:
        temp = pd.DataFrame(
            {
                "txn_id": df[transaction_id_col],
                "amount": pd.to_numeric(df[transaction_amount_col], errors="coerce"),
                "date": pd.to_datetime(df[date_col], errors="coerce"),
            }
        ).dropna(subset=["txn_id", "date"])
        if not temp.empty:
            daily = temp.set_index("date").resample("D")["txn_id"].count().reset_index()
            daily = daily.rename(columns={"txn_id": "txn_count"})
            reports.append(
                TemplateReport(
                    title="Transaction Velocity (Daily)",
                    description="Daily transaction counts.",
                    data=daily,
                    confidence=0.65,
                )
            )

    if risk_grade_col:
        grade_counts = df[risk_grade_col].astype(str).value_counts().reset_index()
        grade_counts.columns = [risk_grade_col, "count"]
        reports.append(
            TemplateReport(
                title="Risk Grade Distribution",
                description="Counts by risk grade/rating.",
                data=grade_counts,
                confidence=0.7,
            )
        )

    if pd_col and lgd_col and ead_col:
        pd_series = pd.to_numeric(df[pd_col], errors="coerce")
        lgd_series = pd.to_numeric(df[lgd_col], errors="coerce")
        ead_series = pd.to_numeric(df[ead_col], errors="coerce")
        expected_loss = (pd_series * lgd_series * ead_series).fillna(0)
        data = pd.DataFrame(
            {
                "total_ead": [ead_series.sum()],
                "average_pd": [pd_series.mean()],
                "average_lgd": [lgd_series.mean()],
                "expected_loss": [expected_loss.sum()],
            }
        )
        reports.append(
            TemplateReport(
                title="Expected Loss Summary",
                description="Expected loss using PD * LGD * EAD.",
                data=data,
                confidence=0.7,
            )
        )

    account_type_col = _first_match_any(columns, ["account_type", "acct_type", "account_category", "deposit_type"])
    account_balance_col = balance_col or _first_match_any(columns, ["available_balance", "current_balance"])
    branch_col = _first_match_any(columns, ["branch", "branch_code", "branch_id", "branch_name"])
    rm_col = _first_match_any(columns, ["rm", "relationship_manager", "manager", "rm_id", "rm_name"])
    if account_type_col and account_balance_col:
        acct_counts = df[account_type_col].astype(str).value_counts().reset_index()
        acct_counts.columns = [account_type_col, "count"]
        reports.append(
            TemplateReport(
                title="Deposit Account Type Mix",
                description="Counts by deposit account type (CASA/product mix).",
                data=acct_counts,
                confidence=0.7,
            )
        )
        balances = (
            df.groupby(account_type_col)[account_balance_col]
            .sum(numeric_only=True)
            .reset_index()
            .rename(columns={account_balance_col: "total_balance"})
        )
        reports.append(
            TemplateReport(
                title="Deposit Balance by Account Type",
                description="Total balances by deposit account type.",
                data=balances,
                confidence=0.7,
            )
        )
        savings_vals = set(v.lower() for v in rules.get("casa_savings_values", []))
        current_vals = set(v.lower() for v in rules.get("casa_current_values", []))
        if savings_vals or current_vals:
            casa_map = []
            for value in df[account_type_col].astype(str).unique():
                key = value.lower()
                if key in savings_vals:
                    casa_map.append({"account_type": value, "casa_type": "Savings"})
                elif key in current_vals:
                    casa_map.append({"account_type": value, "casa_type": "Current"})
                else:
                    casa_map.append({"account_type": value, "casa_type": "Other"})
            casa_df = pd.DataFrame(casa_map)
            mix = (
                df[[account_type_col, account_balance_col]]
                .merge(casa_df, left_on=account_type_col, right_on="account_type", how="left")
                .groupby("casa_type")[account_balance_col]
                .sum(numeric_only=True)
                .reset_index()
                .rename(columns={account_balance_col: "total_balance"})
            )
            reports.append(
                TemplateReport(
                    title="CASA Mix",
                    description="Total balances by CASA type (Savings/Current/Other).",
                    data=mix,
                    confidence=0.7,
                )
            )

    if account_balance_col:
        series = pd.to_numeric(df[account_balance_col], errors="coerce")
        balance_bands = rules.get("balance_bands")
        if balance_bands:
            bins = [-1] + [band["max"] for band in balance_bands]
            labels = [band["label"] for band in balance_bands]
        else:
            bins = [-1, 0, 10000, 50000, 100000, 500000, 1000000, 999999999]
            labels = ["0", "1-10k", "10k-50k", "50k-100k", "100k-500k", "500k-1m", "1m+"]
        bucketed = pd.cut(series, bins=bins, labels=labels)
        counts = bucketed.value_counts().sort_index().reset_index()
        counts.columns = ["balance_band", "count"]
        reports.append(
            TemplateReport(
                title="Deposit Balance Bands",
                description="Distribution of balances across bands.",
                data=counts,
                confidence=0.65,
            )
        )

    if account_balance_col and date_col:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        temp = pd.DataFrame(
            {
                date_col: parsed,
                account_balance_col: pd.to_numeric(df[account_balance_col], errors="coerce"),
            }
        ).dropna()
        if not temp.empty:
            adb = temp.set_index(date_col).resample("D")[account_balance_col].mean().reset_index()
            adb = adb.rename(columns={account_balance_col: "average_daily_balance"})
            reports.append(
                TemplateReport(
                    title="Average Daily Balance",
                    description="Daily average balance time series.",
                    data=adb,
                    confidence=0.65,
                )
            )

    if branch_col:
        branch_counts = df[branch_col].astype(str).value_counts().reset_index()
        branch_counts.columns = [branch_col, "count"]
        reports.append(
            TemplateReport(
                title="Branch Volume Summary",
                description="Counts by branch.",
                data=branch_counts,
                confidence=0.7,
            )
        )
        if outstanding_col:
            branch_bal = (
                df.groupby(branch_col)[outstanding_col]
                .sum(numeric_only=True)
                .reset_index()
                .rename(columns={outstanding_col: "total_outstanding"})
            )
            reports.append(
                TemplateReport(
                    title="Branch Outstanding Summary",
                    description="Outstanding balance totals by branch.",
                    data=branch_bal,
                    confidence=0.7,
                )
            )

    if rm_col:
        rm_counts = df[rm_col].astype(str).value_counts().reset_index()
        rm_counts.columns = [rm_col, "count"]
        reports.append(
            TemplateReport(
                title="Relationship Manager Portfolio",
                description="Counts by relationship manager.",
                data=rm_counts,
                confidence=0.7,
            )
        )
        if outstanding_col:
            rm_bal = (
                df.groupby(rm_col)[outstanding_col]
                .sum(numeric_only=True)
                .reset_index()
                .rename(columns={outstanding_col: "total_outstanding"})
            )
            reports.append(
                TemplateReport(
                    title="Relationship Manager Outstanding",
                    description="Outstanding balance totals by relationship manager.",
                    data=rm_bal,
                    confidence=0.7,
                )
            )
    for inf in semantics:
        if inf.semantic_role == "rate":
            data = pd.DataFrame(
                {
                    "column": [inf.column],
                    "mean": [pd.to_numeric(df[inf.column], errors="coerce").mean()],
                }
            )
            reports.append(
            TemplateReport(
                title=f"Average {inf.column}",
                description="Average rate metric.",
                data=data,
                confidence=0.7,
            )
        )
    return reports
