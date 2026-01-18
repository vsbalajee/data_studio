from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class SemanticInference:
    column: str
    semantic_role: str
    entity: Optional[str]
    metric_type: str
    unit: Optional[str]
    confidence: float


BASE_GLOSSARY = {
    "customer": ("customer", "client", "buyer"),
    "order": ("order", "purchase"),
    "product": ("product", "item", "sku"),
    "invoice": ("invoice", "bill"),
    "amount": ("amount", "price", "cost", "total", "revenue", "sales"),
    "quantity": ("quantity", "qty", "units", "count"),
    "rate": ("rate", "ratio", "percent", "pct", "%"),
}

BANKING_GLOSSARY = {
    "account": (
        "account",
        "acct",
        "account_no",
        "account_number",
        "iban",
        "swift",
        "bic",
    ),
    "customer": (
        "customer",
        "client",
        "holder",
        "cif",
        "cif_id",
        "customer_id",
        "client_id",
    ),
    "transaction": (
        "transaction",
        "txn",
        "transfer",
        "utr",
        "reference_no",
        "rrn",
    ),
    "card": (
        "card",
        "debit",
        "credit",
        "card_no",
        "card_number",
        "pan",
    ),
    "branch": (
        "branch",
        "ifsc",
        "routing",
        "aba",
        "sort_code",
        "branch_code",
    ),
    "loan": (
        "loan",
        "emi",
        "installment",
        "principal",
        "tenure",
        "moratorium",
        "ltv",
    ),
    "interest": (
        "interest",
        "apr",
        "rate",
        "roi",
        "yield",
    ),
    "balance": (
        "balance",
        "outstanding",
        "available_balance",
        "ledger_balance",
    ),
    "fee": ("fee", "charge", "commission", "penalty"),
    "status": ("status", "state", "account_status", "loan_status"),
    "date": (
        "date",
        "posted",
        "value_date",
        "txn_date",
        "booking_date",
        "statement_date",
        "opening_date",
        "maturity_date",
    ),
    "channel": ("channel", "atm", "upi", "imps", "neft", "rtgs", "swift"),
    "merchant": ("merchant", "mcc", "merchant_code"),
    "currency": ("currency", "ccy", "fx", "forex"),
    "risk": ("risk", "rating", "score", "delinquency", "dpd"),
    "kyc": ("kyc", "pan_no", "aadhaar", "ssn", "tin"),
    "limit": ("limit", "credit_limit", "overdraft_limit"),
}

RETAIL_BANKING_GLOSSARY = {
    "account": ("savings", "current", "checking", "salary_account"),
    "card": ("atm", "debit_card", "credit_card"),
    "channel": ("branch", "atm", "upi", "imps", "neft", "rtgs", "mobile"),
    "balance": ("daily_balance", "avg_balance"),
}

LENDING_GLOSSARY = {
    "loan": ("loan_id", "loan_type", "disbursal", "prepayment", "foreclosure"),
    "risk": ("dpd", "npa", "default", "writeoff"),
    "interest": ("interest_rate", "roi", "apr"),
    "date": ("disbursal_date", "due_date", "maturity_date"),
}

CARDS_GLOSSARY = {
    "card": ("card_bin", "card_type", "issuer"),
    "transaction": ("auth", "authorization", "chargeback", "reversal"),
    "merchant": ("merchant_name", "mcc", "terminal_id"),
    "fee": ("interchange", "scheme_fee"),
}

PAYMENTS_GLOSSARY = {
    "transaction": ("utr", "rrn", "payment_id", "gateway_ref"),
    "channel": ("upi", "imps", "neft", "rtgs", "swift"),
    "status": ("success", "failed", "pending", "reversed"),
    "date": ("settlement_date", "value_date", "posted_date"),
}


def _match_glossary(name: str, glossary: Dict[str, tuple]) -> Dict[str, bool]:
    lowered = name.lower()
    return {key: any(term in lowered for term in terms) for key, terms in glossary.items()}


def _detect_unit(series: pd.Series, name: str) -> Optional[str]:
    lowered = name.lower()
    if "%" in lowered or "percent" in lowered or "pct" in lowered:
        return "percent"
    if any(token in lowered for token in ("amount", "price", "cost", "revenue", "sales")):
        return "currency"
    sample = series.dropna().astype(str).head(50).tolist()
    if any(s.strip().startswith(("$", "EUR", "INR", "USD")) for s in sample):
        return "currency"
    if pd.api.types.is_numeric_dtype(series):
        max_val = pd.to_numeric(series, errors="coerce").max()
        if pd.notna(max_val) and max_val <= 1.2:
            return "ratio"
    return None


def _metric_type(series: pd.Series, name: str) -> str:
    lowered = name.lower()
    if not pd.api.types.is_numeric_dtype(series):
        return "non-additive"
    if any(token in lowered for token in ("rate", "ratio", "percent", "pct", "%")):
        return "non-additive"
    if any(token in lowered for token in ("count", "qty", "quantity", "units")):
        return "additive"
    return "additive"


def infer_semantics(
    df: pd.DataFrame, industry: Optional[str] = None, sub_domain: Optional[str] = None
) -> List[SemanticInference]:
    results: List[SemanticInference] = []
    glossary = dict(BASE_GLOSSARY)
    if industry and industry.lower() == "banking":
        glossary.update(BANKING_GLOSSARY)
        if sub_domain and sub_domain.lower() == "retail banking":
            glossary.update(RETAIL_BANKING_GLOSSARY)
        elif sub_domain and sub_domain.lower() == "lending":
            glossary.update(LENDING_GLOSSARY)
        elif sub_domain and sub_domain.lower() == "cards":
            glossary.update(CARDS_GLOSSARY)
        elif sub_domain and sub_domain.lower() == "payments":
            glossary.update(PAYMENTS_GLOSSARY)
    for col in df.columns:
        series = df[col]
        matches = _match_glossary(col, glossary)
        entity = None
        if matches.get("customer"):
            entity = "Customer"
        elif matches.get("account"):
            entity = "Account"
        elif matches.get("transaction"):
            entity = "Transaction"
        elif matches.get("loan"):
            entity = "Loan"
        elif matches.get("order"):
            entity = "Order"
        elif matches.get("product"):
            entity = "Product"
        elif matches.get("invoice"):
            entity = "Invoice"
        semantic_role = "attribute"
        if matches.get("amount"):
            semantic_role = "amount"
        elif matches.get("quantity"):
            semantic_role = "quantity"
        elif matches.get("rate"):
            semantic_role = "rate"
        metric_type = _metric_type(series, col)
        unit = _detect_unit(series, col)
        confidence = 0.6
        if semantic_role != "attribute" or entity:
            confidence = 0.75
        if unit:
            confidence = min(0.9, confidence + 0.1)
        results.append(
            SemanticInference(
                column=str(col),
                semantic_role=semantic_role,
                entity=entity,
                metric_type=metric_type,
                unit=unit,
                confidence=round(confidence, 2),
            )
        )
    return results
