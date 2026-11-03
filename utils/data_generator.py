"""
utils/data_generator.py
Synthetic funeral insurance claims dataset for SADC context.
Based on fraud typologies from Manje (2005) and Debener et al. (2023).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)


# ── Country/region metadata ──────────────────────────────────────────────────
SADC_REGIONS = [
    "Lusaka", "Copperbelt", "Central", "Eastern", "Northern",
    "Western", "Southern", "North-Western", "Harare", "Bulawayo",
    "Gauteng", "Western Cape", "KZN", "Gaborone", "Windhoek",
    "Maputo", "Dar es Salaam", "Lilongwe",
]

CAUSE_OF_DEATH = [
    "Natural Causes", "Cardiac Arrest", "Respiratory Failure",
    "Motor Vehicle Accident", "Cancer", "Stroke", "Diabetes Complications",
    "Pneumonia", "Tuberculosis", "Malaria",
]

DOCUMENT_STATUSES = [
    "Complete", "Incomplete", "Suspicious", "Duplicate Reference",
    "Missing Death Cert", "Forged Certificate (Flagged)",
]


def generate_claims_dataset(n: int = 2000) -> pd.DataFrame:
    """
    Generate a synthetic funeral insurance claims dataset with realistic
    fraud patterns for the SADC region.

    Fraud prevalence: ~4.5% (consistent with 1-5% literature estimate).
    """
    now = datetime.now()

    # ── Base features ─────────────────────────────────────────────────────────
    policy_ages_months = rng.integers(1, 120, n)          # policy age at claim
    claim_amounts = rng.lognormal(9.0, 0.6, n)            # ~USD 1,000–50,000
    claimant_ages = rng.integers(18, 85, n)
    beneficiary_count = rng.integers(1, 6, n)
    days_to_claim = rng.integers(1, 90, n)                # days death→claim
    prior_claims = rng.integers(0, 5, n)
    num_policies_same_beneficiary = rng.integers(1, 4, n)
    region = rng.choice(SADC_REGIONS, n)
    cause_of_death = rng.choice(CAUSE_OF_DEATH, n)

    # Documentation completeness score (0–1)
    doc_score = rng.beta(5, 2, n)

    # Whether death cert was verified against national registry
    registry_verified = rng.choice([0, 1], n, p=[0.35, 0.65])

    # Insurer staff relationship flag
    staff_relationship = rng.choice([0, 1], n, p=[0.92, 0.08])

    # Same-day claims (funeral parlour collusion risk)
    same_day_claim = (days_to_claim <= 1).astype(int)

    # Cross-insurer duplicate flag
    cross_insurer_flag = rng.choice([0, 1], n, p=[0.93, 0.07])

    # ── Fraud label: rule-based with noise (mimics real world) ───────────────
    fraud_score = (
        (policy_ages_months < 6).astype(float) * 0.30          # new policy
        + (claim_amounts > 30000).astype(float) * 0.20          # high value
        + (prior_claims >= 2).astype(float) * 0.25              # repeat claims
        + (doc_score < 0.4).astype(float) * 0.30                # poor docs
        + (registry_verified == 0).astype(float) * 0.20         # no verification
        + (staff_relationship == 1).astype(float) * 0.25        # internal
        + (cross_insurer_flag == 1).astype(float) * 0.30        # duplicate
        + (same_day_claim == 1).astype(float) * 0.15            # fast claim
        + (num_policies_same_beneficiary >= 3).astype(float) * 0.20
        + rng.normal(0, 0.08, n)                                 # noise
    )
    fraud_score = np.clip(fraud_score, 0, 1)
    is_fraud = (fraud_score > 0.55).astype(int)

    # Actual fraud prevalence tweak to stay ~4-5%
    fraud_indices = np.where(is_fraud == 1)[0]
    non_fraud_indices = np.where(is_fraud == 0)[0]
    target_fraud = int(n * 0.045)
    if len(fraud_indices) > target_fraud:
        flip_back = rng.choice(fraud_indices, len(fraud_indices) - target_fraud, replace=False)
        is_fraud[flip_back] = 0

    # ── Derived features ──────────────────────────────────────────────────────
    claim_dates = [now - timedelta(days=int(d)) for d in rng.integers(0, 365, n)]
    claim_ids = [f"FIS-{2024+i//500}-{str(i).zfill(5)}" for i in range(n)]

    doc_status = []
    for i in range(n):
        if is_fraud[i] == 1:
            st = rng.choice(
                ["Suspicious", "Duplicate Reference", "Missing Death Cert",
                 "Forged Certificate (Flagged)", "Incomplete"],
                p=[0.25, 0.25, 0.20, 0.15, 0.15],
            )
        else:
            st = rng.choice(
                ["Complete", "Incomplete", "Suspicious"],
                p=[0.80, 0.15, 0.05],
            )
        doc_status.append(st)

    df = pd.DataFrame(
        {
            "claim_id": claim_ids,
            "claim_date": claim_dates,
            "region": region,
            "cause_of_death": cause_of_death,
            "policy_age_months": policy_ages_months,
            "claim_amount_usd": np.round(claim_amounts, 2),
            "claimant_age": claimant_ages,
            "beneficiary_count": beneficiary_count,
            "days_to_claim": days_to_claim,
            "prior_claims": prior_claims,
            "num_policies_same_beneficiary": num_policies_same_beneficiary,
            "documentation_score": np.round(doc_score, 3),
            "document_status": doc_status,
            "registry_verified": registry_verified,
            "staff_relationship_flag": staff_relationship,
            "same_day_claim": same_day_claim,
            "cross_insurer_duplicate": cross_insurer_flag,
            "raw_fraud_score": np.round(fraud_score, 4),
            "is_fraud": is_fraud,
        }
    )

    return df


def get_fraud_type_breakdown(df: pd.DataFrame) -> dict:
    """Categorise fraud cases into typology per Manje (2005)."""
    fraud_df = df[df["is_fraud"] == 1].copy()
    total = len(fraud_df)
    if total == 0:
        return {}

    typology = {
        "Forged/Manipulated Death Certificate": int(
            (fraud_df["document_status"].isin(
                ["Forged Certificate (Flagged)", "Suspicious"]
            )).sum()
        ),
        "Cross-Insurer Duplicate Claims": int(fraud_df["cross_insurer_duplicate"].sum()),
        "Policy Backdating (Short Tenure)": int((fraud_df["policy_age_months"] < 4).sum()),
        "Funeral Parlour Collusion": int(fraud_df["same_day_claim"].sum()),
        "Internal/Staff-Assisted Fraud": int(fraud_df["staff_relationship_flag"].sum()),
    }
    return typology


if __name__ == "__main__":
    df = generate_claims_dataset(2000)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(df.head(3).to_string())
