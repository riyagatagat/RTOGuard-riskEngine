"""
RTOGuard AI - STEP 8
Final Prototype Risk Bands + Business Action Rules

IMPORTANT:
The boundaries below are prototype/business-policy thresholds derived from
Step 7 exploratory analysis. They are NOT universally optimal thresholds.

OLD and NEW customers are handled separately in the decision logic because
the models have different discrimination characteristics.
"""

import pandas as pd
from pathlib import Path

# ============================================================
# STEP 8 PROTOTYPE RISK POLICY
# ============================================================

# OLD CUSTOMER
# Step 7 exploratory action structure used 0.10 and 0.50 as boundaries.
OLD_BANDS = [
    {
        "min_probability": 0.00,
        "max_probability": 0.10,
        "risk_band": "LOW",
        "action": "NORMAL_PROCESSING",
        "description": "Normal order processing; no additional verification."
    },
    {
        "min_probability": 0.10,
        "max_probability": 0.50,
        "risk_band": "MEDIUM",
        "action": "SOFT_VERIFICATION",
        "description": "Confirm address/phone and optionally send OTP or delivery confirmation."
    },
    {
        "min_probability": 0.50,
        "max_probability": 1.01,
        "risk_band": "HIGH",
        "action": "STRONG_VERIFICATION",
        "description": "Strong verification; consider COD confirmation, OTP and prepaid recommendation."
    }
]

# NEW CUSTOMER
# Same numerical prototype boundaries are retained only because Step 7
# explicitly explored these boundaries for NEW as well. The BUSINESS ACTION
# is intentionally more cautious because NEW has weaker low-risk discrimination.
NEW_BANDS = [
    {
        "min_probability": 0.00,
        "max_probability": 0.10,
        "risk_band": "LOWER_RISK",
        "action": "NORMAL_PROCESSING_WITH_BASIC_CHECK",
        "description": "Basic validation; do not treat as completely safe because NEW customers have no history."
    },
    {
        "min_probability": 0.10,
        "max_probability": 0.50,
        "risk_band": "MEDIUM",
        "action": "SOFT_VERIFICATION",
        "description": "Confirm address/phone, use OTP or delivery confirmation where appropriate."
    },
    {
        "min_probability": 0.50,
        "max_probability": 1.01,
        "risk_band": "HIGH",
        "action": "STRONG_VERIFICATION",
        "description": "Strong verification; confirm COD and consider prepaid/manual review for selected cases."
    }
]


def get_risk_decision(customer_type, probability):
    """
    Convert model probability into:
    risk_score, risk_band, business_action, explanation
    """

    customer_type = str(customer_type).strip().upper()
    probability = float(probability)

    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1.")

    risk_score = round(probability * 100)

    if customer_type == "OLD":
        bands = OLD_BANDS
    elif customer_type == "NEW":
        bands = NEW_BANDS
    else:
        raise ValueError("customer_type must be 'OLD' or 'NEW'.")

    for band in bands:
        if band["min_probability"] <= probability < band["max_probability"]:
            return {
                "customer_type": customer_type,
                "predicted_probability": round(probability, 4),
                "risk_score": risk_score,
                "risk_band": band["risk_band"],
                "business_action": band["action"],
                "explanation": band["description"]
            }

    raise ValueError("Could not assign a risk band.")


# ============================================================
# DEMO EXAMPLES
# ============================================================

print("=" * 70)
print("RTOGuard AI - STEP 8 RISK ENGINE")
print("=" * 70)

examples = [
    ("OLD", 0.08),
    ("OLD", 0.35),
    ("OLD", 0.72),
    ("NEW", 0.08),
    ("NEW", 0.35),
    ("NEW", 0.72),
]

print("\nDEMO PREDICTIONS\n")

for customer_type, probability in examples:
    result = get_risk_decision(customer_type, probability)

    print(f"Customer Type : {result['customer_type']}")
    print(f"Probability   : {result['predicted_probability']}")
    print(f"Risk Score    : {result['risk_score']}/100")
    print(f"Risk Band     : {result['risk_band']}")
    print(f"Action        : {result['business_action']}")
    print(f"Meaning       : {result['explanation']}")
    print("-" * 70)


# ============================================================
# APPLY STEP 8 TO STEP 5 PROBABILITY FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

probability_files = {
    "OLD": BASE_DIR / "old_test_probabilities.csv",
    "NEW": BASE_DIR / "new_test_probabilities.csv",
}

for customer_type, file_path in probability_files.items():

    if not file_path.exists():
        print(f"\n[SKIP] {file_path.name} not found.")
        continue

    df = pd.read_csv(file_path)

    required_columns = {
        "original_index",
        "actual_rto",
        "predicted_probability"
    }

    missing = required_columns - set(df.columns)

    if missing:
        print(f"\n[ERROR] {file_path.name} missing columns: {sorted(missing)}")
        continue

    decisions = df["predicted_probability"].apply(
        lambda p: get_risk_decision(customer_type, p)
    )

    decision_df = pd.DataFrame(list(decisions))

    # Keep useful original information + Step 8 decisions
    output = pd.concat(
        [
            df[["original_index", "actual_rto", "predicted_probability"]].reset_index(drop=True),
            decision_df[
                [
                    "risk_score",
                    "risk_band",
                    "business_action",
                    "explanation"
                ]
            ].reset_index(drop=True)
        ],
        axis=1
    )

    output_file = BASE_DIR / f"{customer_type.lower()}_step8_risk_decisions.csv"
    output.to_csv(output_file, index=False)

    print(f"\n[OK] {customer_type} Step 8 file created:")
    print(f"     {output_file}")


# ============================================================
# POLICY SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 POLICY SUMMARY")
print("=" * 70)

print("""
OLD CUSTOMERS
0 <= p < 0.10  -> LOW
                 -> Normal processing

0.10 <= p < 0.50 -> MEDIUM
                   -> Soft verification

p >= 0.50        -> HIGH
                   -> Strong verification


NEW CUSTOMERS
0 <= p < 0.10  -> LOWER_RISK
                 -> Basic validation + normal processing

0.10 <= p < 0.50 -> MEDIUM
                   -> Soft verification

p >= 0.50        -> HIGH
                   -> Strong verification


IMPORTANT:
These are prototype business-policy bands based on Step 7 exploratory
analysis. They should be validated with real RTO costs, intervention
effectiveness, operational review capacity and future production data.

The system should assist business decisions, not automatically reject
customers solely because of a predicted risk score.
""")
