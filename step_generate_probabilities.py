"""
RTOGuard AI - Generate test-set probability files.

Loads the ALREADY-TRAINED models, recreates the exact train/test split used in
training (test_size=0.20, random_state=42, stratify=y), and saves
predict_proba() output for the test rows only.

No retraining, no refitting, no recalibration, no threshold changes.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (relative to the folder containing this script)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Feature lists - identical to training
# ---------------------------------------------------------------------------
OLD_FEATURES = [
    "historical_orders", "historical_rto_orders", "historical_rto_rate",
    "address_stability_score", "distinct_addresses_used",
    "tenure_months", "orders_per_month", "orders_last_90d",
    "prev_cod_orders", "prev_cod_success_rate", "cod_share_history",
    "usual_order_value", "current_order_value", "order_value_ratio",
    "pincode_valid", "pincode_rto_rate",
    "festive_window", "product_category",
]

NEW_FEATURES = [
    "address_quality_score", "house_number_present", "landmark_present",
    "address_word_count", "pincode_city_match",
    "pincode_valid", "pincode_rto_rate",
    "checkout_hour", "night_order", "checkout_duration_sec",
    "product_category", "current_order_value", "festive_window",
]

TARGET = "actual_rto"
EXPECTED_TEST_ROWS = 2400


def generate(label, features, data_path, model_path, out_path):
    print("=" * 70)
    print(f"{label} MODEL")
    print("=" * 70)

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df[features]
    y = df[TARGET]

    # EXACT same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Use the saved pipeline directly (no fit, no recalibration)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Sanity checks
    assert len(X_test) == EXPECTED_TEST_ROWS, f"X_test has {len(X_test)} rows"
    assert len(probabilities) == EXPECTED_TEST_ROWS, f"probabilities has {len(probabilities)} rows"
    assert len(y_test) == EXPECTED_TEST_ROWS, f"y_test has {len(y_test)} rows"

    result = pd.DataFrame({
        "original_index": X_test.index,
        "actual_rto": y_test.values,
        "predicted_probability": probabilities,
    })
    result.to_csv(out_path, index=False)

    # Verification report
    rto = result[result["actual_rto"] == 1]
    delivered = result[result["actual_rto"] == 0]

    print(f"Rows                          : {len(result)}")
    print(f"RTO orders (actual_rto=1)     : {len(rto)}")
    print(f"Delivered orders (actual_rto=0): {len(delivered)}")
    print(f"Min probability               : {result['predicted_probability'].min():.6f}")
    print(f"Max probability               : {result['predicted_probability'].max():.6f}")
    print(f"Mean probability              : {result['predicted_probability'].mean():.6f}")
    print(f"Mean prob (actual RTO = 0)    : {delivered['predicted_probability'].mean():.6f}")
    print(f"Mean prob (actual RTO = 1)    : {rto['predicted_probability'].mean():.6f}")
    print()


def main():
    old_out = BASE_DIR / "old_test_probabilities.csv"
    new_out = BASE_DIR / "new_test_probabilities.csv"

    generate(
        "OLD", OLD_FEATURES,
        BASE_DIR / "rtoguard_old_customers_clean.csv",
        MODELS_DIR / "rtoguard_old_model.joblib",
        old_out,
    )
    generate(
        "NEW", NEW_FEATURES,
        BASE_DIR / "rtoguard_new_customers_clean.csv",
        MODELS_DIR / "rtoguard_new_model.joblib",
        new_out,
    )

    print("=" * 70)
    print("[OK] OLD probability file created")
    print(f"     {old_out}")
    print("[OK] NEW probability file created")
    print(f"     {new_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
