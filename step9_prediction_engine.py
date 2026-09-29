"""
RTOGuard AI - STEP 9: FINAL PREDICTION ENGINE

Uses the ACTUAL saved Logistic Regression pipelines:
    models/rtoguard_old_model.joblib
    models/rtoguard_new_model.joblib

No retraining.
No new scaler.
No new encoder.
No changes to the trained model.

Step 8 prototype business policy:
OLD:
    p < 0.10       -> LOW       -> NORMAL_PROCESSING
    0.10 <= p < .50 -> MEDIUM    -> SOFT_VERIFICATION
    p >= 0.50      -> HIGH      -> STRONG_VERIFICATION

NEW:
    p < 0.10       -> LOWER_RISK -> NORMAL_PROCESSING_WITH_BASIC_CHECK
    0.10 <= p < .50 -> MEDIUM     -> SOFT_VERIFICATION
    p >= 0.50      -> HIGH       -> STRONG_VERIFICATION
"""

from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "OLD": MODEL_DIR / "rtoguard_old_model.joblib",
    "NEW": MODEL_DIR / "rtoguard_new_model.joblib",
}


OLD_FEATURES = [
    "historical_orders",
    "historical_rto_orders",
    "historical_rto_rate",
    "address_stability_score",
    "distinct_addresses_used",
    "tenure_months",
    "orders_per_month",
    "orders_last_90d",
    "prev_cod_orders",
    "prev_cod_success_rate",
    "cod_share_history",
    "usual_order_value",
    "current_order_value",
    "order_value_ratio",
    "pincode_valid",
    "pincode_rto_rate",
    "festive_window",
    "product_category",
]

NEW_FEATURES = [
    "address_quality_score",
    "house_number_present",
    "landmark_present",
    "address_word_count",
    "pincode_city_match",
    "pincode_valid",
    "pincode_rto_rate",
    "checkout_hour",
    "night_order",
    "checkout_duration_sec",
    "product_category",
    "current_order_value",
    "festive_window",
]


def load_model(customer_type):
    """Load the already-trained pipeline."""
    path = MODEL_FILES[customer_type]

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{path}\n\n"
            "Make sure the models folder is inside the rtoguard folder."
        )

    return joblib.load(path)


def ask_number(name, integer=False):
    """Ask for a numeric value and keep asking until valid."""
    while True:
        value = input(f"Enter {name}: ").strip()

        if value == "":
            print("Please enter a value.")
            continue

        try:
            number = int(value) if integer else float(value)
            return number
        except ValueError:
            print("Please enter a valid number.")


def ask_binary(name):
    """Ask for a Yes/No value and return 1/0."""
    while True:
        value = input(f"{name} (1=Yes, 0=No): ").strip()

        if value in ("0", "1"):
            return int(value)

        print("Please enter only 1 or 0.")


def ask_text(name):
    while True:
        value = input(f"Enter {name}: ").strip()

        if value:
            return value

        print("Please enter a value.")


def get_old_input():
    """Collect the 18 OLD customer features."""
    print("\n" + "=" * 60)
    print("OLD CUSTOMER INPUT")
    print("=" * 60)

    data = {
        "historical_orders": ask_number("historical orders", integer=True),
        "historical_rto_orders": ask_number("historical RTO orders", integer=True),
        "historical_rto_rate": ask_number("historical RTO rate"),
        "address_stability_score": ask_number("address stability score"),
        "distinct_addresses_used": ask_number("distinct addresses used", integer=True),
        "tenure_months": ask_number("tenure months", integer=True),
        "orders_per_month": ask_number("orders per month"),
        "orders_last_90d": ask_number("orders in last 90 days", integer=True),
        "prev_cod_orders": ask_number("previous COD orders", integer=True),
        "prev_cod_success_rate": ask_number("previous COD success rate"),
        "cod_share_history": ask_number("historical COD share"),
        "usual_order_value": ask_number("usual order value"),
        "current_order_value": ask_number("current order value"),
        "order_value_ratio": ask_number("order value ratio"),
        "pincode_valid": ask_binary("Pincode valid"),
        "pincode_rto_rate": ask_number("Pincode RTO rate"),
        "festive_window": ask_binary("Festive window"),
        "product_category": ask_text("product category"),
    }

    return pd.DataFrame([data], columns=OLD_FEATURES)


def get_new_input():
    """Collect the 13 NEW customer features."""
    print("\n" + "=" * 60)
    print("NEW CUSTOMER INPUT")
    print("=" * 60)

    data = {
        "address_quality_score": ask_number("address quality score"),
        "house_number_present": ask_binary("House number present"),
        "landmark_present": ask_binary("Landmark present"),
        "address_word_count": ask_number("address word count", integer=True),
        "pincode_city_match": ask_binary("Pincode-city match"),
        "pincode_valid": ask_binary("Pincode valid"),
        "pincode_rto_rate": ask_number("Pincode RTO rate"),
        "checkout_hour": ask_number("checkout hour (0-23)", integer=True),
        "night_order": ask_binary("Night order"),
        "checkout_duration_sec": ask_number("checkout duration in seconds"),
        "product_category": ask_text("product category"),
        "current_order_value": ask_number("current order value"),
        "festive_window": ask_binary("Festive window"),
    }

    if not 0 <= data["checkout_hour"] <= 23:
        raise ValueError("Checkout hour must be between 0 and 23.")

    return pd.DataFrame([data], columns=NEW_FEATURES)


def get_risk_decision(customer_type, probability):
    """Apply the exact Step 8 prototype policy."""
    risk_score = round(probability * 100)

    if customer_type == "OLD":
        if probability < 0.10:
            return (
                risk_score,
                "LOW",
                "NORMAL_PROCESSING",
                "Normal order processing; no additional verification."
            )
        elif probability < 0.50:
            return (
                risk_score,
                "MEDIUM",
                "SOFT_VERIFICATION",
                "Confirm address/phone and optionally use OTP or delivery confirmation."
            )
        else:
            return (
                risk_score,
                "HIGH",
                "STRONG_VERIFICATION",
                "Strong verification; confirm COD and consider prepaid/manual review."
            )

    else:  # NEW
        if probability < 0.10:
            return (
                risk_score,
                "LOWER_RISK",
                "NORMAL_PROCESSING_WITH_BASIC_CHECK",
                "Basic validation; NEW customers should not be treated as completely safe."
            )
        elif probability < 0.50:
            return (
                risk_score,
                "MEDIUM",
                "SOFT_VERIFICATION",
                "Confirm address/phone and optionally use OTP or delivery confirmation."
            )
        else:
            return (
                risk_score,
                "HIGH",
                "STRONG_VERIFICATION",
                "Strong verification; confirm COD and consider prepaid/manual review."
            )


def predict(customer_type, X):
    """Run the actual saved model."""
    model = load_model(customer_type)

    # The saved pipeline performs the original preprocessing.
    probability = float(model.predict_proba(X)[0, 1])

    risk_score, band, action, explanation = get_risk_decision(
        customer_type, probability
    )

    return probability, risk_score, band, action, explanation


def print_result(customer_type, probability, risk_score, band, action, explanation):
    print("\n")
    print("=" * 60)
    print("RTOGuard AI - ORDER RISK RESULT")
    print("=" * 60)

    print(f"Customer Type    : {customer_type}")
    print(f"RTO Probability  : {probability:.2%}")
    print(f"Risk Score       : {risk_score}/100")
    print(f"Risk Band        : {band}")
    print(f"Recommended Action: {action}")

    print("\nWhy:")
    print(explanation)

    print("\nSuggested business actions:")

    if action == "NORMAL_PROCESSING":
        print("  • Process the order normally")
        print("  • No additional verification required")

    elif action == "NORMAL_PROCESSING_WITH_BASIC_CHECK":
        print("  • Process normally with basic validation")
        print("  • Do not assume the order is completely risk-free")

    elif action == "SOFT_VERIFICATION":
        print("  • Confirm phone/address")
        print("  • Use OTP or delivery confirmation where appropriate")
        print("  • Keep the order active while verifying")

    elif action == "STRONG_VERIFICATION":
        print("  • Confirm customer phone/address")
        print("  • Confirm COD order")
        print("  • Consider prepaid payment")
        print("  • Consider manual review for selected cases")

    print("=" * 60)


def choose_customer_type():
    while True:
        print("\n" + "=" * 60)
        print("RTOGuard AI - CUSTOMER TYPE")
        print("=" * 60)
        print("1. OLD Customer")
        print("2. NEW Customer")

        choice = input("Choose 1 or 2: ").strip()

        if choice == "1":
            return "OLD"
        if choice == "2":
            return "NEW"

        print("Invalid choice. Please enter 1 or 2.")


def run_prediction():
    customer_type = choose_customer_type()

    if customer_type == "OLD":
        X = get_old_input()
    else:
        X = get_new_input()

    probability, risk_score, band, action, explanation = predict(
        customer_type, X
    )

    print_result(
        customer_type,
        probability,
        risk_score,
        band,
        action,
        explanation
    )


def run_demo():
    """
    Demo mode uses manually chosen feature values.
    IMPORTANT: the model still generates the probability.
    No probability is manually assigned.
    """

    print("\n" + "=" * 60)
    print("RTOGuard AI - DEMO MODE")
    print("=" * 60)
    print("The examples below are illustrative inputs.")
    print("The saved models generate the probabilities.\n")

    old_example = pd.DataFrame([{
        "historical_orders": 20,
        "historical_rto_orders": 1,
        "historical_rto_rate": 0.05,
        "address_stability_score": 0.95,
        "distinct_addresses_used": 1,
        "tenure_months": 18,
        "orders_per_month": 1.2,
        "orders_last_90d": 4,
        "prev_cod_orders": 12,
        "prev_cod_success_rate": 0.95,
        "cod_share_history": 0.60,
        "usual_order_value": 1000,
        "current_order_value": 900,
        "order_value_ratio": 0.90,
        "pincode_valid": 1,
        "pincode_rto_rate": 0.08,
        "festive_window": 0,
        "product_category": "clothing",
    }], columns=OLD_FEATURES)

    new_example = pd.DataFrame([{
        "address_quality_score": 0.85,
        "house_number_present": 1,
        "landmark_present": 1,
        "address_word_count": 12,
        "pincode_city_match": 1,
        "pincode_valid": 1,
        "pincode_rto_rate": 0.10,
        "checkout_hour": 14,
        "night_order": 0,
        "checkout_duration_sec": 120,
        "product_category": "clothing",
        "current_order_value": 800,
        "festive_window": 0,
    }], columns=NEW_FEATURES)

    examples = [
        ("OLD", old_example),
        ("NEW", new_example),
    ]

    for customer_type, X in examples:
        try:
            result = predict(customer_type, X)
            print_result(customer_type, *result)
        except Exception as e:
            print(f"\n[DEMO ERROR] {customer_type}: {e}")


def main():
    print("\n" + "=" * 60)
    print("RTOGuard AI - FINAL PREDICTION ENGINE")
    print("=" * 60)
    print("1. Predict a new order")
    print("2. Run demo mode")

    choice = input("\nChoose 1 or 2: ").strip()

    try:
        if choice == "1":
            run_prediction()
        elif choice == "2":
            run_demo()
        else:
            print("Invalid choice. Please run the program again.")
    except Exception as e:
        print("\n[ERROR]")
        print(e)


if __name__ == "__main__":
    main()
