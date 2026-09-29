"""
RTOGuard AI - STEP 2: Preprocessing + Logistic Regression TRAINING.

Flow:  CLEAN DATA -> TRAIN/TEST SPLIT -> PREPROCESSING -> LOGISTIC REGRESSION
       -> SAVE TRAINED MODELS

NO evaluation, NO thresholds, NO risk scoring happens here.
Put this file in the same folder as the two *_clean.csv files.
Trained pipelines are saved into a "models/" folder next to this file.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- settings (same as the split step) ----------
TEST_SIZE = 0.20
RANDOM_STATE = 42
TARGET = "actual_rto"

# class_weight decision (see the class distribution printed below):
#   None       -> every row counts equally (simple baseline, our default)
#   "balanced" -> RTO rows count more, delivered rows count less
# Change this ONE line later if evaluation shows the model ignores RTO orders.
CLASS_WEIGHT = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ---------- the exact same feature lists as the split step ----------
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

FORBIDDEN = {"actual_rto", "order_id", "customer_id", "pincode",
             "order_date", "customer_type", "payment_mode"}

# text columns that must be one-hot encoded; everything else is numerical
CATEGORICAL_FEATURES = ["product_category"]


def build_pipeline(features):
    """Return (unfitted Pipeline, numerical list, categorical list)."""
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    numerical = [c for c in features if c not in CATEGORICAL_FEATURES]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        )),
    ])
    return pipeline, numerical, categorical


def train_one_model(csv_path, features, model_name, save_name):
    df = pd.read_csv(csv_path)

    X = df[features].copy()
    y = df[TARGET].copy()

    # ---- safety checks (same spirit as the split step) ----
    assert not (FORBIDDEN & set(X.columns)), "Forbidden column found in X!"
    assert X.isna().sum().sum() == 0, "Missing values in X!"
    assert len(X) == len(y)

    # ---- identical split to the previous step ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline, numerical, categorical = build_pipeline(features)

    # every categorical column must really be text, every numerical column numeric
    for c in categorical:
        assert not pd.api.types.is_numeric_dtype(X[c]), f"{c} is not text!"
    for c in numerical:
        assert pd.api.types.is_numeric_dtype(X[c]), f"{c} is not numeric!"

    # ---- report BEFORE training ----
    print(f"\n===== {model_name} =====")
    print(f"Training rows            : {len(X_train)}")
    print(f"Test rows                : {len(X_test)}")
    print(f"Original feature count   : {len(features)}")
    print(f"Numerical features       : {len(numerical)}")
    print(f"Categorical features     : {len(categorical)} {categorical}")

    counts = y_train.value_counts().sort_index()
    pct = y_train.value_counts(normalize=True).sort_index() * 100
    print("Class distribution in y_train:")
    print(f"  0 (delivered): {counts[0]} rows ({pct[0]:.2f}%)")
    print(f"  1 (RTO)      : {counts[1]} rows ({pct[1]:.2f}%)")
    print(f"class_weight in use      : {CLASS_WEIGHT}")

    # ---- TRAIN: fit() learns scaling/encoding from X_train ONLY,
    #      then learns the logistic regression on the transformed X_train ----
    pipeline.fit(X_train, y_train)

    # ---- confirm training worked (no scoring on test data) ----
    n_out = pipeline.named_steps["preprocess"].transform(X_train.head(1)).shape[1]
    n_coef = pipeline.named_steps["model"].coef_.shape[1]
    assert n_out == n_coef, "Preprocessor output does not match model inputs!"
    print(f"Features after preprocessing: {n_out} "
          f"({len(numerical)} scaled + one-hot columns)")
    print(f"[OK] {model_name} trained successfully.")

    # ---- save the whole pipeline (preprocessing + model together) ----
    out_path = MODEL_DIR / save_name
    joblib.dump(pipeline, out_path)
    print(f"Saved -> {out_path}")

    return pipeline, X_train, X_test, y_train, y_test


old_pipeline, X_train_old, X_test_old, y_train_old, y_test_old = train_one_model(
    BASE_DIR / "rtoguard_old_customers_clean.csv",
    OLD_FEATURES, "OLD customer model", "rtoguard_old_model.joblib")

new_pipeline, X_train_new, X_test_new, y_train_new, y_test_new = train_one_model(
    BASE_DIR / "rtoguard_new_customers_clean.csv",
    NEW_FEATURES, "NEW customer model", "rtoguard_new_model.joblib")

print("\nDone. Next stage: model evaluation (separate file).")

