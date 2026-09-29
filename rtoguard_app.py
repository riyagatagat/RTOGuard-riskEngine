import streamlit as st
import pandas as pd
import joblib

# ============================================================
# RTOGuard AI - Interactive Hackathon Demo
# ============================================================

OLD_MODEL_PATH = "models/rtoguard_old_model.joblib"
NEW_MODEL_PATH = "models/rtoguard_new_model.joblib"

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


@st.cache_resource
def load_models():
    old_model = joblib.load(OLD_MODEL_PATH)
    new_model = joblib.load(NEW_MODEL_PATH)
    return old_model, new_model


def predict_old(model, data):
    X = pd.DataFrame([data], columns=OLD_FEATURES)
    return float(model.predict_proba(X)[0, 1])


def predict_new(model, data):
    X = pd.DataFrame([data], columns=NEW_FEATURES)
    return float(model.predict_proba(X)[0, 1])


def get_risk_decision(customer_type, probability):
    if customer_type == "OLD":
        if probability < 0.10:
            return "LOW", "NORMAL_PROCESSING", (
                "Normal order processing; no additional verification."
            )
        elif probability < 0.50:
            return "MEDIUM", "SOFT_VERIFICATION", (
                "Confirm address/phone and optionally use OTP or delivery confirmation."
            )
        else:
            return "HIGH", "STRONG_VERIFICATION", (
                "Apply stronger verification/intervention before completing the order."
            )

    # NEW customer
    if probability < 0.10:
        return "LOWER_RISK", "NORMAL_PROCESSING_WITH_BASIC_CHECK", (
            "Basic order checks; no strong intervention required."
        )
    elif probability < 0.50:
        return "MEDIUM", "SOFT_VERIFICATION", (
            "Confirm address/phone and optionally use OTP or delivery confirmation."
        )
    else:
        return "HIGH", "STRONG_VERIFICATION", (
            "Apply stronger verification/intervention before completing the order."
        )


st.set_page_config(
    page_title="RTOGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ RTOGuard AI")
st.subheader("AI-Powered COD / RTO Risk Scoring Engine")
st.caption(
    "Predict RTO probability and recommend an appropriate business action "
    "for OLD and NEW customers."
)

try:
    old_model, new_model = load_models()
except Exception as e:
    st.error("Could not load the saved models.")
    st.code(str(e))
    st.info(
        "Make sure this file is inside the rtoguard folder and the models folder "
        "contains rtoguard_old_model.joblib and rtoguard_new_model.joblib."
    )
    st.stop()

st.divider()

customer_type = st.radio(
    "Select customer type",
    ["OLD", "NEW"],
    horizontal=True,
)

st.markdown("### Enter order information")

if customer_type == "OLD":
    col1, col2, col3 = st.columns(3)

    with col1:
        historical_orders = st.number_input(
            "Historical orders", min_value=0, value=10
        )
        historical_rto_orders = st.number_input(
            "Historical RTO orders", min_value=0, value=1
        )
        historical_rto_rate = st.number_input(
            "Historical RTO rate", min_value=0.0, max_value=1.0,
            value=0.10, step=0.01
        )
        address_stability_score = st.number_input(
            "Address stability score", min_value=0.0, max_value=1.0,
            value=0.90, step=0.01
        )
        distinct_addresses_used = st.number_input(
            "Distinct addresses used", min_value=1, value=1
        )
        tenure_months = st.number_input(
            "Tenure (months)", min_value=0, value=12
        )

    with col2:
        orders_per_month = st.number_input(
            "Orders per month", min_value=0.0, value=2.0, step=0.1
        )
        orders_last_90d = st.number_input(
            "Orders in last 90 days", min_value=0, value=5
        )
        prev_cod_orders = st.number_input(
            "Previous COD orders", min_value=0, value=5
        )
        prev_cod_success_rate = st.number_input(
            "Previous COD success rate", min_value=0.0, max_value=1.0,
            value=0.90, step=0.01
        )
        cod_share_history = st.number_input(
            "COD share in history", min_value=0.0, max_value=1.0,
            value=0.50, step=0.01
        )
        usual_order_value = st.number_input(
            "Usual order value (₹)", min_value=0.0, value=800.0, step=50.0
        )

    with col3:
        current_order_value = st.number_input(
            "Current order value (₹)", min_value=0.0, value=700.0, step=50.0
        )
        order_value_ratio = st.number_input(
            "Order value ratio", min_value=0.0, value=0.88, step=0.01
        )
        pincode_valid = st.selectbox("Pincode valid?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        pincode_rto_rate = st.number_input(
            "Pincode RTO rate", min_value=0.0, max_value=1.0,
            value=0.20, step=0.01
        )
        festive_window = st.selectbox(
            "Festive window?", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )
        product_category = st.selectbox(
            "Product category",
            ["fashion", "electronics", "beauty", "grocery", "home", "other"]
        )

    data = {
        "historical_orders": historical_orders,
        "historical_rto_orders": historical_rto_orders,
        "historical_rto_rate": historical_rto_rate,
        "address_stability_score": address_stability_score,
        "distinct_addresses_used": distinct_addresses_used,
        "tenure_months": tenure_months,
        "orders_per_month": orders_per_month,
        "orders_last_90d": orders_last_90d,
        "prev_cod_orders": prev_cod_orders,
        "prev_cod_success_rate": prev_cod_success_rate,
        "cod_share_history": cod_share_history,
        "usual_order_value": usual_order_value,
        "current_order_value": current_order_value,
        "order_value_ratio": order_value_ratio,
        "pincode_valid": pincode_valid,
        "pincode_rto_rate": pincode_rto_rate,
        "festive_window": festive_window,
        "product_category": product_category,
    }

else:
    col1, col2, col3 = st.columns(3)

    with col1:
        address_quality_score = st.number_input(
            "Address quality score", min_value=0.0, max_value=1.0,
            value=0.85, step=0.01
        )
        house_number_present = st.selectbox(
            "House number present?", [1, 0], format_func=lambda x: "Yes" if x else "No"
        )
        landmark_present = st.selectbox(
            "Landmark present?", [1, 0], format_func=lambda x: "Yes" if x else "No"
        )
        address_word_count = st.number_input(
            "Address word count", min_value=1, value=12
        )
        pincode_city_match = st.selectbox(
            "Pincode-city match?", [1, 0], format_func=lambda x: "Yes" if x else "No"
        )

    with col2:
        pincode_valid = st.selectbox(
            "Pincode valid?", [1, 0], format_func=lambda x: "Yes" if x else "No"
        )
        pincode_rto_rate = st.number_input(
            "Pincode RTO rate", min_value=0.0, max_value=1.0,
            value=0.25, step=0.01
        )
        checkout_hour = st.slider(
            "Checkout hour (0-23)", min_value=0, max_value=23, value=14
        )
        night_order = st.selectbox(
            "Night order?", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )
        checkout_duration_sec = st.number_input(
            "Checkout duration (seconds)", min_value=1, value=90
        )

    with col3:
        product_category = st.selectbox(
            "Product category",
            ["fashion", "electronics", "beauty", "grocery", "home", "other"]
        )
        current_order_value = st.number_input(
            "Current order value (₹)", min_value=0.0, value=700.0, step=50.0
        )
        festive_window = st.selectbox(
            "Festive window?", [0, 1], format_func=lambda x: "Yes" if x else "No"
        )

    data = {
        "address_quality_score": address_quality_score,
        "house_number_present": house_number_present,
        "landmark_present": landmark_present,
        "address_word_count": address_word_count,
        "pincode_city_match": pincode_city_match,
        "pincode_valid": pincode_valid,
        "pincode_rto_rate": pincode_rto_rate,
        "checkout_hour": checkout_hour,
        "night_order": night_order,
        "checkout_duration_sec": checkout_duration_sec,
        "product_category": product_category,
        "current_order_value": current_order_value,
        "festive_window": festive_window,
    }

st.divider()

if st.button("🔍 Predict RTO Risk", type="primary", use_container_width=True):
    try:
        if customer_type == "OLD":
            probability = predict_old(old_model, data)
        else:
            probability = predict_new(new_model, data)

        score = round(probability * 100)
        band, action, why = get_risk_decision(customer_type, probability)

        st.markdown("## 🎯 Prediction Result")

        m1, m2, m3 = st.columns(3)
        m1.metric("RTO Probability", f"{probability * 100:.2f}%")
        m2.metric("Risk Score", f"{score}/100")
        m3.metric("Customer Type", customer_type)

        st.progress(min(score, 100) / 100)

        if band == "LOW":
            st.success(f"Risk Band: **{band}**")
        elif band == "LOWER_RISK":
            st.success(f"Risk Band: **{band}**")
        elif band == "MEDIUM":
            st.warning(f"Risk Band: **{band}**")
        else:
            st.error(f"Risk Band: **{band}**")

        st.info(f"**Recommended Action:** {action}")
        st.write(f"**Why:** {why}")

        st.markdown("### Suggested business actions")

        if action == "NORMAL_PROCESSING":
            st.write("• Process the order normally")
            st.write("• No additional verification required")
        elif action == "NORMAL_PROCESSING_WITH_BASIC_CHECK":
            st.write("• Perform basic order checks")
            st.write("• Avoid unnecessary friction for the customer")
        elif action == "SOFT_VERIFICATION":
            st.write("• Confirm phone/address")
            st.write("• Use OTP or delivery confirmation where appropriate")
            st.write("• Keep the order active while verifying")
        else:
            st.write("• Apply stronger verification/intervention")
            st.write("• Consider additional confirmation before fulfillment")
            st.write("• Do not automatically reject the customer")

    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))

st.divider()
st.caption(
    "Prototype decision policy: thresholds are based on the RTOGuard Step 7/8 "
    "analysis. They are not universal business-optimal thresholds."
)
