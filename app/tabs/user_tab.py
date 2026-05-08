import streamlit as st
import pandas as pd
import numpy as np


BRAZILIAN_STATES = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO",
    "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
    "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"
]

PRODUCT_CATEGORIES = [
    "bed_bath_table", "health_beauty", "sports_leisure",
    "computers_accessories", "furniture_decor", "housewares",
    "watches_gifts", "telephony", "auto", "toys",
    "cool_stuff", "garden_tools", "perfumery", "baby",
    "electronics", "pet_shop", "stationery", "books_general_interest",
    "luggage_accessories", "fashion_bags_accessories",
    "small_appliances", "construction_tools_construction",
    "office_furniture", "musical_instruments", "food",
    "unknown"
]

PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

REGION_MAP = {
    "sp": "southeast", "rj": "southeast", "mg": "southeast", "es": "southeast",
    "pr": "south",     "sc": "south",     "rs": "south",
    "mt": "centerwest","ms": "centerwest","go": "centerwest","df": "centerwest",
    "ba": "northeast", "se": "northeast", "al": "northeast", "pe": "northeast",
    "pb": "northeast", "rn": "northeast", "ce": "northeast", "pi": "northeast",
    "ma": "northeast",
    "pa": "north",     "am": "north",     "ac": "north",     "ro": "north",
    "rr": "north",     "ap": "north",     "to": "north",
}


def build_feature_vector(inputs: dict, feature_meta: dict) -> pd.DataFrame:
    f = inputs

    customer_state = f["customer_state"].lower()
    seller_state   = f["seller_state"].lower()
    same_state     = int(customer_state == seller_state)
    c_region       = REGION_MAP.get(customer_state, "other")
    s_region       = REGION_MAP.get(seller_state, "other")
    same_region    = int(c_region == s_region)
    cross_region   = 1 - same_region

    volume = f["length_cm"] * f["height_cm"] * f["width_cm"]
    vol_weight = volume / 5000
    freight_ratio = f["freight"] / f["price"] if f["price"] > 0 else 0
    price_per_item = f["price"] / f["item_count"] if f["item_count"] > 0 else f["price"]
    freight_per_kg = (
        f["freight"] / (f["weight_g"] / 1000)
        if f["weight_g"] > 0 else 0
    )

    row = {
        "purchase_dow"                  : f["purchase_dow"],
        "purchase_month"                : f["purchase_month"],
        "purchase_quarter"              : (f["purchase_month"] - 1) // 3 + 1,
        "purchase_hour"                 : f["purchase_hour"],
        "purchase_is_weekend"           : int(f["purchase_dow"] >= 5),
        "purchase_year"                 : f["purchase_year"],
        "purchase_near_holiday"         : f["purchase_near_holiday"],
        "purchase_black_friday"         : f["purchase_black_friday"],
        "estimated_delivery_days"       : f["estimated_delivery_days"],
        "item_count"                    : f["item_count"],
        "total_price"                   : f["price"],
        "total_freight"                 : f["freight"],
        "avg_price"                     : price_per_item,
        "unique_sellers"                : 1,
        "unique_products"               : 1,
        "product_weight_g"              : f["weight_g"],
        "product_length_cm"             : f["length_cm"],
        "product_height_cm"             : f["height_cm"],
        "product_width_cm"              : f["width_cm"],
        "product_volume_cm3"            : volume,
        "volumetric_weight"             : vol_weight,
        "product_name_lenght"           : f["product_name_length"],
        "product_description_lenght"    : f["product_desc_length"],
        "product_photos_qty"            : f["photos_qty"],
        "freight_per_kg"                : freight_per_kg,
        "freight_price_ratio"           : freight_ratio,
        "price_per_item"                : price_per_item,
        "is_heavy_product"              : int(f["weight_g"] > 5000),
        "is_bulky_product"              : int(volume > 30000),
        "total_payment_value"           : f["price"] + f["freight"],
        "payment_installments"          : f["payment_installments"],
        "payment_methods_used"          : 1,
        "same_state"                    : same_state,
        "same_region"                   : same_region,
        "cross_region"                  : cross_region,
        "customer_state_avg_variance"   : feature_meta.get("global_cat_avg", -11.24),
        "seller_state_avg_variance"     : feature_meta.get("global_cat_avg", -11.24),
        "seller_avg_variance"           : feature_meta.get("global_seller_avg", -11.24),
        "seller_std_variance"           : feature_meta.get("global_seller_std", 9.66),
        "seller_order_count"            : 5,
        "seller_late_rate"              : feature_meta.get("global_seller_late", 0.08),
        "customer_order_count"          : 1,
        "is_repeat_customer"            : 0,
        "category_avg_variance"         : feature_meta.get("global_cat_avg", -11.24),
    }

    features = feature_meta["final_features"]
    return pd.DataFrame([row])[features]


def render(model, feature_meta, validation_results):
    st.markdown(
        "Predict how many days early or late an order will be "
        "delivered relative to the estimated delivery date."
    )

    with st.expander("ℹ️ How to use this tool", expanded=False):
        st.markdown("""
        - Fill in the order details below
        - Click **Predict** to get the estimated delivery variance
        - **Negative** = delivered early (good)
        - **Positive** = delivered late (bad)
        - The 90% confidence interval means the true variance will 
          fall within that range 90% of the time
        """)

    st.divider()

    # ── INPUT FORM ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📍 Location")
        customer_state = st.selectbox(
            "Customer State", BRAZILIAN_STATES, index=BRAZILIAN_STATES.index("SP")
        )
        seller_state = st.selectbox(
            "Seller State", BRAZILIAN_STATES, index=BRAZILIAN_STATES.index("SP")
        )
        estimated_delivery_days = st.slider(
            "Estimated Delivery Window (days)",
            min_value=1, max_value=60, value=23,
            help="Days between order placement and estimated delivery date"
        )

    with col2:
        st.subheader("📦 Product")
        category = st.selectbox("Product Category", sorted(PRODUCT_CATEGORIES))
        price    = st.number_input("Order Price (R$)", min_value=0.0,
                                    max_value=5000.0, value=150.0, step=10.0)
        freight  = st.number_input("Freight Value (R$)", min_value=0.0,
                                    max_value=500.0, value=20.0, step=1.0)
        weight_g = st.number_input("Product Weight (g)", min_value=0,
                                    max_value=30000, value=500, step=100)
        item_count = st.number_input("Number of Items", min_value=1,
                                      max_value=20, value=1, step=1)

    with col3:
        st.subheader("📅 Order Details")
        purchase_month = st.selectbox(
            "Purchase Month",
            list(range(1, 13)),
            format_func=lambda x: MONTH_NAMES[x - 1]
        )
        purchase_dow = st.selectbox(
            "Day of Week",
            list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
        )
        purchase_hour = st.slider("Purchase Hour", 0, 23, 14)
        payment_type  = st.selectbox("Payment Type", PAYMENT_TYPES)
        payment_installments = st.slider("Payment Installments", 1, 12, 1)

    with st.expander("⚙️ Advanced Product Dimensions", expanded=False):
        adv1, adv2, adv3, adv4 = st.columns(4)
        with adv1:
            length_cm = st.number_input("Length (cm)", 1.0, 120.0, 25.0)
        with adv2:
            height_cm = st.number_input("Height (cm)", 1.0, 120.0, 13.0)
        with adv3:
            width_cm  = st.number_input("Width (cm)",  1.0, 120.0, 20.0)
        with adv4:
            photos_qty = st.number_input("Photos", 1, 20, 2)

        desc_length = st.slider("Product Description Length (chars)", 50, 3000, 600)
        name_length = st.slider("Product Name Length (chars)", 10, 70, 50)

    near_holiday    = st.checkbox("Order placed near a Brazilian holiday?")
    black_friday    = st.checkbox("Order placed on Black Friday?")
    purchase_year   = st.selectbox("Purchase Year", [2017, 2018], index=1)

    st.divider()

    # ── PREDICT BUTTON ──────────────────────────────────────────
    if st.button("🔮 Predict Delivery Variance", type="primary", use_container_width=True):
        inputs = {
            "customer_state"         : customer_state,
            "seller_state"           : seller_state,
            "estimated_delivery_days": estimated_delivery_days,
            "price"                  : price,
            "freight"                : freight,
            "weight_g"               : weight_g,
            "item_count"             : item_count,
            "length_cm"              : length_cm,
            "height_cm"              : height_cm,
            "width_cm"               : width_cm,
            "photos_qty"             : photos_qty,
            "product_name_length"    : name_length,
            "product_desc_length"    : desc_length,
            "purchase_month"         : purchase_month,
            "purchase_dow"           : purchase_dow,
            "purchase_hour"          : purchase_hour,
            "purchase_year"          : purchase_year,
            "purchase_near_holiday"  : int(near_holiday),
            "purchase_black_friday"  : int(black_friday),
            "payment_type"           : payment_type,
            "payment_installments"   : payment_installments,
        }

        feature_vector = build_feature_vector(inputs, feature_meta)
        prediction     = model.predict(feature_vector)[0]
        conformal_q    = validation_results["conformal_q_90"]
        lower          = prediction - conformal_q
        upper          = prediction + conformal_q

        # ── RESULT DISPLAY ──────────────────────────────────────
        st.subheader("📊 Prediction Result")

        res1, res2, res3 = st.columns(3)

        with res1:
            if prediction > 0:
                st.metric(
                    "Predicted Variance",
                    f"+{prediction:.1f} days",
                    delta="LATE",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    "Predicted Variance",
                    f"{prediction:.1f} days",
                    delta="EARLY",
                    delta_color="normal"
                )

        with res2:
            st.metric(
                "90% Confidence Interval",
                f"{lower:.1f}d  to  {upper:.1f}d"
            )

        with res3:
            st.metric(
                "Interval Width",
                f"± {conformal_q:.1f} days"
            )

        # Interpretation
        st.divider()
        if prediction > 5:
            st.error(
                f"⚠️ **High risk of late delivery.** "
                f"Predicted {prediction:.1f} days late. "
                f"Consider notifying the customer proactively."
            )
        elif prediction > 0:
            st.warning(
                f"⚠️ **Slight risk of late delivery.** "
                f"Predicted {prediction:.1f} days late."
            )
        elif prediction > -5:
            st.success(
                f"✅ **On-time delivery expected.** "
                f"Predicted {abs(prediction):.1f} days early."
            )
        else:
            st.success(
                f"✅ **Well ahead of schedule.** "
                f"Predicted {abs(prediction):.1f} days early."
            )

        # ── Key Factors ─────────────────────────────────────────
        st.subheader("🔍 Key Factors")
        factor_col1, factor_col2 = st.columns(2)
        with factor_col1:
            same_s = customer_state == seller_state
            st.info(
                f"**Route:** {'Same state ✓' if same_s else 'Cross state ✗'}  \n"
                f"**Window:** {estimated_delivery_days} days estimated  \n"
                f"**Season:** {MONTH_NAMES[purchase_month - 1]}"
            )
        with factor_col2:
            st.info(
                f"**Freight:** R$ {freight:.2f}  \n"
                f"**Weight:** {weight_g}g  \n"
                f"**Category:** {category}"
            )