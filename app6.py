import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------
# Load trained model + encoder
# ------------------------------
model = joblib.load("xgb_hdb_price_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")

# Load reference dataset for market averages
# You need to prepare a CSV with town, flat_type, and avg_price
# Example structure: town,flat_type,avg_price
market_ref = pd.read_csv("market_avg.csv")

# ------------------------------
# UI Setup
# ------------------------------
st.set_page_config(page_title="HDB Price Estimator", layout="centered")
st.title("🏘️ Singapore HDB Resale Price Estimator")
st.markdown("Estimate resale prices of HDB flats with ML + compare with market averages.")

# ------------------------------
# Sidebar: Buyer or Seller
# ------------------------------
mode = st.sidebar.radio("Select mode:", ["Buyer", "Seller"])

# ------------------------------
# Static dropdown options
# ------------------------------
towns = market_ref["town"].unique().tolist()
flat_types = market_ref["flat_type"].unique().tolist()
flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified', 'Premium Apartment',
               'Maisonette', 'Apartment', 'DBSS', 'Type S1', 'Type S2']

storey_ranges = {
    '01 TO 03': 2, '04 TO 06': 5, '07 TO 09': 8, '10 TO 12': 11,
    '13 TO 15': 14, '16 TO 18': 17, '19 TO 21': 20, '22 TO 24': 23,
    '25 TO 27': 26, '28 TO 30': 29, '31 TO 33': 32, '34 TO 36': 35,
    '37 TO 39': 38, '40 TO 42': 41, '43 TO 45': 44, '46 TO 48': 47,
    '49 TO 51': 50
}

def map_floor_to_median(floor_input):
    for rng, median in storey_ranges.items():
        low, high = map(int, rng.split(" TO "))
        if low <= floor_input <= high:
            return median
    return np.nan

# ------------------------------
# Input Form
# ------------------------------
with st.form("input_form"):
    town = st.selectbox("Town", towns)
    flat_type = st.selectbox("Flat Type", flat_types)
    flat_model = st.selectbox("Flat Model", flat_models)

    floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=30.0, max_value=200.0, value=80.0, step=1.0)
    floor_input = st.number_input("Storey", min_value=1, max_value=51, value=5, step=1)
    cbd_dist = st.number_input("Distance to CBD (km)", min_value=0.0, max_value=25.0, value=8.0, step=1.0)
    lease_year = st.number_input("Lease Commencement Year", min_value=1960, max_value=2025, value=2000, step=1)

    if mode == "Buyer":
        budget = st.number_input("💰 Your Budget (SGD)", min_value=50000.0, max_value=2000000.0, step=1000.0)
    else:
        asking_price = st.number_input("💰 Your Asking Price (SGD)", min_value=50000.0, max_value=2000000.0, step=1000.0)

    submitted = st.form_submit_button("Estimate Price")

# ------------------------------
# Prediction Logic
# ------------------------------
if submitted:
    storey_median = map_floor_to_median(int(floor_input))

    if np.isnan(storey_median):
        st.error("Invalid floor number. Must be between 1 and 51.")
    else:
        # Prepare input
        user_df = pd.DataFrame([{
            'floor_area_sqm': floor_area_sqm,
            'storey_median': storey_median,
            'cbd_dist': cbd_dist,
            'lease_commence_date': lease_year,
            'town': town,
            'flat_model': flat_model,
            'flat_type': flat_type,
            # Keep economic indicators fixed (latest known values)
            'GDP': 510000.0,
            'GDP per Capita': 85000.0,
            'GNI per Capita': 79000.0,
            'Population': 5600000,
            'Resident Household': 1200000,
            'HDB Dwellings': 1000000,
            'Inflation Rate': 1.5,
            'Interest Rate': 3.0,
            'Unemployment Rate': 2.1,
            'Yield_5y_interest': 2.5
        }])

        # Encode categorical
        user_df[['town', 'flat_model', 'flat_type']] = encoder.transform(
            user_df[['town', 'flat_model', 'flat_type']]
        )

        # Prediction
        prediction = model.predict(user_df)[0]
        st.success(f"🏠 Estimated Resale Price: SGD {prediction:,.2f}")

        # Market comparison
        avg_price = market_ref.query("town == @town and flat_type == @flat_type")["avg_price"].mean()
        if not pd.isna(avg_price):
            st.info(f"📊 Average resale price for {flat_type} in {town}: SGD {avg_price:,.0f}")
            diff = ((prediction - avg_price) / avg_price) * 100
            st.write(f"Your estimate is **{diff:+.1f}%** compared to market average.")

        # Mode-specific feedback
        if mode == "Buyer":
            if budget >= prediction:
                st.success(f"✅ This property is **within your budget** (Budget: SGD {budget:,.0f}).")
            else:
                st.warning(f"⚠️ This property **exceeds your budget** (Budget: SGD {budget:,.0f}).")
        else:  # Seller
            diff = ((asking_price - prediction) / prediction) * 100
            if abs(diff) < 5:
                st.success("✅ Your asking price is in line with the model estimate.")
            elif diff > 5:
                st.warning(f"⚠️ Your asking price is **{diff:.1f}% above** the model estimate.")
            else:
                st.info(f"ℹ️ Your asking price is **{diff:.1f}% below** the model estimate.")
