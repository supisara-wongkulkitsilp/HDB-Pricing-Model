import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved model and encoder
model = joblib.load("hdb_price_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")

# Define categorical options (adjust to your actual dataset categories if needed)
towns = ['Ang Mo Kio', 'Bedok', 'Bishan', 'Bukit Batok', 'Bukit Merah', 'Bukit Panjang', 'Bukit Timah', 'Central Area', 'Choa Chu Kang', 'Clementi', 'Geylang', 'Hougang', 'Jurong East', 'Jurong West', 'Kallang/Whampoa', 'Marine Parade', 'Pasir Ris', 'Punggol', 'Queenstown', 'Sembawang', 'Sengkang', 'Serangoon', 'Tampines', 'Toa Payoh', 'Woodlands', 'Yishun']

flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified']
closest_mrts = ['Mayflower MRT Station', 'Ang Mo Kio MRT Station',
 'Yio Chu Kang MRT Station', 'Lentor MRT Station',
 'Bedok Reservoir MRT Station', 'Kembangan MRT Station',
 'Kaki Bukit MRT Station', 'Tanah Merah MRT Station',
 'Bedok North MRT Station', 'Ubi MRT Station', 'Bishan MRT Station',
 'Braddell MRT Station', 'Upper Thomson MRT Station',
 'Marymount MRT Station', 'Bukit Batok MRT Station',
 'Chinese Garden MRT Station', 'Bukit Gombak MRT Station',
 'Tiong Bahru MRT Station', 'Outram Park MRT Station',
 'Queenstown MRT Station', 'HarbourFront MRT Station', 'Redhill MRT Station',
 'Labrador Park MRT Station', 'Telok Blangah MRT Station',
 'Bukit Panjang MRT Station', 'Cashew MRT Station',
 'Beauty World MRT Station', 'Farrer Road MRT Station', 'Bugis MRT Station',
 'Jalan Besar MRT Station', 'Chinatown MRT Station',
 'Tanjong Pagar MRT Station', 'Little India MRT Station',
 'Choa Chu Kang MRT Station', 'Yew Tee MRT Station', 'Clementi MRT Station',
 'Dover MRT Station', 'Eunos MRT Station', 'Aljunied MRT Station',
 'Mountbatten MRT Station', 'Mattar MRT Station', 'Paya Lebar MRT Station',
 'MacPherson MRT Station', 'Geylang Bahru MRT Station', 'Dakota MRT Station',
 'Tai Seng MRT Station', 'Kovan MRT Station', 'Serangoon MRT Station',
 'Hougang MRT Station', 'Buangkok MRT Station', 'Jurong East MRT Station',
 'Lakeside MRT Station', 'Boon Lay MRT Station', 'Pioneer MRT Station',
 'Nicoll Highway MRT Station', 'Farrer Park MRT Station',
 'Boon Keng MRT Station', 'Toa Payoh MRT Station', 'Kallang MRT Station',
 'Esplanade MRT Station', 'Lavender MRT Station',
 'Pasir Panjang MRT Station', 'Pasir Ris MRT Station',
 'Tampines East MRT Station', 'Buona Vista MRT Station',
 'Commonwealth MRT Station', 'Holland Village MRT Station',
 'one-north MRT Station', 'Lorong Chuan MRT Station', 'Simei MRT Station',
 'Tampines MRT Station', 'Upper Changi MRT Station', 'Bartley MRT Station',
 'Potong Pasir MRT Station', 'Woodleigh MRT Station',
 'Marsiling MRT Station', 'Woodlands MRT Station',
 'Woodlands North MRT Station', 'Admiralty MRT Station',
 'Yishun MRT Station', 'Canberra MRT Station', 'Khatib MRT Station',
 'Bras Basah MRT Station', 'Rochor MRT Station',
 'Botanic Gardens MRT Station', 'Bencoolen MRT Station',
 'Clarke Quay MRT Station', 'King Albert Park MRT Station',
 'Novena MRT Station', 'Bendemeer MRT Station', 'Changi Airport MRT Station',
 'Woodlands South MRT Station', 'Sembawang MRT Station',
 'Sengkang MRT Station', 'Caldecott MRT Station', 'Punggol MRT Station']



st.title("🏠 HDB Resale Price Estimator")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", towns)
        flat_type = st.selectbox("Flat Type", flat_types)
        flat_model = st.selectbox("Flat Model", flat_models)
        floor_area = st.number_input("Floor Area (sqm)", 20.0, 200.0, 90.0)
        storey = st.number_input("Storey (median)", 1, 50, 10)
        lease_year = st.number_input("Lease Commence Year", 1960, 2025, 2000)

    with col2:
        latitude = st.number_input("Latitude", 1.2, 1.5, 1.3)
        longitude = st.number_input("Longitude", 103.6, 104.0, 103.8)
        cbd_dist = st.number_input("CBD Distance (km)", 0.0, 30.0, 10.0)
        mrt = st.selectbox("Closest MRT", closest_mrts)
        gdp = st.number_input("GDP (in million SGD)", 300000, 700000, 500000)
        gdp_per_capita = st.number_input("GDP per Capita", 40000, 100000, 70000)
        gni_per_capita = st.number_input("GNI per Capita", 40000, 100000, 70000)
        population = st.number_input("Population", 4000000, 6000000, 5500000)

    # More economic variables
    household = st.number_input("Resident Households", 1000000, 2000000, 1500000)
    hdb_dwellings = st.number_input("HDB Dwellings", 800000, 1200000, 1000000)
    inflation = st.number_input("Inflation Rate (%)", -2.0, 10.0, 2.0)
    interest_rate = st.number_input("Interest Rate (%)", 0.0, 10.0, 3.5)
    unemployment = st.number_input("Unemployment Rate (%)", 0.0, 10.0, 2.0)
    yield_5y = st.number_input("5Y Bond Yield (%)", 0.0, 5.0, 2.0)

    submit = st.form_submit_button("Predict Resale Price 💰")

# When user submits the form
if submit:
    # Organize inputs
    input_df = pd.DataFrame([[
        floor_area, storey, latitude, longitude,
        mrt, cbd_dist, lease_year, town,
        flat_model, flat_type,
        gdp, gdp_per_capita, gni_per_capita, population,
        household, hdb_dwellings, inflation,
        interest_rate, unemployment, yield_5y
    ]], columns=[
        'floor_area_sqm', 'storey_median', 'latitude', 'longitude',
        'closest_mrt', 'cbd_dist', 'lease_commence_date', 'town',
        'flat_model', 'flat_type',
        'GDP', 'GDP per Capita', 'GNI per Capita', 'Population',
        'Resident Household', 'HDB Dwellings', 'Inflation Rate',
        'Interest Rate', 'Unemployment Rate', 'Yield_5y_interest'
    ])

    # Encode categorical
    categorical_cols = ['town', 'flat_model', 'flat_type', 'closest_mrt']
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"💵 Estimated Resale Price: SGD {prediction:,.2f}")
