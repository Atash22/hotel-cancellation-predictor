import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# Load model and preprocessor
model = keras.models.load_model('model_final.keras')
preprocessor = joblib.load('preprocessor_final.pkl')

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Fill in the booking details to predict if it will be cancelled.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input("Lead Time (days)", 0, 700, 30)
    adults = st.number_input("Adults", 1, 10, 2)
    children = st.number_input("Children", 0, 10, 0)
    babies = st.number_input("Babies", 0, 5, 0)
    stays_weekend = st.number_input("Weekend Nights", 0, 10, 1)
    stays_week = st.number_input("Week Nights", 0, 20, 2)
    adr = st.number_input("Avg Daily Rate (€)", 0, 1000, 100)

with col2:
    hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
    meal = st.selectbox("Meal Plan", ["BB", "FB", "HB", "SC", "Undefined"])
    market_segment = st.selectbox("Market Segment", [
                                  "Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups", "Complementary", "Aviation"])
    deposit_type = st.selectbox(
        "Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    customer_type = st.selectbox(
        "Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])
    reserved_room = st.selectbox("Reserved Room Type", [
                                 "A", "B", "C", "D", "E", "F", "G", "H", "L"])
    distribution_channel = st.selectbox(
        "Distribution Channel", ["Direct", "Corporate", "TA/TO", "GDS", "Undefined"])

# Predict button
if st.button("Predict"):
    input_dict = {
        "lead_time": float(lead_time),
        "arrival_date_week_number": 27.0,
        "arrival_date_day_of_month": 15.0,
        "stays_in_weekend_nights": float(stays_weekend),
        "stays_in_week_nights": float(stays_week),
        "adults": float(adults),
        "children": float(children),
        "babies": float(babies),
        "is_repeated_guest": 0.0,
        "previous_cancellations": 0.0,
        "previous_bookings_not_canceled": 0.0,
        "required_car_parking_spaces": 0.0,
        "total_of_special_requests": 0.0,
        "adr": float(adr),
        "hotel": hotel,
        "arrival_date_month": 7,
        "meal": meal,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "reserved_room_type": reserved_room,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
    }

    input_df = pd.DataFrame([input_dict])
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)[0][0]

    st.divider()
    if prediction > 0.5:
        st.error(f"⚠️ Likely to CANCEL — {prediction:.1%} probability")
    else:
        st.success(f"✅ Likely to KEEP booking — {1-prediction:.1%} confidence")
