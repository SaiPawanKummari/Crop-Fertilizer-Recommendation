import streamlit as st
import pickle
import numpy as np

# Load Models
with open("Crop_recommendation.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("fertilizer.pkl", "rb") as f:
    fertilizer_model = pickle.load(f)

# Crop and Fertilizer Labels
crop_labels = {
    0: 'Apple', 1: 'Banana', 2: 'Blackgram', 3: 'Chickpea', 4: 'Coconut',
    5: 'Coffee', 6: 'Cotton', 7: 'Grapes', 8: 'Jute', 9: 'Kidneybeans',
    10: 'Lentil', 11: 'Maize', 12: 'Mango', 13: 'Mothbeans', 14: 'Mungbean',
    15: 'Muskmelon', 16: 'Orange', 17: 'Papaya', 18: 'PigeonPeas',
    19: 'Pomegranate', 20: 'Rice', 21: 'Watermelon'
}

fertilizer_labels = {
    1: "Urea", 2: "DAP", 3: "14-35-14", 4:'28-28',5:'17-17-17',6:'20-20',7:'10-26-26'
}

# Prediction Functions
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = crop_model.predict(features)[0]
    return crop_labels.get(prediction, "Unknown Crop")

def predict_fertilizer(N, P, K, soil_type, crop_type, moisture, temperature, growth_stage):
    features = np.array([[N, P, K, soil_type, crop_type, moisture, temperature, growth_stage]])
    prediction = fertilizer_model.predict(features)[0]
    return fertilizer_labels.get(prediction, "Unknown Fertilizer")

# Streamlit App UI
st.set_page_config(page_title="🌱 Crop & Fertilizer Recommendation", layout="wide")
st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50 !important;
            color: white !important;
            font-size: 18px !important;
            padding: 10px 24px !important;
            border-radius: 8px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 AI-Powered Crop & Fertilizer Recommendation System")
st.markdown("### 🌍 Optimize Your Farming Decisions with Smart AI Insights")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🔬 Soil & Weather Inputs")
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

with col2:
    st.subheader("🌱 Crop & Fertilizer Inputs")
    soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Clayey", "Silty"], index=1)
    crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Cotton", "Barley"], index=0)
    moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=50.0)
    growth_stage = st.number_input("Crop Growth Stage", min_value=0.0, max_value=10.0, value=5.0)

# Encode categorical inputs
soil_mapping = {"Sandy": 0, "Loamy": 1, "Clayey": 2, "Silty": 3}
crop_mapping = {"Rice": 0, "Wheat": 1, "Maize": 2, "Cotton": 3, "Barley": 4}
soil_encoded = soil_mapping[soil_type]
crop_encoded = crop_mapping[crop_type]

if st.button("🔍 Predict Crop & Fertilizer"):
    with st.spinner("🔄 Processing your inputs..."):
        recommended_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        recommended_fertilizer = predict_fertilizer(N, P, K, soil_encoded, crop_encoded, moisture, temperature, growth_stage)
        
        st.success(f"🌱 **Recommended Crop:** {recommended_crop}")
        st.success(f"💊 **Recommended Fertilizer:** {recommended_fertilizer}")

st.markdown("---")
st.markdown("### 🚀 Powered by AI & Machine Learning for Smarter Farming Decisions")
