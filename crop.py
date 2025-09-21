import streamlit as st
import pickle
import numpy as np

# Load Crop Model
with open("Crop_recommendation.pkl", "rb") as f:
    crop_model = pickle.load(f)

# Crop Labels
crop_labels = {
    0: 'Apple', 1: 'Banana', 2: 'Blackgram', 3: 'Chickpea', 4: 'Coconut',
    5: 'Coffee', 6: 'Cotton', 7: 'Grapes', 8: 'Jute', 9: 'Kidneybeans',
    10: 'Lentil', 11: 'Maize', 12: 'Mango', 13: 'Mothbeans', 14: 'Mungbean',
    15: 'Muskmelon', 16: 'Orange', 17: 'Papaya', 18: 'PigeonPeas',
    19: 'Pomegranate', 20: 'Rice', 21: 'Watermelon'
}

# Prediction Function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = crop_model.predict(features)[0]
    return crop_labels.get(prediction, "Unknown Crop")

# Streamlit App UI
st.set_page_config(page_title="🌱 Crop Recommendation", layout="wide")
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

st.title("🌾 AI-Powered Crop Recommendation System")
st.markdown("### 🌍 Optimize Your Farming Decisions with Smart AI Insights")

st.subheader("🔬 Soil & Weather Inputs")
N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

if st.button("🔍 Predict Crop"):
    with st.spinner("🔄 Processing your inputs..."):
        recommended_crop = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        st.success(f"🌱 **Recommended Crop:** {recommended_crop}")

st.markdown("---")
st.markdown("### 🚀 Powered by AI & Machine Learning for Smarter Farming Decisions")