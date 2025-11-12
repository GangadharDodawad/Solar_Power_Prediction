import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
# Import the Keras load_model function
from tensorflow.keras.models import load_model

# --- Model & Scaler Paths ---
model_dir = 'models'
model_path = os.path.join(model_dir, 'spfnet_model.h5') 
scaler_X_path = os.path.join(model_dir, 'sc_X.pkl')
scaler_y_path = os.path.join(model_dir, 'sc_y.pkl')

@st.cache_resource
def load_model_and_scalers():
    # Ensure the directory and files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please re-save your model as 'spfnet_model.h5' in the 'models' directory.")
        st.stop()
    if not os.path.exists(scaler_X_path):
        st.error(f"Scaler X file not found at {scaler_X_path}.")
        st.stop()
    if not os.path.exists(scaler_y_path):
        st.error(f"Scaler Y file not found at {scaler_y_path}.")
        st.stop()

    try:
        model = load_model(model_path, compile=False)
        
        with open(scaler_X_path, 'rb') as f:
            scaler_X = pickle.load(f)
        with open(scaler_y_path, 'rb') as f:
            scaler_y = pickle.load(f)
            
        return model, scaler_X, scaler_y
        
    except ImportError as ie:
        st.error(f"TensorFlow Import Error: {ie}. This is often caused by a mismatch in Python versions (TensorFlow may not support Python 3.12 fully) or missing C++ dependencies.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scalers: {e}")
        st.stop()

# --- Load Models ---
spfnet, sc_X, sc_y = load_model_and_scalers()

# --- Column Definitions ---
column_names = [
    'temperature_2_m_above_gnd',
    'relative_humidity_2_m_above_gnd',
    'mean_sea_level_pressure_MSL',
    'total_precipitation_sfc',
    'snowfall_amount_sfc',
    'total_cloud_cover_sfc',
    'high_cloud_cover_high_cld_lay',
    'medium_cloud_cover_mid_cld_lay',
    'low_cloud_cover_low_cld_lay',
    'shortwave_radiation_backwards_sfc',
    'wind_speed_10_m_above_gnd',
    'wind_direction_10_m_above_gnd',
    'wind_speed_80_m_above_gnd',
    'wind_direction_80_m_above_gnd',
    'wind_speed_900_mb',
    'wind_direction_900_mb',
    'wind_gust_10_m_above_gnd',
    'angle_of_incidence',
    'zenith',
    'azimuth'
]

feature_ranges = {
    'temperature_2_m_above_gnd': {'min': -5.35, 'max': 34.90, 'mean': 15.07},
    'relative_humidity_2_m_above_gnd': {'min': 7.00, 'max': 100.00, 'mean': 51.36},
    'mean_sea_level_pressure_MSL': {'min': 997.50, 'max': 1046.80, 'mean': 1019.34},
    'total_precipitation_sfc': {'min': 0.00, 'max': 3.20, 'mean': 0.03},
    'snowfall_amount_sfc': {'min': 0.00, 'max': 1.68, 'mean': 0.00},
    'total_cloud_cover_sfc': {'min': 0.00, 'max': 100.00, 'mean': 34.06},
    'high_cloud_cover_high_cld_lay': {'min': 0.00, 'max': 100.00, 'mean': 14.46},
    'medium_cloud_cover_mid_cld_lay': {'min': 0.00, 'max': 100.00, 'mean': 20.02},
    'low_cloud_cover_low_cld_lay': {'min': 0.00, 'max': 100.00, 'mean': 21.37},
    'shortwave_radiation_backwards_sfc': {'min': 0.00, 'max': 952.30, 'mean': 387.76},
    'wind_speed_10_m_above_gnd': {'min': 0.00, 'max': 23.36, 'mean': 6.00},
    'wind_direction_10_m_above_gnd': {'min': 0.54, 'max': 360.00, 'mean': 195.08},
    'wind_speed_80_m_above_gnd': {'min': 0.00, 'max': 66.88, 'mean': 18.98},
    'wind_direction_80_m_above_gnd': {'min': 1.12, 'max': 360.00, 'mean': 191.17},
    'wind_speed_900_mb': {'min': 0.00, 'max': 61.11, 'mean': 16.36},
    'wind_direction_900_mb': {'min': 1.12, 'max': 360.00, 'mean': 192.45},
    'wind_gust_10_m_above_gnd': {'min': 0.72, 'max': 84.96, 'mean': 20.58},
    'angle_of_incidence': {'min': 3.76, 'max': 121.64, 'mean': 50.84},
    'zenith': {'min': 17.73, 'max': 128.42, 'mean': 59.98},
    'azimuth': {'min': 54.38, 'max': 289.05, 'mean': 169.17}
}

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Solar Forecast", page_icon="☀️")

# --- Custom CSS for Modern Solar Theme ---
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Page Background - Solar Gradient */
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #ffa726 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Make main content area centered and flexible */
[data-testid="stAppViewContainer"] > .main .block-container {
   background: none;
   padding-top: 2rem;
   padding-bottom: 8rem;
   max-width: 1200px;
   margin: 0 auto;
   flex: 1;
}

/* Sidebar Styling - Dark Glass */
[data-testid="stSidebar"] > div:first-child {
    background: rgba(30, 30, 45, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 2px solid rgba(255, 255, 255, 0.1);
    box-shadow: 4px 0 30px rgba(0, 0, 0, 0.3);
}

/* Sidebar Header */
[data-testid="stSidebar"] h2 {
    color: #ffa726;
    text-shadow: 0 0 20px rgba(255, 167, 38, 0.5);
    font-weight: 700;
    font-size: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    letter-spacing: 1px;
}

/* Sidebar Widget Labels */
[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 600;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    font-size: 0.9rem;
}

/* Sidebar Sliders and Number Inputs */
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #667eea, #ffa726);
}

[data-testid="stSidebar"] input {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 2px solid rgba(255, 167, 38, 0.5) !important;
    color: #1e1e2d !important;
    border-radius: 8px;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.5rem !important;
}

[data-testid="stSidebar"] input:focus {
    border-color: #ffa726 !important;
    box-shadow: 0 0 10px rgba(255, 167, 38, 0.5) !important;
    background: rgba(255, 255, 255, 1) !important;
}

/* Number input buttons */
[data-testid="stSidebar"] button[kind="stepperButton"] {
    background: rgba(255, 167, 38, 0.2) !important;
    color: white !important;
}

[data-testid="stSidebar"] button[kind="stepperButton"]:hover {
    background: rgba(255, 167, 38, 0.4) !important;
}

/* Main Title */
h1 {
    color: white;
    text-shadow: 0 4px 20px rgba(0,0,0,0.4), 0 0 40px rgba(255,255,255,0.3);
    text-align: center;
    font-weight: 700;
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    letter-spacing: 2px;
    animation: titlePulse 3s ease-in-out infinite;
}

@keyframes titlePulse {
    0%, 100% { text-shadow: 0 4px 20px rgba(0,0,0,0.4), 0 0 40px rgba(255,255,255,0.3); }
    50% { text-shadow: 0 4px 20px rgba(0,0,0,0.4), 0 0 60px rgba(255,255,255,0.5); }
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
    margin: 1.5rem auto;
    max-width: 800px;
}

/* Subtitle */
.stMarkdown h3 {
    text-align: center;
    color: #3CB371;
    font-weight: 600;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    font-size: 1.3rem;
}

/* Center all markdown content */
.stMarkdown {
    text-align: center;
}

/* Glass Container for Prediction */
.glass-container {
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border-radius: 25px;
    border: 2px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 50px rgba(0, 0, 0, 0.3), inset 0 0 30px rgba(255, 255, 255, 0.1);
    padding: 3rem;
    margin: 2rem auto;
    color: white;
    text-align: center;
    position: relative;
    overflow: hidden;
    max-width: 600px;
}

.glass-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: rotate 10s linear infinite;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.glass-container h3 {
    position: relative;
    z-index: 1;
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: #ffa726;
    text-shadow: 0 0 20px rgba(255, 167, 38, 0.5);
}

.glass-container h1 {
    position: relative;
    z-index: 1;
    font-size: 4rem;
    margin: 1rem 0;
    background: linear-gradient(135deg, #ffffff, #ffa726, #f5576c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: none;
}

/* Custom Button Styling */
.stButton {
    display: flex;
    justify-content: center;
}

.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f5576c 100%);
    color: white;
    border-radius: 15px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    padding: 1rem 2rem;
    font-weight: 700;
    font-size: 1.2rem;
    transition: all 0.4s ease;
    width: 100%;
    max-width: 500px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.3), inset 0 0 20px rgba(255,255,255,0.1);
    letter-spacing: 1px;
    text-transform: uppercase;
}

.stButton button:hover {
    background: linear-gradient(135deg, #f5576c 0%, #764ba2 50%, #667eea 100%);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 0 30px rgba(255,255,255,0.2);
    transform: translateY(-3px) scale(1.02);
    border-color: rgba(255, 255, 255, 0.5);
}

.stButton button:active {
    transform: translateY(-1px) scale(1.01);
}

/* Success/Error Messages */
[data-testid="stSuccess"] {
    background: linear-gradient(135deg, rgba(40, 167, 69, 0.85), rgba(56, 193, 114, 0.85));
    backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    text-align: center;
    margin: 0 auto;
}

[data-testid="stError"] {
    background: linear-gradient(135deg, rgba(220, 53, 69, 0.85), rgba(255, 87, 108, 0.85));
    backdrop-filter: blur(10px);
    border-radius: 15px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    text-align: center;
    margin: 0 auto;
}

/* Footer - Fixed at bottom */
.footer-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 999;
    padding: 0;
    margin: 0;
}

.footer-text {
    text-align: center;
    color: #f0f8ff;
    font-weight: 500;
    text-shadow: 0 1px 5px rgba(0,0,0,0.3);
    background: rgba(0, 100, 150, 0.3);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    padding: 1.25rem;
    border-top: 1px solid rgba(255, 255, 255, 0.15);
    margin: 0;
    transition: all 0.3s ease;
}

.footer-text:hover {
    background: rgba(0, 100, 150, 0.4);
    box-shadow: 0 -4px 30px rgba(255, 255, 255, 0.1);
}

/* Adjust columns to center content */
[data-testid="column"] {
    display: flex;
    justify-content: center;
    align-items: center;
}

</style>
""", unsafe_allow_html=True)


st.title("☀️ Solar Power Generation Forecasting")
st.markdown("---")
st.markdown('<h3>Use the sidebar to input weather data and click Predict.</h3>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Input Features")

user_inputs = []
for col in column_names:
    stats = feature_ranges.get(col, {'min': 0.0, 'max': 100.0, 'mean': 50.0})

    if 'total_precipitation_sfc' in col or 'snowfall_amount_sfc' in col or 'cloud_cover' in col:
        value = st.sidebar.number_input(f"**{col.replace('_', ' ').title()}**",
                                          min_value=float(stats['min']),
                                          max_value=float(stats['max']),
                                          value=float(stats['mean']),
                                          step=0.01,
                                          format="%.2f")
    else:
        value = st.sidebar.slider(f"**{col.replace('_', ' ').title()}**",
                                    min_value=float(stats['min']),
                                    max_value=float(stats['max']),
                                    value=float(stats['mean']),
                                    step=0.01,
                                    format="%.2f")
    user_inputs.append(value)

input_data = np.array(user_inputs).reshape(1, -1)

# --- Main Page Button and Output ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 Predict Generated Power"):
        try:
            # Scale the input features
            scaled_input = sc_X.transform(input_data)
    
            # Make prediction
            scaled_prediction = spfnet.predict(scaled_input)
    
            # Inverse transform the prediction to get original scale
            predicted_power = sc_y.inverse_transform(scaled_prediction)[0][0]
    
            # Display result in the glass container
            st.markdown(
                f"""
                <div class="glass-container">
                    <h3>⚡ Predicted Solar Power Generation</h3>
                    <h1>{predicted_power:,.2f} KW</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
    
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Fixed Footer at bottom
st.markdown(
    """
    <div class="footer-container">
        <div class="footer-text">
            🌞 This app predicts solar power generation based on various weather and solar angle parameters using advanced neural networks.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)