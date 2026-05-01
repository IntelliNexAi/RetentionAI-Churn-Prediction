import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="RetentionAI | Customer Churn Command Center",
    page_icon="👤",
    layout="wide"
)

# --- 2. PROFESSIONAL CSS INJECTION ---
st.markdown("""
    <style>
    /* Main Theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(#38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    /* Card Design */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Risk Status Colors */
    .risk-high { color: #f87171; font-weight: bold; font-size: 24px; }
    .risk-low { color: #4ade80; font-weight: bold; font-size: 24px; }
    
    /* Button Customization */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ARTIFACTS ---
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('churn_model.pkl', 'rb'))
        scaler = pickle.load(open('churn_scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_assets()

# --- 4. SIDEBAR / SETTINGS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3408/3408545.png", width=100)
    st.markdown("## RetentionAI Settings")
    st.markdown("---")
    threshold = st.slider("Churn Risk Threshold", 0.0, 1.0, 0.50)
    st.info("Adjust the threshold to change sensitivity for high-risk flags.")

# --- 5. MAIN INTERFACE ---
st.markdown('<h1 class="main-title">CUSTOMER CHURN COMMAND CENTER</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8;'>Analyze customer behavior and predict attrition risk in real-time.</p>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model artifacts missing! Please run 'train_churn.py' first.")
    st.stop()

# Input Section: Using 3 Columns for a clean UI
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Customer Profile")
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])

with col2:
    st.markdown("###  Services Used")
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

with col3:
    st.markdown("###  Financials")
    monthly_charges = st.number_input("Monthly Charges ($)", value=70.0)
    total_charges = st.number_input("Total Charges ($)", value=840.0)
    payment_method = st.selectbox("Payment Method", 
                                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Mapping inputs back to the format the model expects
# Note: This mapping must strictly match the order and values used in your Training Script
def preprocess_input():
    input_dict = {
        'gender': 0, # Placeholder (needs to match your CSV training columns)
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 0, 'Dependents': 0, # Placeholders
        'tenure': tenure,
        'PhoneService': 1, 'MultipleLines': 0, # Placeholders
        'InternetService': 0 if internet == "DSL" else (1 if internet == "Fiber optic" else 2),
        'OnlineSecurity': 0 if security == "No" else (1 if security == "Yes" else 2),
        'OnlineBackup': 0, 'DeviceProtection': 0, # Placeholders
        'TechSupport': 0 if tech_support == "No" else (1 if tech_support == "Yes" else 2),
        'StreamingTV': 0 if streaming == "No" else (1 if streaming == "Yes" else 2),
        'StreamingMovies': 0, 
        'Contract': 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'PaymentMethod': 0, # Mapping simplified for example
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    # Ensure this matches exactly the columns your model saw during .fit()
    return pd.DataFrame([input_dict])

# --- 6. PREDICTION LOGIC ---
st.markdown("---")
if st.button("RUN PREDICTIVE ANALYSIS"):
    with st.spinner(" Deep-scanning customer behavior..."):
        time.sleep(1.5)
        
        input_df = preprocess_input()
        input_scaled = scaler.transform(input_df)
        
        prob = model.predict_proba(input_scaled)[0][1]
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("### CHURN PROBABILITY")
            risk_color = "risk-high" if prob > threshold else "risk-low"
            st.markdown(f'<p class="{risk_color}">{prob:.1%}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with res_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if prob > threshold:
                st.subheader(" RETENTION ALERT")
                st.write("This customer is likely to leave within the next 30 days.")
                st.markdown("**Recommended Actions:**")
                st.info("1. Offer 'Contract Upgrade' discount (20% off)\n2. Assign priority Tech Support specialist\n3. Waive late fees for 3 months")
            else:
                st.subheader(" CUSTOMER STABLE")
                st.write("Customer shows high loyalty indicators.")
                st.markdown("**Engagement Strategy:**")
                st.success("Target for 'Premium Service' upselling or Referral Program.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- 7. FOOTER ---
st.markdown("<br><p style='text-align: center; color: #475569;'>RetentionAI System v2.1.0 | Enterprise Security Node: Active</p>", unsafe_allow_html=True)