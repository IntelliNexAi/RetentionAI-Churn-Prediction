RetentionAI: Customer Churn Command Center
RetentionAI is an end-to-end predictive analytics solution designed to combat customer attrition in the telecommunications industry. This project combines a high-performance XGBoost machine learning model with a professional Streamlit dashboard to provide real-time risk assessment and automated retention strategies.

📋 Table of Contents
Problem Statement

Technical Architecture

Installation & Setup

How to Run

Model Training Logic

Dashboard Features

Future Roadmap

🎯 Problem Statement
In the telecom industry, acquiring a new customer is 5x more expensive than retaining an existing one. This system allows businesses to:

Identify high-risk customers with a probability score.

Understand the "why" behind potential churn (Tenure, Contract type, Charges).

Take immediate action through AI-generated retention recommendations.

🛠 Technical Architecture
The project is built using a decoupled architecture, separating the Training Environment from the Inference Environment.

Language: Python 3.9+

Modeling: XGBoost (Gradient Boosting)

Preprocessing: Scikit-Learn (StandardScaler, LabelEncoder)

UI Framework: Streamlit (Custom CSS-injected)

Data Handling: Pandas, NumPy

Serialization: Pickle

⚙️ Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/YOUR_USERNAME/RetentionAI-Churn-Prediction.git
cd RetentionAI-Churn-Prediction
2. Create a Virtual Environment (Recommended)
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
Bash
pip install pandas numpy scikit-learn xgboost streamlit
🚀 How to Run
Step 1: Train the Model
Before launching the dashboard, you must generate the model artifacts. Ensure Telco_customer_chun.csv is in the root directory.

Bash
python train_churn.py
This will perform data cleaning, handle class imbalance, and save churn_model.pkl and churn_scaler.pkl.

Step 2: Launch the Dashboard
Bash
streamlit run app.py
The dashboard will automatically load the saved model and scaler.

🧠 Model Training Logic (Senior Engineer Level)
This project implements several advanced data science techniques to ensure accuracy:

Handling Missing Values: TotalCharges is converted to numeric, and empty strings are imputed with the mean.

Class Imbalance: Since churners are usually a minority, we use scale_pos_weight=3 in XGBoost to penalize errors on churn cases more heavily.

Feature Scaling: We use StandardScaler to normalize financial figures (Monthly Charges) and time-based figures (Tenure) to the same scale.

Stratified Splitting: Ensuring the ratio of churn vs. non-churn is the same in both Training and Testing sets.

🖥️ Dashboard Features
1. Real-Time Risk Analysis
Input customer data (Contract, Tenure, Internet Service) and click "RUN PREDICTIVE ANALYSIS". The system uses a loading spinner to simulate a "deep scan" of customer behavior.

2. Risk Threshold Control
Use the sidebar slider to adjust the sensitivity of the model.

Lower Threshold: Catch more potential churners (High Sensitivity).

Higher Threshold: Focus only on the most certain cases (High Precision).

3. Actionable Retention Strategies
Based on the risk score, the app provides specific business advice:

High Risk: Offers discounts, priority tech support, or fee waivers.

Low Risk: Suggests upselling premium services or referral programs.

📁 Project Structure
Plaintext
├── Telco_customer_chun.csv   # Raw Dataset
├── train_churn.py            # Training & Evaluation Script
├── app.py                    # Streamlit Dashboard (Inference)
├── churn_model.pkl           # Saved XGBoost Model (Generated)
├── churn_scaler.pkl          # Saved Scaler Object (Generated)
├── README.md                 # Project Documentation
└── requirements.txt          # Python Dependencies
