import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. LOAD DATA ---
print(" Loading Telco Dataset...")
df = pd.read_csv('Telco_customer_chun.csv')

# --- 2. DATA CLEANING (Senior Engineer Level) ---
# Drop ID - it has zero predictive power
df.drop('customerID', axis=1, inplace=True)

# TotalCharges fix: It looks like numbers, but often contains " " (empty strings)
# errors='coerce' turns the spaces into NaN (Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# --- 3. ENCODING ---
# Convert 'Churn' target to 0 and 1
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Encode all 'Yes/No' and Category columns automatically
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# --- 4. DATA SPLIT ---
X = df.drop('Churn', axis=1)
y = df['Churn']

# Using 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. FEATURE SCALING ---
# Crucial for models to treat 'MonthlyCharges' ($100) and 'tenure' (2 months) fairly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. HIGH-PERFORMANCE TRAINING (XGBoost) ---
print(" Training Professional Churn Model (XGBoost)...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    scale_pos_weight=3, # Fixes Imbalance: churners are fewer than non-churners
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train)

# --- 7. EVALUATION ---
preds = model.predict(X_test_scaled)
print("\n" + "="*30)
print(" CHURN PERFORMANCE REPORT")
print("="*30)
print(f"Model Accuracy: {accuracy_score(y_test, preds):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, preds))

# --- 8. SAVE ARTIFACTS ---
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('churn_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n Success! Saved churn_model.pkl and churn_scaler.pkl")
