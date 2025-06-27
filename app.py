import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ---------------------- Load & Train Model ----------------------
@st.cache_data
def load_and_train():
    df = pd.read_csv("winequality-red.csv")

    X = df.drop('quality', axis=1)
    y = df['quality']  # Multi-class target: quality scores from 3 to 8

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBClassifier(objective='multi:softprob', num_class=6, eval_metric='mlogloss')
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns, y

# Load model and data
model, scaler, feature_names, y_all = load_and_train()

# ---------------------- Streamlit UI ----------------------
st.title("🍷 Advanced Wine Quality Predictor")
st.write("Predict the quality of wine (3–8) using an advanced machine learning model.")

st.sidebar.header("Enter Wine Features")

# Get user input using sliders
user_input = {}
for feature in feature_names:
    min_val = float(y_all.min())
    max_val = float(y_all.max())
    default = 5.0
    user_input[feature] = st.sidebar.slider(feature.replace('_', ' ').capitalize(), 0.0, 20.0, default, 0.1)

# Predict on form submission
if st.button("Predict Quality"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    pred_probs = model.predict_proba(input_scaled)[0]
    pred_class = model.predict(input_scaled)[0]

    st.success(f"✅ Predicted Wine Quality: **{int(pred_class)}**")
    
    # Show confidence for each class
    st.subheader("📊 Prediction Confidence (All Classes)")
    conf_df = pd.DataFrame(pred_probs, index=model.classes_, columns=['Probability']).sort_index()
    st.bar_chart(conf_df)

    st.subheader("🔍 Input Features")
    st.write(input_df)

# ---------------------- Feature Importance ----------------------
st.subheader("📈 Feature Importance")

# Plot feature importance
importances = model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.barh(feature_names[sorted_idx], importances[sorted_idx], color='teal')
plt.xlabel("Importance Score")
plt.title("Feature Importance (XGBoost)")
st.pyplot(plt)
