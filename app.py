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

    # Extract features and labels
    X = df.drop('quality', axis=1)
    y_raw = df['quality']

    # Map wine quality to 0-based labels for XGBoost
    label_mapping = {val: idx for idx, val in enumerate(sorted(y_raw.unique()))}
    y = y_raw.map(label_mapping)

    # Save reverse mapping for display
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train model
    model = XGBClassifier(objective='multi:softprob',
                          num_class=len(label_mapping),
                          eval_metric='mlogloss',
                          use_label_encoder=False)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns, reverse_mapping



# Load model and data
model, scaler, feature_names, reverse_mapping = load_and_train()

# ---------------------- Streamlit UI ----------------------
st.title("🍷 Advanced Wine Quality Predictor")
st.write("Predict the quality of wine (3–8) using an advanced machine learning model.")

st.sidebar.header("Enter Wine Features")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.slider(
        feature.replace('_', ' ').capitalize(),
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.1
    )

# Predict on form submission
if st.button("Predict Quality"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    pred_probs = model.predict_proba(input_scaled)[0]
    pred_class = model.predict(input_scaled)[0]
    true_quality = reverse_mapping[pred_class]

    st.success(f"✅ Predicted Wine Quality: **{true_quality}**")
    
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
