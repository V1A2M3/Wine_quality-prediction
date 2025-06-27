import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# -------------------- DATA & MODEL SETUP --------------------

@st.cache_data
def load_and_train():
    df = pd.read_csv(r"C:/Users/chitt/Downloads/winequality-red.csv")

    # Binary classification: quality >= 6 is good
    df['quality'] = df['quality'].apply(lambda x: 1 if x >= 5 else 0)

    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    voting = VotingClassifier(estimators=[('rf', rf), ('gb', gb)], voting='soft')
    voting.fit(X_train_scaled, y_train)

    return voting, scaler, X.columns

# Load model and scaler
model, scaler, feature_names = load_and_train()

# -------------------- STREAMLIT UI --------------------

st.title("🍷 Wine Quality Prediction App")
st.write("Enter the physicochemical properties of wine to predict its quality (Good or Not Good).")

# User input form
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}:", min_value=0.0, value=5.0)

# Predict button
if st.button("Predict Quality"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ This wine is predicted to be of **GOOD quality** (Confidence: {prob:.2%})")
    else:
        st.warning(f"❌ This wine is predicted to be of **NOT GOOD quality** (Confidence: {prob:.2%})")

    st.subheader("🔍 Input Summary:")
    st.write(input_df)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & scikit-learn")
