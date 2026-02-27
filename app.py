import streamlit as st
import joblib
import numpy as np

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")

model = load_model()

# ---------------- UI ----------------
st.title("❤️ Heart Disease Prediction App")

st.subheader("Enter Patient Details")

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex (0 Female, 1 Male)", [0,1])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])

# Row 2
col4, col5, col6 = st.columns(3)

with col4:
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0,1])

# Row 3
col7, col8, col9 = st.columns(3)

with col7:
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0,1])

# Row 4
col10, col11, col12 = st.columns(3)

with col10:
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)

with col11:
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Major Vessels (ca)", [0,1,2,3])

with col12:
    thal = st.selectbox("Thal", [0,1,2,3])

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_data = np.array([[ 
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]])

    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
    else:
        prob = None

    st.subheader("Result")

    if prediction == 1:
        if prob is not None:
            st.error(f"⚠️ Heart Disease Risk — Probability: {prob:.2f}")
        else:
            st.error("⚠️ Heart Disease Risk")
    else:
        if prob is not None:
            st.success(f"✅ No Heart Disease — Confidence: {1-prob:.2f}")
        else:
            st.success("✅ No Heart Disease")
