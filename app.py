import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")

model = load_model()

st.title("❤️ Heart Disease Prediction")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 female, 1 male)", [0,1])
cp = st.number_input("Chest pain type", 0, 3, 1)
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol] + [0]*(model.n_features_in_-5)])
    pred = model.predict(input_data)

    if pred[0] == 1:
        st.error("⚠️ Heart Disease Risk")
    else:
        st.success("✅ No Heart Disease Risk")
