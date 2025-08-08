import streamlit as st
import pandas as pd
import joblib
from datetime import datetime


model = joblib.load("Logistic.pkl")
scaler = joblib.load("Scaler_heart.pkl")
expected_columns = joblib.load("Columns.pkl")


st.title("💓 Heart Disease Risk Prediction")
st.markdown("Predict the risk of heart disease based on patient health information.")

def get_user_input():
    st.sidebar.header("📝 Enter Patient Details")
    age = st.sidebar.slider("Age", 18, 100, 40)
    Sex = st.sidebar.selectbox("Sex", ["M", "F"])
    chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    Cholestrol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120mg/dL", [0, 1])
    resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    Exercise_Induced = st.sidebar.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    ST_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

    return {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": Cholestrol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + Sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + Exercise_Induced: 1,
        "ST_Slope_" + ST_slope: 1
    }


raw_input = get_user_input()
input_df = pd.DataFrame([raw_input])


if st.sidebar.button("🔍 Predict"):
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    
    input_df = input_df[expected_columns]


    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    
    st.subheader("📊 Prediction Result")
    st.write(f"**Risk Probability:** {probability * 100:.2f}%")

    if prediction == 1:
        st.error("🚨 High Risk of Heart Disease")
        st.markdown("### ⚠️ Lifestyle changes & consultation recommended.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.markdown("### 👍 Keep maintaining a healthy lifestyle!")


    result_df = input_df.copy()
    result_df["Prediction"] = ["High Risk" if prediction == 1 else "Low Risk"]
    result_df["Probability"] = probability
    result_df["Timestamp"] = datetime.now()

    csv = result_df.to_csv(index=False)
    st.download_button("📥 Download Result as CSV", data=csv, file_name="heart_prediction.csv", mime="text/csv")
