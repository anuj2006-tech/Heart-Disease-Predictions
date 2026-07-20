import streamlit as st
import pandas as pd
import joblib

# Load model files
try:
    model = joblib.load('KNN_Heart.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('columns.pkl')
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.title("Heart Disease Prediction App")
st.markdown("Provide the following details to predict the likelihood of heart disease:")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ['M', 'F'])
chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=50,
    max_value=250,
    value=120
)
cholesterol = st.number_input(
    "Cholesterol (mg/dl)",
    min_value=100,
    max_value=600,
    value=200
)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
max_hr = st.slider(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=220,
    value=150
)
exercise_angina = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)
oldpeak = st.slider(
    "Oldpeak (ST depression)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)
st_slope = st.selectbox("ST_Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    try:
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,   # Change to oldpeak if your model used lowercase

            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("High likelihood of heart disease. Please consult a doctor.")
        else:
            st.success("Low likelihood of heart disease.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")