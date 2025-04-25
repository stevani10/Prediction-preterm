# Auto-install requirements if missing (good for local dev)
import os
import sys
import subprocess

# Install requirements.txt before importing anything else
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
except Exception as e:
    print(f"Failed to install requirements: {e}")

# Now import the rest
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load Model
@st.cache_resource
def load_model():
    try:
        return joblib.load("preterm_risk_model.pkl")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# File Upload
st.title("Preterm Birth Prediction")
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.sidebar.success("Dataset uploaded!")
    st.write("### Data Preview")
    st.dataframe(df.head())

    df['Previous pregnancy other issue status'] = df['Previous pregnancy other issue status'].str.lower()

    categories = {
        "abortus|bo|keguguran": 1,
        "prematur|preterm|premature": 2,
        "meninggal|lahir mati|iufd|stillbirth|bayi mati": 3,
        "anemia|hbsag|hepatitis|saraf|asma|kek": 4,
        "partus|sungsang|cpd|plasenta|kpd|post term|posterm|sunsang|su": 5,
        "gemeli|gameli|susp. gemeli|twin": 6,
        "perdarahan|pendarahan|hpp": 7,
        "ketuban pecah dini|kpd": 8
    }

    def simplify_status(value):
        for pattern, category in categories.items():
            if pd.notna(value) and any(keyword in value for keyword in pattern.split('|')):
                return category
        return 9

    df['Simplified Pregnancy Issues'] = df['Previous pregnancy other issue status'].apply(simplify_status)

    def replace_neg_pos(value):
        if isinstance(value, str):
            value_lower = value.lower()
            if "negatif" in value_lower:
                return 0
            elif "positif" in value_lower:
                return 1
        return value

    df = df.applymap(replace_neg_pos)

    baby_category = {
        "lahir_hidup": 1,
        "lahir_mati": 0
    }

    def simplify_status_baby(value):
        for pattern, category in baby_category.items():
            if pd.notna(value) and any(keyword in value for keyword in pattern.split('|')):
                return category
        return "0"

    df['Status Baby'] = df['status_baby'].apply(simplify_status_baby)

    st.sidebar.header("Enter ID for Prediction")
    id_input = st.sidebar.text_input("Enter ID")

    if id_input and model:
        id_input = str(id_input).strip()
        df['ID'] = df['ID'].astype(str)
        selected_data = df[df['ID'] == id_input]

        if selected_data.empty:
            st.error(f"ID {id_input} not found in dataset.")
        else:
            st.success(f"Prediction for ID: {id_input}")
            latest_record = selected_data.sort_values(by='visit_date', ascending=False).iloc[0]

            required_features = [
                'Abortus', 'Partus',
                'occupation_siswa__mahasiswa', 'occupation_pns', 'occupation_karyawan_swasta',
                'occupation_wiraswasta__wirausaha', 'occupation_ibu_rumah_tangga', 'occupation_lainnya',
                'Previous pregnancy preeclampsia status','Previous pregnancy eclampsia status', 
                'Previous pregnancy convulsion status','previous_preg_issue_gestational_',
                'Previous pregnancy heavy bleeding status', 'Previous pregnancy macrosomia status',
                'Simplified Pregnancy Issues', 'HIV status of the mother based on a test', 
                'Hepatitis B status of the mother based on a test', 'Syphilis status of the mother based on a test',
                'body_height', 'body_weight', 'mid_upper_arm_circum', 'systolic_blood_pressure',
                'diastolic_blood_pressure', "Mother's age", 'body_temperature', 'pulse', 
                'hemoglobinometer_result', 'fasting_glucose_result', 'random_glucose_test','Status Baby'    
            ]

            try:
                X_latest = latest_record[required_features].to_frame().T.astype(float)
                prediction_proba = model.predict_proba(X_latest)
                preterm_risk = prediction_proba[0][1]

                if preterm_risk >= 0.5:
                    st.write(f"### Prediction: **Preterm Birth Risk: {preterm_risk:.2%}**")
                else:
                    st.write(f"### Prediction: **Term Birth Likely: {1 - preterm_risk:.2%}**")

                if preterm_risk >= 0.5:
                    st.write("#### Why this Prediction?")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_latest)
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, X_latest, show=False)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
