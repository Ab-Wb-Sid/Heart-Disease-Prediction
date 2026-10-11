from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURES = ["age", "cp", "thalach", "oldpeak"]
CP_LABELS = {
    1: "1 - Typical angina",
    2: "2 - Atypical angina",
    3: "3 - Non-anginal pain",
    4: "4 - Asymptomatic",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "feature_ranges": {
            "age": {"min": 29.0, "max": 77.0},
            "cp": {"min": 1.0, "max": 4.0},
            "thalach": {"min": 71.0, "max": 202.0},
            "oldpeak": {"min": 0.0, "max": 6.2},
        },
        "cp_values": [1.0, 2.0, 3.0, 4.0],
    }


def get_range(metadata, feature):
    values = metadata["feature_ranges"][feature]
    return float(values["min"]), float(values["max"])


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=":heart:",
    layout="centered",
)

st.title("Heart Disease Prediction")
st.caption("AIC354 educational machine-learning project")
st.warning(
    "This app is for learning and assignment demonstration only. "
    "It is not a medical diagnosis tool and should not be used for health decisions."
)

metadata = load_metadata()

try:
    model = load_model()
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.info("Run the notebook first to create models/heart_disease_pipeline.joblib.")
    st.exception(exc)
    st.stop()

age_min, age_max = get_range(metadata, "age")
thalach_min, thalach_max = get_range(metadata, "thalach")
oldpeak_min, oldpeak_max = get_range(metadata, "oldpeak")
allowed_cp = [int(value) for value in metadata.get("cp_values", [1, 2, 3, 4])]

st.subheader("Patient Inputs")

age = st.number_input(
    "Age",
    min_value=int(age_min),
    max_value=int(age_max),
    value=55,
    step=1,
    help=f"Range observed in the dataset: {age_min:.0f} to {age_max:.0f}",
)

cp_label = st.selectbox(
    "Chest pain type (cp)",
    options=[CP_LABELS[value] for value in allowed_cp],
    index=allowed_cp.index(4) if 4 in allowed_cp else 0,
    help="UCI coding: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic.",
)
cp = int(cp_label.split(" - ")[0])

thalach = st.number_input(
    "Maximum heart rate achieved (thalach)",
    min_value=int(thalach_min),
    max_value=int(thalach_max),
    value=150,
    step=1,
    help=f"Range observed in the dataset: {thalach_min:.0f} to {thalach_max:.0f}",
)

oldpeak = st.number_input(
    "ST depression induced by exercise (oldpeak)",
    min_value=float(oldpeak_min),
    max_value=float(oldpeak_max),
    value=1.5,
    step=0.1,
    format="%.1f",
    help=f"Range observed in the dataset: {oldpeak_min:.1f} to {oldpeak_max:.1f}",
)

input_df = pd.DataFrame(
    [{"age": age, "cp": cp, "thalach": thalach, "oldpeak": oldpeak}],
    columns=FEATURES,
)

if st.button("Predict", type="primary"):
    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
    except Exception as exc:
        st.error("Prediction failed. Please check the model artifact and input values.")
        st.exception(exc)
        st.stop()

    if prediction == 1:
        st.error("Prediction: Disease present")
    else:
        st.success("Prediction: No disease")

    st.write(
        {
            "Probability of no disease": round(float(probabilities[0]), 3),
            "Probability of disease present": round(float(probabilities[1]), 3),
        }
    )

with st.expander("Model and dataset notes"):
    st.write(
        "The model uses exactly four inputs: age, cp, thalach, and oldpeak. "
        "The target was created from the UCI Cleveland `num` column: 0 means no disease, "
        "and any value greater than 0 means disease present."
    )
    if "metrics" in metadata:
        st.write("Held-out test metrics from the notebook:")
        st.json(metadata["metrics"])
