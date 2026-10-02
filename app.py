import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE SETUP
# -----------------------------

st.set_page_config(
    page_title="AI Diabetes Research",
    page_icon="🧬",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return joblib.load("diabetes_random_forest_model.pkl")


model = load_model()

# -----------------------------
# TITLE
# -----------------------------

st.title("AI Diabetes Research")

st.subheader(
    "An independent machine-learning investigation "
    "into diabetes classification"
)

st.warning(
    "Research and educational demonstration only. "
    "This application is NOT a medical diagnostic tool "
    "and should not be used to make healthcare decisions."
)

# -----------------------------
# PROJECT INTRODUCTION
# -----------------------------

st.markdown(
    """
This project investigates how machine-learning models can
classify diabetes outcomes using a historical public dataset.

Three machine-learning approaches were investigated:

- Logistic Regression
- Random Forest
- Support Vector Machine

The models were evaluated using accuracy, precision, recall,
F1 score, ROC-AUC and cross-validation.
"""
)

st.divider()

# -----------------------------
# EXAMPLE INPUTS
# -----------------------------

st.header("Interactive Model Demonstration")

st.info(
    "Use example values only. Do not enter real personal "
    "medical information."
)

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# -----------------------------
# CREATE INPUT DATA
# -----------------------------

input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [diabetes_pedigree],
    "Age": [age]
})

# -----------------------------
# RUN MODEL
# -----------------------------

if st.button("Run Research Model", type="primary"):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_no = probabilities[0]
    probability_yes = probabilities[1]

    st.divider()

    st.subheader("Model Output")

    if prediction == 1:

        st.error(
            "The model classified this example as "
            "'Diabetes'."
        )

    else:

        st.success(
            "The model classified this example as "
            "'No diabetes'."
        )

    st.caption(
        "This is a machine-learning output based on the "
        "historical research dataset. It is not a diagnosis."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Model output: No diabetes",
            f"{probability_no:.1%}"
        )

    with col2:

        st.metric(
            "Model output: Diabetes",
            f"{probability_yes:.1%}"
        )

# -----------------------------
# RESEARCH RESULTS
# -----------------------------

st.divider()

st.header("Model Performance")

try:

    results = pd.read_csv("model_comparison.csv")

    st.dataframe(
        results,
        use_container_width=True
    )

except FileNotFoundError:

    st.info(
        "Model comparison results are not currently available."
    )

# -----------------------------
# CROSS VALIDATION
# -----------------------------

st.header("Cross-Validation Results")

try:

    cv_results = pd.read_csv(
        "cross_validation_results.csv"
    )

    st.dataframe(
        cv_results,
        use_container_width=True
    )

except FileNotFoundError:

    st.info(
        "Cross-validation results are not currently available."
    )

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

st.header("Feature Importance")

try:

    feature_importance = pd.read_csv(
        "feature_importance.csv"
    )

    st.dataframe(
        feature_importance,
        use_container_width=True
    )

except FileNotFoundError:

    st.info(
        "Feature importance results are not currently available."
    )

# -----------------------------
# METHODOLOGY
# -----------------------------

st.divider()

st.header("Research Methodology")

st.markdown(
    """
### Dataset

The project uses a historical public diabetes dataset
containing health measurements and an outcome variable.

### Machine-learning models

The investigation compares:

- Logistic Regression
- Random Forest
- Support Vector Machine

### Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrices
- Cross-validation

Feature importance is also investigated for the Random
Forest model.
"""
)

# -----------------------------
# LIMITATIONS
# -----------------------------

st.header("Limitations")

st.markdown(
    """
This project has important limitations.

- The dataset represents a specific historical population.
- The dataset is relatively small.
- Model performance on this dataset does not establish
  clinical validity.
- Machine-learning models can produce false positives
  and false negatives.
- Feature importance does not prove causation.
- The model has not been clinically validated.
- The application must not be used for medical decisions.
- The displayed outputs represent model estimates from
  this research dataset rather than real-world medical
  probabilities.
"""
)

# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "Independent educational machine-learning research project"
)
