import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Diabetes Research",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("diabetes_random_forest_model.pkl")


model = load_model()

# ============================================================
# TITLE
# ============================================================

st.title("AI Diabetes Research")

st.subheader(
    "An independent machine-learning investigation "
    "into diabetes classification"
)

st.warning(
    "Educational research demonstration only. "
    "This application is NOT a medical diagnostic tool "
    "and must not be used to make healthcare decisions."
)

st.markdown(
    """
This project investigates whether machine-learning algorithms
can classify diabetes outcomes using health measurements from
a historical public dataset.

Three classification algorithms were investigated:

- Logistic Regression
- Random Forest
- Support Vector Machine

The models were evaluated using multiple statistical metrics
and cross-validation.
"""
)

# ============================================================
# RESEARCH QUESTION
# ============================================================

st.header("Research Question")

st.markdown(
    """
**How effectively can machine-learning classification algorithms
distinguish diabetes outcomes using the variables contained
within the selected historical dataset?**
"""
)

st.divider()

# ============================================================
# INTERACTIVE MODEL
# ============================================================

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
        "This is a model-generated output based on the "
        "historical research dataset. It is not a diagnosis "
        "and is not a clinically validated probability."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Model estimate: No diabetes",
            f"{probability_no:.1%}"
        )

    with col2:

        st.metric(
            "Model estimate: Diabetes",
            f"{probability_yes:.1%}"
        )

# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.header("Model Performance Comparison")

try:

    results = pd.read_csv("model_comparison.csv")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    metric_options = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    available_metrics = [
        metric for metric in metric_options
        if metric in results.columns
    ]

    if available_metrics:

        selected_metric = st.selectbox(
            "Select a metric to visualise",
            available_metrics
        )

        chart_data = results.set_index("Model")[
            selected_metric
        ]

        st.bar_chart(chart_data)

except Exception as e:

    st.error(
        f"Could not load model comparison results: {e}"
    )

# ============================================================
# CROSS VALIDATION
# ============================================================

st.header("Cross-Validation")

try:

    cv_results = pd.read_csv(
        "cross_validation_results.csv"
    )

    st.dataframe(
        cv_results,
        use_container_width=True,
        hide_index=True
    )

    if "AUC" in cv_results.columns:

        auc_chart = cv_results.set_index("Model")["AUC"]

        st.subheader("Cross-Validation ROC-AUC")

        st.bar_chart(auc_chart)

except Exception as e:

    st.error(
        f"Could not load cross-validation results: {e}"
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("Random Forest Feature Importance")

try:

    feature_importance = pd.read_csv(
        "feature_importance.csv"
    )

    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=True
    )

    st.dataframe(
        feature_importance,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

except Exception as e:

    st.error(
        f"Could not load feature importance results: {e}"
    )

# ============================================================
# METHODOLOGY
# ============================================================

st.divider()

st.header("Research Methodology")

st.markdown(
    """
### 1. Dataset

The investigation uses a historical public diabetes dataset
containing eight input variables and a binary outcome variable.

### 2. Data preparation

The dataset was divided into training and testing data.
Missing values were handled using median imputation where
required.

### 3. Machine-learning models

Three classification algorithms were investigated:

**Logistic Regression**

A statistical classification method that models the
relationship between input variables and a binary outcome.

**Random Forest**

An ensemble method that combines many decision trees to
produce a classification.

**Support Vector Machine**

A classification algorithm that attempts to separate
classes using a decision boundary.

### 4. Evaluation

Performance was investigated using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrices
- Five-fold stratified cross-validation

### 5. Feature importance

Random Forest feature importance was examined to investigate
which input variables contributed most strongly to the model's
predictions.
"""
)

# ============================================================
# LIMITATIONS
# ============================================================

st.header("Limitations")

st.markdown(
    """
This project has several important limitations.

- The dataset represents a specific historical population.
- The dataset is relatively small.
- Results may not generalise to other populations.
- Model performance on this dataset does not establish
  clinical effectiveness.
- Machine-learning models can produce false positives and
  false negatives.
- Feature importance does not demonstrate causation.
- The model has not undergone clinical validation.
- The displayed model estimates should not be interpreted as
  real-world medical probabilities.
- The application is an educational research demonstration
  rather than a healthcare product.
"""
)

# ============================================================
# ETHICS
# ============================================================

st.header("Ethical Considerations")

st.markdown(
    """
Machine-learning systems used in healthcare can have
significant consequences if their limitations are ignored.

This project therefore does not claim to diagnose disease.
The application is intended to demonstrate machine-learning
methods and research evaluation rather than provide medical
advice.

Users should not enter real personal health information into
this demonstration.
"""
)

# ============================================================
# CONCLUSION
# ============================================================

st.header("Research Conclusion")

st.markdown(
    """
The investigation compares three machine-learning approaches
for classifying diabetes outcomes within the selected
historical dataset.

The results demonstrate how different algorithms can produce
different performance characteristics and why evaluating a
model using several metrics is more informative than relying
on accuracy alone.

The findings are specific to the dataset and experimental
methodology used in this project and should not be interpreted
as evidence of clinical effectiveness.
"""
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Independent educational machine-learning research project"
)
