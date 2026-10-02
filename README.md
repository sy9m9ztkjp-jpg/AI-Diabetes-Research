AI Diabetes Research

An Independent Machine Learning Investigation into Diabetes Classification

An educational machine-learning research project investigating how different classification algorithms identify diabetes outcomes using a historical public dataset.

The project compares Logistic Regression, Random Forest and Support Vector Machine models, using statistical evaluation, cross-validation and feature importance analysis.

Research Focus: Artificial Intelligence · Machine Learning · Healthcare Data Science · Classification

⸻

Live Application

Explore the interactive research demonstration:

Launch AI Diabetes Research

The application allows users to explore model outputs, compare algorithm performance and investigate the variables used in the research.

Disclaimer: This application is strictly for educational and research purposes. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

⸻

Research Question

How effectively can different machine-learning classification algorithms distinguish diabetes outcomes using variables contained within a historical public dataset?

The investigation explores how algorithm selection, evaluation metrics and validation techniques influence the interpretation of machine-learning performance.

⸻

Project Overview

Machine learning has applications in healthcare research, particularly in the analysis of patterns within medical datasets.

This project investigates three supervised learning algorithms to examine their ability to classify diabetes outcomes.

Rather than relying solely on accuracy, the investigation considers multiple performance metrics, including precision, recall, F1 score and ROC-AUC.

The project also explores feature importance and the limitations of applying machine-learning models to healthcare-related datasets.

⸻

Machine Learning Models

Algorithm	Description
Logistic Regression	Statistical classification algorithm used as a baseline model
Random Forest	Ensemble learning algorithm combining multiple decision trees
Support Vector Machine	Classification algorithm that identifies decision boundaries between classes

All three models were trained and evaluated using a consistent dataset split.

⸻

Dataset

The investigation uses the historical Pima Indians Diabetes dataset.

* 768 observations
* 8 input variables
* 1 binary outcome variable
* Supervised binary classification

Input Variables

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. Body Mass Index (BMI)
7. Diabetes Pedigree Function
8. Age

The target variable represents the recorded diabetes outcome.

Dataset source: Pima Indians Diabetes Dataset

The dataset represents a specific historical population and should not be assumed to represent all populations.

⸻

Research Methodology

1. Data Preparation

The dataset was separated into input variables and the target outcome.

A stratified train-test split was used:

* 80% training data
* 20% testing data

A fixed random state was used to support reproducibility.

2. Preprocessing

Missing values were handled using median imputation.

Feature standardisation was applied to Logistic Regression and Support Vector Machine models.

Preprocessing was incorporated into machine-learning pipelines to reduce data leakage during evaluation.

3. Model Training

The three algorithms were trained using the training dataset.

4. Performance Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion matrices

5. Cross-Validation

Five-fold stratified cross-validation was performed to investigate model performance across multiple data partitions.

6. Feature Importance

Random Forest feature importance was examined to investigate which variables contributed most strongly to the fitted model.

Feature importance indicates model reliance, not causation.

⸻

Evaluation Metrics

Metric	Purpose
Accuracy	Proportion of correct classifications
Precision	Proportion of positive predictions that were correct
Recall	Proportion of actual positive cases identified
F1 Score	Harmonic mean of precision and recall
ROC-AUC	Measures discrimination across classification thresholds

Multiple metrics are considered because accuracy alone can conceal important differences in classification performance.

⸻

Interactive Research Dashboard

The Streamlit application includes:

* Interactive model demonstration
* Model-generated classifications
* Model output estimates
* Performance comparison table
* Selectable performance metric charts
* Cross-validation results
* ROC-AUC comparison
* Random Forest feature importance visualisation
* Research methodology
* Limitations and ethical considerations

⸻

Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib
* Google Colab
* GitHub

⸻

Limitations

Several limitations must be considered when interpreting the results:

1. The dataset represents a specific historical population.
2. The dataset is relatively small.
3. Performance may not generalise to other populations.
4. Model performance does not establish clinical effectiveness.
5. False positives and false negatives are possible.
6. Feature importance does not establish causal relationships.
7. The models have not undergone clinical validation.
8. Model-generated estimates should not be interpreted as real-world medical probabilities.

The findings are specific to the dataset and experimental methodology used.

⸻

Ethical Considerations

Machine-learning systems applied to healthcare-related problems require careful evaluation.

This project is intended to demonstrate machine-learning methodology and statistical evaluation, not to provide medical advice or diagnosis.

No real personal health information should be entered into the interactive demonstration.

The research acknowledges the importance of dataset representativeness, model limitations, transparency and responsible interpretation.

⸻

Repository Structure

AI-Diabetes-Research/
│
├── app.py
├── requirements.txt
├── diabetes_random_forest_model.pkl
├── model_comparison.csv
├── cross_validation_results.csv
├── feature_importance.csv
└── README.md

⸻

Running the Project Locally

1. Clone the repository

git clone YOUR-GITHUB-REPOSITORY-LINK

2. Install dependencies

pip install -r requirements.txt

3. Launch the application

streamlit run app.py

⸻

Research Conclusion

This project demonstrates how different machine-learning algorithms can be investigated for binary classification using a historical healthcare-related dataset.

By comparing multiple algorithms, statistical metrics and cross-validation results, the investigation highlights the importance of evaluating machine-learning systems beyond a single accuracy score.

The findings are limited to the dataset and experimental design and do not demonstrate clinical effectiveness.

⸻

Author

Independent educational machine-learning research project.

Developed using Python, Scikit-learn, Google Colab, GitHub and Streamlit.

⸻

Educational Research Project | Artificial Intelligence | Healthcare Data Science

This project is not intended for medical diagnosis, treatment or clinical decision-making.
