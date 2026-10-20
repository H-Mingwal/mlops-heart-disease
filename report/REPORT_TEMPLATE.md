# End-to-End MLOps for Heart Disease Classification
**Course:** AIMLCZG523 — Machine Learning Operations  
**Assignment:** 01  
**Student:** Hemlata Mingwal  
**Date:** 20 Oct, 2026

> Replace every bracketed TODO with results from your own execution. Do not invent metrics, screenshots, URLs, or deployment outcomes.

## 1. Executive Summary
This project implements an end-to-end MLOps workflow for binary heart disease classification using the UCI Heart Disease dataset. The workflow covers data exploration, preprocessing, model development, experiment tracking, automated testing, model packaging, API development, containerization, and Kubernetes deployment.

The dataset contains 303 observations and 14 columns, including 13 input features and one target. Exploratory analysis identified six missing values across the `ca` and `thal` features. Logistic Regression and Random Forest were evaluated using stratified five-fold cross-validation and a held-out test set.

Logistic Regression achieved a mean cross-validation ROC-AUC of 0.9025 and a test ROC-AUC of 0.9665, with test accuracy of 86.89%. Random Forest achieved test accuracy of 90.16% and test ROC-AUC of 0.9502. Logistic Regression was selected for deployment in the serving API.

The model is packaged using Joblib and served through an API with health, prediction, and metrics endpoints. The container was built with Docker and deployed to a local Kubernetes environment with two replicas. Automated tests passed during development. The project demonstrates reproducible model delivery and basic API monitoring. It is an educational demonstration only and is not intended for clinical diagnosis or treatment decisions.

## 2. Problem Statement and Objectives
The objective is to develop a reproducible binary classification system that predicts the presence or absence of heart disease from the dataset's input features. The system demonstrates how a trained machine learning model can be packaged and served through an API.

The project objectives are to:

- Explore and preprocess the dataset, including handling missing values and encoding categorical features.
- Train and compare Logistic Regression and Random Forest classifiers.
- Evaluate models using cross-validation, accuracy, precision, recall, and ROC-AUC.
- Track experiments using MLflow and save the selected model with its preprocessing pipeline.
- Automate quality checks and tests through a CI workflow.
- Package the inference service using Docker and deploy it to local Kubernetes.
- Expose health, prediction, and metrics endpoints for testing and basic monitoring.

Success is assessed through reproducible execution, passing automated tests, measured model performance, successful API responses, and evidence of deployment. Predictions are for educational demonstration only. The dataset is limited in size and representativeness, and the system has not been clinically validated.



## 3. Dataset and Acquisition
The input features are `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`. The target column is `target`.

The binary target represents absence or presence of heart disease. Where the source contains multiple positive severity values, values above zero are mapped to class 1 and zero is mapped to class 0. This mapping converts the problem into the binary classification task required by the assignment. The dataset contains 303 records; 164 belong to class 0 and 139 to class 1.

Source: UCI Machine Learning Repository, Heart Disease dataset, dataset ID 45. The project uses the local dataset file `data/raw/heart_disease.csv`. The original download date should be recorded from project history if available; it has not been independently verified here.

## 4. Exploratory Data Analysis
### Dataset Overview and Data Quality

The Heart Disease UCI dataset used in this project contains **303 observations and 14 columns**, including 13 input features and one target column. The features describe patient characteristics and clinical measurements, including age, sex, chest-pain type, resting blood pressure, cholesterol, maximum heart rate, and other indicators.

The dataset contains 6 missing values in total: 4 in `ca` and 2 in `thal`. The preprocessing pipeline handles missing values using most-frequent-value imputation for the relevant categorical features.

The target distribution is relatively balanced: **54.13% belong to class 0 and 45.87% to class 1**. This distribution provides examples of both classes for model training and evaluation. The project uses a binary classification setup, with class 0 representing absence of heart disease and class 1 representing presence of heart disease.


### 4.1 Target Class Distribution

![Target Class Distribution](screenshots/target_distribution.png)

The dataset contains 54.13% class 0 observations and
45.87% class 1 observations. Class 0 is slightly more
frequent, but the distribution is relatively balanced.
This helps ensure that both classes are represented
during model training and evaluation.

### 4.2 Feature Distributions

![Feature Histograms](screenshots/feature_distributions.png)

Histograms were used to examine the distributions of numerical features, including age, resting blood pressure, cholesterol, maximum heart rate, and oldpeak. These visualizations help identify variation, skewness, and potential outliers in the input variables. The distributions also help inform preprocessing decisions, such as standardization and missing-value handling.


### 4.3 Age Distribution by Heart Disease Status

![Feature Histograms](screenshots/age_distribution_by_heart_disease.png)

The box plot compares age distributions for individuals without heart disease (class 0) and individuals with heart disease (class 1).

The median age for class 1 is higher than that for class 0 in this dataset. The two groups also have overlapping age ranges, indicating that age alone does not completely distinguish between the classes. A younger observation can belong to either group, and older observations can also appear in both groups.

This visualization suggests an association between age and the target in the observed dataset. However, it does not establish causation or demonstrate that age alone can accurately predict heart disease.

### 4.4 Feature Correlation Heatmap

![Feature Histograms](screenshots/feature_correlations.png)

The correlation heatmap summarizes pairwise linear correlations among the numerical and encoded variables, including the target. Redder cells indicate positive correlations, while bluer cells indicate negative correlations. Values closer to zero indicate weaker linear relationships.

The heatmap shows a negative correlation between maximum heart rate (thalach) and the target, while variables such as chest-pain type (cp), exercise-induced angina (exang), oldpeak, and thal show positive correlations with the target in this encoded dataset.

These patterns provide exploratory insights into potential relationships between the input features and the target. Correlation does not imply causation, and the strength and interpretation of correlations involving coded categorical variables depend on how those variables are represented.




## 5. Feature Engineering and Preprocessing
The preprocessing pipeline separates numeric measurements from categorical or low-cardinality coded features. Numeric features are imputed using the median and standardized so that their scales are more comparable. Categorical features are imputed using the most frequent value and one-hot encoded where appropriate.

The preprocessing steps are included in the scikit-learn pipeline so that transformations are fitted on the training data and reused consistently during evaluation and inference. This helps reduce data leakage and prevents the serving API from applying different transformations from those used during training.

The dataset is relatively small, and several variables are encoded as integer values even though they represent categories. The interpretation of these variables depends on the dataset documentation and preprocessing configuration. Imputation can introduce bias, and standardization does not correct for sampling bias or make the data clinically representative.

## 6. Model Development and Evaluation
### Model Comparison and Selection

Two classification algorithms, Logistic Regression and Random Forest, were evaluated. Stratified five-fold cross-validation was performed on the training partition, followed by evaluation on a held-out test partition.

Logistic Regression achieved a mean cross-validation ROC-AUC of 0.9025 with a standard deviation of 0.0148. Random Forest achieved a mean cross-validation ROC-AUC of 0.8955 with a standard deviation of 0.0310.

On the held-out test set, Logistic Regression achieved an accuracy of 86.89%, precision of 81.25%, recall of 92.86%, and ROC-AUC of 0.9665. Random Forest achieved an accuracy of 90.16%, precision of 84.38%, recall of 96.43%, and ROC-AUC of 0.9502.

Random Forest performed better on test accuracy, precision, and recall, while Logistic Regression achieved the higher test ROC-AUC and cross-validation ROC-AUC mean. Logistic Regression was retained as the selected model for deployment, consistent with the model choice used by the serving API.

### Interpretation and Limitations

Recall is relevant because false negatives represent positive cases that the classifier incorrectly predicts as negative. Precision helps describe the proportion of predicted positive cases that are actually positive in the test data. ROC-AUC evaluates the model's ability to rank positive cases above negative cases across classification thresholds.

These results describe performance on the dataset used in this educational project. They do not establish clinical validity, and the predictions must not be used for medical diagnosis or treatment decisions.


## 7. Experiment Tracking with MLflow
MLflow was used to record model training experiments and support comparison between Logistic Regression and Random Forest. The project includes an `mlruns` directory containing local experiment records and an exported model comparison file at `artifacts/model_comparison.csv`.

The recorded evaluation results include cross-validation ROC-AUC statistics and held-out test metrics. The experiment records and associated artifacts should be reviewed to confirm which parameters, classification reports, confusion matrices, and ROC plots were logged for each run.

**Evidence required:** Insert a screenshot of the MLflow experiment page showing the runs and their logged metrics. Include artifact screenshots only for plots and reports that are actually present in the recorded runs.

## 8. Reproducibility and Packaging
The selected Logistic Regression model was serialized using Joblib and saved as `model.joblib`. The saved model includes the preprocessing pipeline for consistent data transformation during training and inference. Project dependencies are maintained in `requirements.txt` to support reproducibility.

## 9. CI/CD and Automated Testing
The project includes a GitHub Actions workflow and automated tests. During local development, the test suite completed successfully with four tests passing. The CI workflow is intended to provide automated quality checks and help identify errors before changes are accepted.

The workflow configuration should be reviewed to document the exact steps it executes, including linting, testing, model training, and any artifacts or logs saved for each run. A workflow should fail when a required check or test fails, preventing an unsuccessful run from being treated as a successful validation.

**Evidence required:** Insert a screenshot of a successful GitHub Actions run. Describe only the steps confirmed in the workflow YAML and the recorded run; do not claim that every step passed in CI solely because local tests passed.

## 10. API, Docker, Deployment and Monitoring
The selected Logistic Regression pipeline is served through an API exposing `/health`, `/predict`, and `/metrics` endpoints. The health endpoint confirmed that the API was running and the model was loaded. A sample prediction request returned HTTP 200 and included a predicted class, label, confidence value, and model name.

The service was containerized using Docker. It was then deployed to a local Kubernetes cluster with two replicas of the API. The deployment and service were applied using the Kubernetes manifest in `deployment/k8s.yaml`. The API was successfully accessed using Kubernetes port forwarding.

The `/metrics` endpoint exposes Prometheus-format metrics, including API request counters and request-duration metrics. This provides basic application-level monitoring. A separate Prometheus server or Grafana dashboard has not been confirmed, so this report does not claim that a full Prometheus/Grafana monitoring stack is deployed.

The deployment is local rather than a publicly hosted cloud deployment. For local testing, start the Kubernetes port-forward command from the project environment:

`kubectl port-forward service/heart-disease-api 8000:8000`

Then open `http://localhost:8000/docs` to test the API interactively. The local address is available only while the port-forward process is running and is not a public endpoint.

**Evidence required:** Add actual screenshots of the Docker build or run, `/health`, `/predict`, `/metrics`, Kubernetes rollout and service status, and relevant application logs.

Architecture:
```text
UCI Heart Disease data
        |
        v
Data cleaning + EDA
        |
        v
Preprocessing pipeline -> Logistic Regression / Random Forest
        |                          |
        +------ MLflow tracking ---+
                                   |
                              Selected model
                                   |
                         FastAPI /predict endpoint
                                   |
                       Docker container / Kubernetes
                                   |
                     Request logs + Prometheus metrics
```

## 11. Risks, Limitations and Ethics
The dataset contains only 303 observations, so its results may not generalize to other populations or clinical settings. Missing values occur in `ca` and `thal`; although imputation enables model execution, it can affect the learned relationships. The dataset may also reflect historical sampling and demographic biases.

Model performance depends on the train/test split, preprocessing choices, decision threshold, and characteristics of the evaluation sample. A probability or confidence score should not be interpreted as a clinically calibrated risk without a dedicated calibration analysis. False negatives and false positives can have different consequences, so threshold selection would require careful evaluation for any real-world application.

The project does not implement clinical validation, prospective testing, or a medical-device approval process. Data and model drift have not been demonstrated as automatically detected. Any future deployment handling real patient data would require appropriate authentication, access control, privacy safeguards, security reviews, monitoring, and governance.

This system is intended solely for educational demonstration and must not be used to diagnose, treat, or make medical decisions about individuals.

## 12. Conclusion and Future Work
This project demonstrates an end-to-end MLOps workflow for heart disease classification, from dataset exploration and preprocessing to model evaluation, experiment tracking, packaging, API serving, Docker containerization, and local Kubernetes deployment.

Logistic Regression achieved a test accuracy of 86.89% and a test ROC-AUC of 0.9665. Random Forest achieved a higher test accuracy of 90.16%, while Logistic Regression had the higher test ROC-AUC and mean cross-validation ROC-AUC. Logistic Regression was selected for the serving API. These results are specific to this dataset and evaluation setup and do not establish clinical usefulness.

The API's health, prediction, and metrics endpoints were tested, and the local Kubernetes deployment ran with two replicas. The project also has automated tests, with four tests passing during local development.

Future work includes probability calibration, decision-threshold analysis, broader external validation, automated data/model drift detection, stronger API security, and deployment to a managed cloud environment. Additional work could integrate a dedicated monitoring dashboard and expand testing for production failure scenarios.

## Appendix A — Setup and Run Commands

# Activate the existing virtual environment
.\.venv\Scripts\Activate.ps1

# Install project dependencies
python -m pip install -r requirements.txt

# Run the project tests
python -m pytest

# Train and evaluate models
python -m src.train

# Build the Docker image
docker build -t heart-disease-mlops .

# Apply the Kubernetes configuration
kubectl apply -f .\deployment\k8s.yaml

# Check deployment and service
kubectl get deployments
kubectl get pods
kubectl get service heart-disease-api

# Access the API locally through port forwarding
kubectl port-forward service/heart-disease-api 8000:8000


## Appendix B — Repository and Video
- GitHub repository: [TODO: your repository URL]
- Pipeline video: [TODO: link to your recording]
- Deployment screenshots: [TODO: folder/path]
