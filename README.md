# Heart Disease Prediction — End-to-End MLOps Project

## 1. Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for heart disease prediction using structured patient health data.

The project covers data exploration, model training, evaluation, experiment tracking, automated testing, model serving through a REST API, and Docker-based deployment.

**Key objectives:**
- Explore and prepare heart disease data.
- Train and compare Logistic Regression and Random Forest classifiers.
- Evaluate model performance using classification metrics and ROC-AUC.
- Track experiments and model metrics using MLflow.
- Automate code checks, unit tests, and model training with GitHub Actions.
- Serve predictions through a FastAPI REST API.
- Package the application using Docker.

## 2. Technology Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Experiment tracking:** MLflow
- **API framework:** FastAPI
- **API testing:** Swagger UI
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Code quality:** Ruff
- **Testing:** Pytest
- **Version control:** Git and GitHub

## 3. Dataset and Exploratory Data Analysis

The project uses a heart disease dataset containing patient health and clinical measurements.

Exploratory Data Analysis (EDA) includes:
- Target class distribution.
- Missing-value analysis.
- Age distribution by heart disease status.
- Feature distribution and relationship analysis.
- Data preparation for model training.

The dataset contains two target classes: 0 (no heart disease) and 1 (heart disease).

The observed target distribution is approximately:
- Class 0: 54.13%
- Class 1: 45.87%

The classes are reasonably balanced, so both accuracy and class-sensitive metrics such as precision, recall, and ROC-AUC are useful for model evaluation.

## 4. Model Training and Evaluation
Two classification models were trained and evaluated:

1. Logistic Regression
2. Random Forest

The models were compared using cross-validation ROC-AUC and test-set performance metrics.

### Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Mean CV ROC-AUC | 0.9025 | 0.8955 |
| CV ROC-AUC Standard Deviation | 0.0148 | 0.0310 |
| Test Accuracy | 0.8689 | 0.9016 |
| Test Precision | 0.8125 | 0.8438 |
| Test Recall | 0.9286 | 0.9643 |
| Test ROC-AUC | 0.9665 | 0.9502 |

### Results and Interpretation

- Random Forest achieved higher test accuracy (90.16%) and recall (96.43%).
- Logistic Regression achieved higher test ROC-AUC (0.9665) and a higher mean cross-validation ROC-AUC (0.9025).
- Logistic Regression also showed lower variation in cross-validation ROC-AUC across folds.
- Both models demonstrated useful classification performance on the test dataset.

The preferred model depends on the evaluation objective. In a health-related classification task, false negatives and false positives should be considered carefully rather than relying on accuracy alone.

## 5. MLflow Experiment Tracking

MLflow is used to track model training experiments and evaluation metrics.

The experiment is named `heart-disease-mlops`.

The tracked runs include:
- `logistic_regression`
- `random_forest`
- `model_comparison_summary`

Metrics include cross-validation ROC-AUC, test accuracy, precision, recall, and test ROC-AUC, along with the applicable model parameters.

The MLflow UI provides a visual interface for comparing model runs and reviewing logged results.

## 6. API Development and Testing

A FastAPI application exposes the trained model through REST API endpoints.

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/metrics` | Metrics endpoint |
| POST | `/predict` | Generate a heart disease prediction |

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

### Prediction API Response

The `/predict` endpoint was tested successfully through Swagger UI and returned HTTP 200.

The observed response included:
- Prediction: `1`
- Label: `disease_present`
- Confidence: approximately `99.71%`
- Model: `logistic_regression`

The API response also includes a disclaimer stating that the result is for educational prediction only and is not a medical diagnosis.

## 7. Docker Deployment

The application is packaged into a Docker image:

`heart-disease-mlops:latest`

The running container is named:

`heart-disease-api`

The application is exposed on port 8000.

### Build the Docker Image

Run the following command from the project root:

```powershell
docker build -t heart-disease-mlops .
```

### Run the Container

If the container does not already exist, start it using:

```powershell
docker run -d --name heart-disease-api -p 8000:8000 heart-disease-mlops
```

If the container already exists, do not run this command again with the same name.

### Verify the Container

```powershell
docker ps
```

### Test the API

Open the following URL in a browser:

`http://127.0.0.1:8000/docs`

Use the `/predict` endpoint to submit a JSON request and inspect the prediction response.

## 8. Continuous Integration with GitHub Actions

The project includes a GitHub Actions workflow defined in `.github/workflows/ci.yml`.

The CI workflow automates the following steps:

1. Install project dependencies.
2. Run Ruff code-quality checks.
3. Execute unit tests using Pytest.
4. Train the models and log experiments.
5. Upload generated artifacts.

The workflow was successfully executed in GitHub Actions, with the latest successful run showing a green status.

This automation helps verify the project after code changes and reduces the need to run every validation step manually.

## 9. Project Artifacts

The project generates artifacts such as:
- Model comparison results in CSV and JSON formats.
- Classification reports.
- Confusion matrix plots.
- ROC curve plots.
- Trained model files.
- MLflow experiment records.

These artifacts support model evaluation, reproducibility, and reporting.

## 10. Repository Structure

```text
mlops_heart_disease_assignment/
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/
├── data/
│   └── raw/
├── deployment/
├── models/
├── notebooks/
├── report/
├── screenshots/
├── src/
├── tests/
├── Dockerfile
├── requirements.txt
├── sample_request.json
└── README.md
```

## 11. Conclusion

This project demonstrates a practical end-to-end MLOps workflow, from exploratory data analysis and model evaluation to experiment tracking, automated testing, and containerized API deployment.

MLflow provides experiment visibility, GitHub Actions automates quality checks and training, and Docker packages the prediction service for consistent execution.

The project illustrates how machine learning models can be integrated into a testable and deployable software workflow.

**Disclaimer:** This project is intended for educational purposes only. Its predictions must not be used as a substitute for professional medical advice or diagnosis.
