# End-to-End MLOps for Heart Disease Classification
**Course:** AIMLCZG523 — Machine Learning Operations  
**Assignment:** 01  
**Student:** [Your name]  
**Date:** [Submission date]

> Replace every bracketed TODO with results from your own execution. Do not invent metrics, screenshots, URLs, or deployment outcomes.

## 1. Executive Summary
[TODO: Summarize the problem, solution, best model, measured results, and deployment status in 150–250 words.]

## 2. Problem Statement and Objectives
The objective is to build a reproducible binary classifier from the UCI Heart Disease dataset and deliver it through a monitored API. The output is for educational demonstration only, not clinical decision-making.

[TODO: Explain intended use, limitations, and measurable success criteria.]

## 3. Dataset and Acquisition
Source: UCI Machine Learning Repository, Heart Disease dataset (ID 45).  
[TODO: Record download date, number of rows and columns, feature names, target distribution, missing-value counts, and any exclusions.]

The original target may encode values 0–4. This implementation maps 0 to 0 and values above 0 to 1. Explain why this binary mapping matches the assignment problem.

## 4. Exploratory Data Analysis
[TODO: Insert class-balance chart, selected histograms, correlation heatmap, and short interpretations. Describe missing data and feature distributions.]

## 5. Feature Engineering and Preprocessing
The pipeline uses median imputation and standardization for numeric features, and most-frequent imputation plus one-hot encoding for low-cardinality features. Preprocessing is fitted inside a scikit-learn Pipeline to reduce leakage between training and evaluation.

[TODO: Explain which features your run treated as categorical/numeric and why. Discuss limitations.]

## 6. Model Development and Evaluation
Models: Logistic Regression and Random Forest. Evaluation: stratified 5-fold cross-validation on the training partition, plus a held-out test partition.

[TODO: Insert actual CV ROC-AUC mean/std, test accuracy, precision, recall, ROC-AUC, confusion matrices and ROC curves. State the model selection criterion and chosen model. Do not report expected or invented numbers.]

Discuss false negatives versus false positives and why recall/ROC-AUC are relevant for this educational risk-classification task. Avoid clinical claims.

## 7. Experiment Tracking with MLflow
[TODO: Add screenshot of the MLflow experiment page. Describe logged parameters, metrics, classification reports, confusion matrices, ROC plots, and model comparison.]

## 8. Reproducibility and Packaging
The selected Logistic Regression model was serialized using Joblib and saved as `model.joblib`. The saved model includes the preprocessing pipeline for consistent data transformation during training and inference. Project dependencies are maintained in `requirements.txt` to support reproducibility.

## 9. CI/CD and Automated Testing
[TODO: Add GitHub Actions screenshot. Describe linting, unit tests, training step, artifacts, and what causes the workflow to fail.]

## 10. API, Docker, Deployment and Monitoring
[TODO: Add architecture diagram and screenshots of Docker build/run, `/health`, `/predict`, `/metrics`, Kubernetes rollout/service, and logs. State whether deployment is local or cloud. Include the actual endpoint only if publicly deployed.]

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
[TODO: Discuss dataset size, representativeness, missing values, possible bias, threshold selection, probability calibration, data drift, privacy, and the fact that this is not a medical device.]

## 12. Conclusion and Future Work
[TODO: Summarize measured results and actual deployment state. Potential future work: calibration, threshold tuning, data/model drift alerts, authentication, privacy controls, and a managed cloud deployment.]

## Appendix A — Setup and Run Commands
[TODO: Include commands you used and test evidence.]

## Appendix B — Repository and Video
- GitHub repository: [TODO: your repository URL]
- Pipeline video: [TODO: link to your recording]
- Deployment screenshots: [TODO: folder/path]
