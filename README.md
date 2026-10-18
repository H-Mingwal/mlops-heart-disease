# MLOps Assignment 01 — Heart Disease Risk Classification

This repository is a practical implementation scaffold for AIMLCZG523 Assignment 01. It covers data acquisition, EDA, feature preprocessing, two classifiers, cross-validation, MLflow tracking, tests, a FastAPI prediction API, Docker, GitHub Actions, Kubernetes manifests, and basic request logging.

> Academic integrity: run the experiments yourself, inspect the results, personalize the analysis and architecture, and be prepared to explain each decision. Deployment screenshots, a public endpoint, and a video must come from your own successful run; they are not fabricated in this package.

## 1. Prerequisites (Windows + VS Code)

Use **Python 3.12 (64-bit)**. Python 3.14 may not have compatible wheels for all pinned ML packages.

Open the VS Code terminal (PowerShell), navigate to this folder, then run:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal only:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

In VS Code, select interpreter `.venv\Scripts\python.exe`.

## 2. Download data and run EDA

```powershell
python -m src.data
jupyter notebook notebooks/01_eda.ipynb
```

The downloader uses the UCI Heart Disease dataset (repository dataset ID 45) through `ucimlrepo`. It writes `data/raw/heart_disease.csv`. The target is converted to binary: 0 = no disease, values above 0 = disease. This is a modeling label, not a clinical diagnosis.

If the UCI download is unavailable, download the Heart Disease dataset from the UCI Machine Learning Repository and save a CSV at `data/raw/heart_disease.csv`; check its column names and target column before training.

## 3. Train and track experiments

```powershell
python -m src.train
mlflow ui --backend-store-uri ./mlruns
```

Open the MLflow UI URL printed in the terminal (normally `http://127.0.0.1:5000`). Training compares Logistic Regression and Random Forest using a stratified train/test split and stratified cross-validation. The final model is selected by cross-validation ROC-AUC and saved to `models/model.joblib`; metrics and plots are logged to MLflow.

## 4. Tests and lint

```powershell
pytest -q
ruff check .
```

## 5. Run the API locally

Train first so `models/model.joblib` exists.

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs`. Example request body is available in `sample_request.json`. `POST /predict` returns a binary prediction and model probability. This is an educational demo, not a medical device.

## 6. Docker

Install and start Docker Desktop. From the project root:

```powershell
python -m src.train
docker build -t heart-disease-mlops:local .
docker run --rm -p 8000:8000 heart-disease-mlops:local
```

In another terminal:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -InFile .\sample_request.json
```

The Docker image expects the trained `models/model.joblib` to exist before the build.

## 7. CI/CD

Push this repository to your own GitHub repository. The workflow in `.github/workflows/ci.yml` installs dependencies, runs Ruff, runs tests, downloads the UCI data and trains models. The workflow uploads test/training artifacts when available. Review the Actions tab and capture your own successful run.

## 8. Kubernetes (local Minikube or Docker Desktop Kubernetes)

Build the image and make it available to your cluster. For Minikube:

```powershell
minikube start
minikube image build -t heart-disease-mlops:local .
kubectl apply -f deployment/k8s.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,services
kubectl port-forward service/heart-disease-api 8000:8000
```

Then test `http://127.0.0.1:8000/docs`. For Docker Desktop Kubernetes, enable Kubernetes in Docker Desktop and build the image locally. For a cloud cluster, push the image to a registry, replace the image name in `deployment/k8s.yaml`, and configure a suitable LoadBalancer/Ingress and cloud credentials. Do not publish secrets in Git.

## 9. Monitoring

The API writes structured request information to stdout, including endpoint, status, latency, and prediction outcome. `/metrics` exposes Prometheus metrics. Optional local Prometheus/Grafana dashboards can be added; at minimum, verify `/metrics` and container logs and capture screenshots.

## 10. Report and submission checklist

Use `report/REPORT_TEMPLATE.md` as the basis for the required 10-page report. Replace all TODOs with your actual findings and screenshots:
- EDA plots and class balance
- CV/test metrics and confusion matrix / ROC curve
- MLflow experiment comparison
- architecture diagram
- GitHub Actions run
- Docker build and API test
- Kubernetes rollout/service and endpoint test
- monitoring/logs
- repository URL and short pipeline video link

Do not claim cloud deployment unless you actually deploy it. A local Minikube deployment is explicitly allowed by the assignment.

## Dataset notes

The UCI Cleveland data commonly has 303 rows and a target with values 0–4. This project maps target 0 to class 0 and target 1–4 to class 1. The exact usable row count can vary after missing-value handling. Report the values observed in your own run.

## Files

- `src/data.py`: UCI data download and target normalization
- `notebooks/01_eda.ipynb`: EDA notebook
- `src/train.py`: preprocessing, CV, model comparison, MLflow, evaluation artifacts
- `src/api.py`: FastAPI `/predict`, `/health`, `/metrics`
- `tests/`: data and API tests
- `.github/workflows/ci.yml`: CI pipeline
- `Dockerfile`: API container
- `deployment/k8s.yaml`: Deployment and Service
- `report/REPORT_TEMPLATE.md`: report structure
