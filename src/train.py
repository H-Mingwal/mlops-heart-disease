"""Train two classifiers, compare cross-validation ROC-AUC, and log with MLflow."""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_score,
    recall_score, roc_auc_score, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.data import load_data

RANDOM_STATE = 42
TARGET = "target"
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")


def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    categorical = [c for c in X.columns if X[c].nunique(dropna=True) <= 10]
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([("preprocessor", prep), ("classifier", estimator)])


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    if TARGET not in df:
        raise ValueError("Dataset must contain a target column.")
    df = df.replace("?", np.nan)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X = df.drop(columns=[TARGET]).apply(pd.to_numeric, errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    X = X.dropna(axis=1, how="all")
    if y.nunique() != 2:
        raise ValueError(f"Expected binary target, found {sorted(y.unique())}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    mlflow.set_experiment("heart-disease-mlops")
    results = []
    best_name, best_score, best_pipeline = None, -np.inf, None

    for name, estimator in candidates.items():
        pipe = build_pipeline(X_train, estimator)
        with mlflow.start_run(run_name=name):
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=None)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics = {
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std()),
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_precision": float(precision_score(y_test, pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, proba)),
            }
            mlflow.log_params({"model": name, "random_state": RANDOM_STATE, "test_size": 0.2, "cv_folds": 5})
            mlflow.log_metrics(metrics)
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)
            with open(ARTIFACT_DIR / f"{name}_classification_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(str(ARTIFACT_DIR / f"{name}_classification_report.json"))

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set(title=f"{name.replace('_', ' ').title()} — Confusion Matrix",
                   xlabel="Predicted", ylabel="Actual")
            fig.tight_layout()
            plot_path = ARTIFACT_DIR / f"{name}_confusion_matrix.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path))

            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
            ax.set_title(f"{name.replace('_', ' ').title()} — ROC Curve")
            fig.tight_layout()
            roc_path = ARTIFACT_DIR / f"{name}_roc_curve.png"
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(roc_path))

            results.append({"model": name, **metrics})
            if metrics["cv_roc_auc_mean"] > best_score:
                best_name, best_score, best_pipeline = name, metrics["cv_roc_auc_mean"], pipe

    results_df = pd.DataFrame(results)
    results_df.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    with mlflow.start_run(run_name="model_comparison_summary"):
        mlflow.log_artifact(str(ARTIFACT_DIR / "model_comparison.csv"))
        mlflow.log_params({"selected_model": best_name, "selection_metric": "mean 5-fold CV ROC-AUC"})
        mlflow.log_metric("best_cv_roc_auc_mean", float(best_score))

    joblib.dump({"model": best_pipeline, "feature_names": list(X.columns), "selected_model": best_name},
                MODEL_DIR / "model.joblib")
    results_df.to_json(ARTIFACT_DIR / "model_comparison.json", orient="records", indent=2)
    print(results_df.to_string(index=False))
    print(f"\nSelected {best_name} by mean cross-validation ROC-AUC ({best_score:.4f})")
    print(f"Saved model to {MODEL_DIR / 'model.joblib'}")
    print("Review artifacts/ and the MLflow UI before writing conclusions.")


if __name__ == "__main__":
    main()
