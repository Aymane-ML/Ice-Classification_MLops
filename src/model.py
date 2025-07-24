from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from src.preprocessing import binarize_target
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)


def train_model_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    model_class: BaseEstimator,
    param_grid: dict,
    test_size: float = 0.2,
    model_name: str = "model"
) -> Dict[str, Any]:
    y_binary, threshold = binarize_target(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size
    )
    model = model_class
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )
    with mlflow.start_run(run_name=model_name):
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_probs = best_model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.0, 1.0, 0.01)
        f1_scores = [(f1_score(
            y_test,
            (y_probs >= t).astype(int)
        )) for t in thresholds]
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("optimal_threshold", optimal_threshold)
        mlflow.log_metric("cv_best_f1", grid_search.best_score_)
        mlflow.log_metric("optimal_f1", optimal_f1)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred_optimal))
        mlflow.sklearn.log_model(best_model, f"{model_name}_model")
    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "cv_best_f1": grid_search.best_score_,
        "optimal_threshold": optimal_threshold,
        "f1_score": optimal_f1,
        "accuracy": accuracy_score(y_test, y_pred_optimal),
        "classification_report": classification_report(y_test, y_pred_optimal),
        "confusion_matrix": confusion_matrix(y_test, y_pred_optimal),
        "y_test": y_test,
        "y_probs": y_probs
    }


def train_multiple_models_with_mlflow(
    X: pd.DataFrame,
    y: pd.Series,
    models_and_params: List[Tuple[BaseEstimator, dict]],
    test_size: float = 0.2
) -> Dict[str, Any]:
    y_binary, _ = binarize_target(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=test_size, random_state=42
    )
    best_f1 = 0
    best_model_info = None
    model_predictions = {}
    for model_instance, param_grid in models_and_params:
        model_name = model_instance.__class__.__name__
        print(f"Training {model_name}...")
        model_clone = clone(model_instance)
        grid_search = GridSearchCV(
            estimator=model_clone,
            param_grid=param_grid,
            scoring='f1',
            cv=5,
            n_jobs=-1
        )
        with mlflow.start_run(run_name=model_name):
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_probs = best_model.predict_proba(X_test)[:, 1]
            thresholds = np.arange(0, 1, 0.01)
            f1_scores = [f1_score(
                y_test,
                (y_probs >= t).astype(int)
            ) for t in thresholds]
            optimal_idx = np.argmax(f1_scores)
            optimal_f1 = f1_scores[optimal_idx]
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("optimal_f1", optimal_f1)
            mlflow.sklearn.log_model(
                best_model,
                artifact_path=f"{model_name}_model"
            )
            print(f"{model_name} F1 score: {optimal_f1:.4f}")
            model_predictions[model_name] = y_probs
            if optimal_f1 > best_f1:
                best_f1 = optimal_f1
                best_model_info = {
                    "model_name": model_name,
                    "best_model": best_model,
                    "f1_score": optimal_f1,
                    "classification_report": classification_report(
                        y_test,
                        (y_probs >= thresholds[optimal_idx]).astype(int)
                    )
                }
    best_model_info["y_test"] = y_test
    best_model_info["model_probs"] = model_predictions
    return best_model_info
