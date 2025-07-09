from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from ice_classification.preprocessing import binarize_target
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)

def train_model_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    model_class: BaseEstimator,
    param_grid: dict,
    test_size: float=0.2
)->Dict[str, Any]:
    """
    Entraîne un modèle avec GridSearchCV et détermine le seuil optimal pour le F1-score.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Niveau de glasse (Cible).
        model_class (sklearn.base.BaseEstimator): Classe du modèle (ex: RandomForestClassifier).
        param_grid (dict): Grille des hyperparamètres.
        test_size (float): Proportion du test set.
        random_state (int): Pour reproductibilité.

    Returns:
        dict: Résultats (modèle, seuil optimal, scores, etc.).
    """
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
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_probs = best_model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = [(f1_score(y_test, (y_probs >= t).astype(int))) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
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
