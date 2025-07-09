import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ice_classification import model

def test_train_model_gridsearch():
    np.random.seed(42)
    X = pd.DataFrame({
        'feat1': np.random.randn(100),
        'feat2': np.random.randn(100)
    })
    y = pd.Series(np.concatenate([
        np.random.uniform(0, 1, 95),
        np.random.uniform(100, 400, 5)
    ]))

    model_class = LogisticRegression(max_iter=1000)
    param_grid = {
        'C': [0.1, 1.0]
    }
    results = model.train_model_gridsearch(X, y, model_class, param_grid, test_size=0.2)
    expected_keys = {
        "best_model", "best_params", "cv_best_f1",
        "optimal_threshold", "f1_score", "accuracy",
        "classification_report", "confusion_matrix",
        "y_test", "y_probs"
    }
    assert isinstance(results, dict)
    assert expected_keys.issubset(results.keys())
    assert isinstance(results["best_model"], LogisticRegression)
    assert 0 <= results["f1_score"] <= 1
    assert 0 <= results["accuracy"] <= 1
    assert len(results["y_test"]) == len(results["y_probs"])
    assert isinstance(results["classification_report"], str)
