import numpy as np
import pytest
from src import utils


def test_plot_roc_curve(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    y_test = np.array([0, 0, 1, 1])
    model_probs_dict = {
        'model1': np.array([0.1, 0.4, 0.35, 0.8]),
        'model2': np.array([0.2, 0.3, 0.5, 0.7]),
    }

    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)
    utils.plot_roc_curve(y_test, model_probs_dict)
