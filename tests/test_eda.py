import sys
import os
from src import eda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest
from typing import Union, List
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)


def test_plot_target_distribution(
) -> None:
    test_inputs: List[Union[List[int], pd.Series, np.ndarray]] = [
        [1, 2, 3, 4, 5, 3, 2],
        pd.Series([1, 2, 2, 3]),
        np.array([0, 0, 1, 1, 2, 2, 2])
    ]
    for y in test_inputs:
        eda.plot_target_distribution(y)
        plt.close()


def test_summarize_target(
    capsys: pytest.CaptureFixture
) -> None:
    y = pd.Series([0, 2, 2, 100, 0, 2, 100, 100])
    eda.summarize_target(y)
    captured = capsys.readouterr()
    assert "Distribution des valeurs de Y1" in captured.out
    assert "0" in captured.out
    assert "2" in captured.out
    assert "100" in captured.out
