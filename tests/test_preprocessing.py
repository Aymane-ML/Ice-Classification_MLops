import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from src import preprocessing 

def test_load_data(tmp_path):
    df_features = pd.DataFrame({
        't2m': [-21.9, -23.7],
        'u10': [-0.97, -6.50],
        'v10': [3.14, 2.49],
        'SST': [-1.69, -1.69],
        'SIC': [90.7, 88.5],
        'r1_MAR': [0.0345, 0.0345],
        'r2_MAR': [0.0333, 0.0333],
        'r3_MAR': [0.0, 0.0],
        'r4_MAR': [0.0, 0.0],
        'r5_MAR': [0.0, 0.0],
        'year': [2013, 2013],
        'month': [1, 1],
        'day': [1, 2],
    })
    df_targets = pd.DataFrame({
        'Y1': [0, 100]
    })
    features_path = tmp_path / "data_Features.csv"
    targets_path = tmp_path / "data_Targets.csv"
    df_features.to_csv(features_path, index=False)
    df_targets.to_csv(targets_path, index=False)
    X, y = preprocessing.load_data(str(features_path), str(targets_path))
    pd.testing.assert_frame_equal(X, df_features)
    pd.testing.assert_frame_equal(y, df_targets)

def test_preprocess_features():
    df = pd.DataFrame({
        'time': ['2023-01-01', '2023-02-15', '2023-12-31'],
        't2m': [-21.9, -23.7, -25.6]
    })
    result = preprocessing.preprocess_features(df)
    expected = pd.DataFrame({
        't2m': [-21.9, -23.7, -25.6],
        'year': [2023, 2023, 2023],
        'month': [1, 2, 12],
        'day': [1, 15, 31]
    })
    expected = expected.astype(result.dtypes.to_dict())  # <- fix dtype
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_prepare_data(monkeypatch):
    fake_features = pd.DataFrame({'time': ['2023-01-01'], 'feat1': [1]})
    fake_targets = pd.DataFrame({'Y1': [5]})

    def fake_load_data():
        return fake_features, fake_targets

    def fake_preprocess_features(df):
        return pd.DataFrame({'feat1': [1], 'year': [2023], 'month': [1], 'day': [1]})

    monkeypatch.setattr(preprocessing, "load_data", fake_load_data)
    monkeypatch.setattr(preprocessing, "preprocess_features", fake_preprocess_features)

    X, y = preprocessing.prepare_data()

    expected_X = pd.DataFrame({'feat1': [1], 'year': [2023], 'month': [1], 'day': [1]})
    expected_y = pd.Series([5], name='Y1')

    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_series_equal(y.reset_index(drop=True), expected_y)

def test_binarize_target():
    y = pd.Series([1, 2, 3, 4, 5])
    y_binary, threshold = preprocessing.binarize_target(y)
    expected_threshold = 3
    expected_binary = pd.Series([0, 0, 0, 1, 1])

    assert threshold == expected_threshold
    pd.testing.assert_series_equal(y_binary.reset_index(drop=True), expected_binary)

def test_scale_features():
    X = pd.DataFrame({
        't2m': [-21.9, -23.7, -25.6, -23.5, -22.8],
        'u10': [-0.97, -6.50, -3.55, -1.88, -2.74],
        'v10': [3.14, 2.49, 1.02, -3.48, -3.49],
        'SST': [-1.69, -1.69, -1.68, -1.69, -1.68],
        'SIC': [90.7, 88.5, 88.7, 89.1, 91.6],
        'r1_MAR': [0.0345, 0.0345, 0.0345, 0.0345, 0.0344],
        'r2_MAR': [0.0333, 0.0333, 0.0333, 0.0333, 0.0332],
        'r3_MAR': [0.0, 0.0, 0.0, 0.0, 0.0],
        'r4_MAR': [0.0, 0.0, 0.0, 0.0, 0.0],
        'r5_MAR': [0.0, 0.0, 0.0, 0.0, 0.0],
        'year': [2013, 2013, 2013, 2013, 2013],
        'month': [1, 1, 1, 1, 1],
        'day': [1, 2, 3, 4, 5],
    })
    X_scaled, scaler = preprocessing.scale_features(X)
    assert isinstance(scaler, StandardScaler)
    assert X_scaled.shape == X.shape
    np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-7)

    stds = X_scaled.std(axis=0)
    mask = stds > 1e-8  # ignore colonnes constantes
    np.testing.assert_allclose(stds[mask], 1, atol=1e-7)
