import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_data(
    features_path: str="data/raw/data_Features.csv",
    targets_path: str="data/raw/data_Targets.csv"
)->Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les fichiers de données d'entrée (features) et de sortie (targets) depuis des chemins CSV.

    Args:
        features_path (str, optional): Chemin du fichier CSV contenant les variables explicatives.
        Defaults to "data/raw/data_Features.csv".
        targets_path (str, optional): Chemin du fichier CSV contenant la variable cible.
        Defaults to "data/raw/data_Targets.csv".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Deux DataFrames : df_features (X) et df_targets (y).
    """
    df_features = pd.read_csv(features_path)
    df_targets = pd.read_csv(targets_path)
    return df_features, df_targets

def preprocess_features(
    df_features: pd.DataFrame
)->pd.DataFrame:
    """Transforme les colonnes temporelles en variables numériques (année, mois, jour).

    Args:
        df_features (pd.DataFrame):  DataFrame contenant les variables explicatives, incluant une colonne 'time'.

    Returns:
        pd.DataFrame: DataFrame nettoyé avec les colonnes 'year', 'month', 'day' à la place de 'time'.
    """
    df=df_features.copy()
    df['time']=pd.to_datetime(df['time'], errors='coerce')
    df['year']=df['time'].dt.year
    df['month']=df['time'].dt.month
    df['day']=df['time'].dt.day
    df.drop(columns=['time'], inplace=True)
    return df

def prepare_data(
)->Tuple[pd.DataFrame, pd.Series]:
    """Prépare les données pour l'entraînement du modèle.
    - Charge les données
    - Prétraite les features (transformation temporelle)
    - Extrait la cible 'Y1'

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            X : Variables explicatives
            y : Variable cible (Y1)
    """
    df_features, df_targets=load_data()
    X=preprocess_features(df_features)
    y=df_targets['Y1']
    return X, y

def scale_features(
    X: pd.DataFrame
)->Tuple[pd.DataFrame, StandardScaler]:
    """Applique une normalisation (StandardScaler) aux features.

    Args:
        X (pd.DataFrame): Les variables explicatives à normaliser.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]:
            X_scaled : Données normalisées
            scaler : L'instance StandardScaler utilisée
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def binarize_target(
    y: pd.Series
)->Tuple[pd.Series, float]:
    """ Transforme une variable continue en binaire selon sa médiane.

    Tout élément > médiane devient 1, sinon 0.

    Args:
        y (pd.Series): Variable cible continue (ex : niveau de glace).

    Returns:
        Tuple[pd.Series, float]
            y_binary : Série binaire
            threshold : Valeur du seuil (médiane)
    """
    threshold = y.median()
    y_binary = (y > threshold).astype(int)
    return y_binary, threshold

    