import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Union


def plot_target_distribution(
    y: Union[pd.Series, np.ndarray, list]
) -> None:
    """Affiche la distribution de la variable cible (Y1) à l'aide
    d'un histogramme avec courbe KDE.

    Args:
        y (Union[pd.Series, np.ndarray, list]): La variable cible à visualiser.
        Elle peut être sous forme de liste, tableau NumPy, ou série pandas.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(y, bins=30, kde=True, color='blue')
    plt.title('Distribution de Y1')
    plt.xlabel('Valeurs de Y1')
    plt.ylabel('Fréquence')
    plt.grid()
    plt.show()


def summarize_target(
    y: Union[pd.Series, list]
) -> None:
    """    Affiche la distribution des niveaux de glace prédits ou observés.

    Args:
        y (_type_):  Les niveaux de glace (ex. : '0', '2', '100') à résumer.
    """
    classe = y.value_counts().reset_index()
    print("Distribution des valeurs de Y1 (brutes) :\n", classe)
