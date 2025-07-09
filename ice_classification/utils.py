from typing import Dict
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(
    y_test: np.ndarray,
    model_probs_dict: Dict[str, np.ndarray]
)->None:
    """Affiche les courbes ROC pour plusieurs modèles à partir de leurs probabilités prédites.
    
    - y_test : Vraies étiquettes binaires (0 ou 1) du niveau de glace (faible ou élevé).
    - model_probs_dict : dict {nom du modèle : y_probs (probabilités classe positive)}
    """
    plt.figure(figsize=(8, 6))
    
    for model_name, y_prob in model_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC des Modèles')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
