# 🧊 Classification de la Glace au Groenland à partir de Données Climatiques

Ce projet applique des techniques de **machine learning supervisé** pour prédire la **quantité de glace au Groenland** (faible ou élevée), en utilisant des données climatiques et environnementales. Le but est de développer un modèle robuste à partir de variables comme la température, les vents, la concentration de glace, etc.

---

## 🎯 Objectif

Prédire une variable binaire représentant la quantité de glace (`élevée` ou `faible`), à partir d'un ensemble de variables explicatives climatiques. Les modèles sont évalués en fonction de leur capacité à bien distinguer ces deux classes.

---

## 📁 Structure du projet

Le projet suit une architecture simple et modulaire pour faciliter le développement et la maintenance :  

- Un dossier `data/raw/` contient les fichiers de données brutes, conservés hors du contrôle de version.  
- Le notebook `Ice_Classification.ipynb` centralise toutes les étapes d’analyse, de modélisation et d’évaluation.  
- Le fichier `requirements.txt` liste les dépendances Python nécessaires.  
- Le fichier `.gitignore` exclut notamment les données volumineuses et les fichiers temporaires.  
- Le fichier `README.md` présente une documentation complète du projet.  
- Un `Dockerfile` permet de conteneuriser l’application pour faciliter le déploiement et la reproductibilité.  
- Des workflows GitHub Actions dans `.github/workflows/` automatisent les tests, le linting, et la construction/déploiement.

---

## ⚙️ Méthodologie

1. **Prétraitement** : gestion des valeurs manquantes, standardisation des variables.  
2. **Binarisation de la cible** : transformation de la variable `Y1` en variable binaire (faible vs élevée).  
3. **Modèles testés** :  
   - Arbre de Décision  
   - Random Forest  
   - Extra Trees  
   - Bagging  
4. **Évaluation** avec :  
   - Accuracy, Precision, Recall, F1-score  
   - AUC/ROC  
   - Recherche du **seuil optimal** pour maximiser le F1-score à partir des probabilités prédites.

---

## 🚀 Résultats

Les modèles Random Forest et Extra Trees ont obtenu les meilleurs résultats :  

- **F1-score** : > 0.90  
- **AUC** : ~0.94  
- **Rappel** : 91 %  

Le seuil de décision a été ajusté pour maximiser le F1-score, avec des matrices de confusion détaillées fournies pour chaque modèle.

---

## 🛠 Librairies Python utilisées

- `pandas`  
- `numpy`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`  

---

## 💾 Gestion des données et pipeline

- Les données brutes sont stockées dans `data/raw/` (non versionnées dans Git pour éviter la surcharge du repo).  
- Le notebook `Ice_Classification.ipynb` contient tout le pipeline : exploration, nettoyage, modélisation, validation et visualisation.  
- La structure du projet est pensée pour faciliter l’évolutivité, la maintenance, et l’intégration avec des workflows d’ingénierie de données.

---

## 🐳 Conteneurisation avec Docker

Un fichier `Dockerfile` est fourni pour créer une image Docker contenant l’environnement Python avec toutes les dépendances installées. Cela permet :

- La reproductibilité complète du projet sur n’importe quelle machine.  
- Le déploiement facilité sur serveurs ou cloud.  
- Une isolation propre des dépendances.

---

**Commandes utiles :**

```bash
docker build -t ice-classification .
docker run -it --rm ice-classification

---

## ⚙️ Intégration Continue (CI) & Déploiement Continu (CD)

Le projet utilise GitHub Actions pour automatiser :

- Le **linting** du code avec `pylint` pour garantir la qualité et la conformité au style PEP8.  
- L’exécution des **tests unitaires** avec `pytest` pour vérifier le bon fonctionnement des fonctions clés.  
- L’exécution du notebook pour s’assurer qu’il s’exécute sans erreur.  
- La construction et la publication automatique d’images Docker (optionnel, si connecté à un registre).

Ces pratiques assurent une qualité constante et un déploiement sécurisé.

---

## 📂 Fichiers du projet

| Fichier                         | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `Ice_Classification.ipynb`     | Notebook Python avec tout le pipeline : nettoyage, modélisation, validation |
| `data/raw/data_Features.csv`   | Données d'entrée : température, vents, glace de mer, décharges, etc.        |
| `data/raw/data_Targets.csv`    | Variable cible (`Y1`), binarisée pour la classification                      |
| `requirements.txt`             | Liste des librairies nécessaires à l’exécution du projet                    |
| `Dockerfile`                   | Script pour construire l’image Docker du projet                             |
| `.github/workflows/ci.yml`     | Pipeline GitHub Actions pour CI/CD                                          |
| `.gitignore`                   | Fichiers et dossiers ignorés par Git                                        |
| `README.md`                   | Documentation complète du projet                                            |

---

## 🧪 Installation

1. Cloner le repository :

git clone https://github.com/Aymane-ML/Ice-Classification.git
cd ice-classification

2. Installer les dépendances :

pip install -r requirements.txt

---

## 📬 Auteurs

- **Aymane Mimoun**
- **Mohtadi Hammami**.

