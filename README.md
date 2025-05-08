# Projet d'Explicabilité - Prédiction de la Popularité des Titres Spotify

Ce projet implémente plusieurs modèles de machine learning pour prédire la popularité des titres Spotify, en mettant l'accent sur l'explicabilité des modèles. Il applique la méthodologie du TD2 d'explicabilité.

## Objectifs

- Prédire la popularité d'un titre Spotify à partir de ses caractéristiques audio et métadonnées
- Comparer différents modèles prédictifs (Régression Linéaire, Random Forest, SVM)
- Interpréter les modèles de manière globale et locale
- Identifier les facteurs clés qui influencent la popularité d'un titre

## Structure du Projet

```
├── Data/                 # Dossier contenant les données Spotify
│   ├── high_popularity_spotify_data.csv
│   └── low_popularity_spotify_data.csv
├── img/                  # Dossier contenant les visualisations générées
├── venv/                 # Environnement virtuel Python
├── ultime.py             # Script principal d'analyse
├── README.md             # Documentation du projet
└── requirements.txt      # Dépendances Python
```

## Installation

1. Cloner ce dépôt
2. Créer un environnement virtuel et l'activer :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/MacOS
# ou
venv\Scripts\activate     # Sur Windows
```
3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Exécuter le script principal :
```bash
python ultime.py
```

Le script effectue automatiquement :
1. Le chargement et la préparation des données
2. La création et l'évaluation des modèles baseline
3. L'interprétation du modèle linéaire
4. L'optimisation des modèles complexes
5. L'interprétation globale et locale du meilleur modèle
6. La génération de visualisations dans le dossier `img/`

## Méthodologie

### Préparation des Données

- Nettoyage et analyse exploratoire des données
- Transformation des variables (logarithmique pour les distributions asymétriques)
- Création de features dérivées (statistiques par artiste, âge du morceau)
- Winsorisation pour gérer les outliers
- Standardisation des variables numériques uniquement

### Modélisation

- **Régression Linéaire** : Modèle interprétable pour comprendre les relations linéaires
- **Random Forest** : Modèle complexe optimisé avec RandomizedSearchCV
- **SVM** : Support Vector Machine avec différents noyaux

### Explicabilité

- **Interprétation Globale** :
  - Importance des features
  - Permutation Feature Importance
  - Partial Dependence Plots (PDP)

- **Explicabilité Locale** :
  - Individual Conditional Expectation (ICE)
  - LIME (Local Interpretable Model-agnostic Explanations)
  - SHAP (SHapley Additive exPlanations)

## Résultats

Les résultats complets sont générés lors de l'exécution du script, incluant :
- Comparaison des performances des modèles (RMSE, R², MAE)
- Identification des variables les plus importantes
- Visualisations interactives des relations entre variables
- Explications détaillées de prédictions individuelles

## Configuration Python

Ce projet a été développé avec Python 3.9 et les bibliothèques suivantes :
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
- lime

Pour les versions précises, voir `requirements.txt`.

## Hyperparamètres Optimaux

Les hyperparamètres optimaux sont déterminés dynamiquement pendant l'exécution du script via la validation croisée. Ils sont affichés dans les résultats finaux.

## Licence

Ce projet est fourni sous licence MIT.

## Contributeurs

- Noa (étudiant)

## Remerciements

- Professeur de Data Science pour le TD2 d'explicabilité qui a servi de méthodologie de référence.