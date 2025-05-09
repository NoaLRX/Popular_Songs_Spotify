# Analyse et Prédiction de la Popularité des Titres Spotify

## Résumé du Projet

Ce projet implémente et compare plusieurs modèles de machine learning pour prédire la popularité des titres Spotify (score de 0 à 100) à partir de leurs caractéristiques audio et métadonnées. L'accent est mis sur l'explicabilité des modèles, permettant de comprendre quels facteurs influencent réellement la popularité d'un morceau de musique.

## Problématique

Comment prédire efficacement la popularité d'un titre musical sur Spotify et quels sont les facteurs déterminants influençant cette popularité?

## Dataset

Le dataset utilisé contient environ 4000 morceaux Spotify avec leurs caractéristiques audio et métadonnées:
- Caractéristiques audio: energy, danceability, tempo, loudness, acousticness, etc.
- Métadonnées: artiste, album, date de sortie, genre, etc.
- Variable cible: track_popularity (score de 0 à 100)

## Méthodologie

### 1. Préparation des Données

- Nettoyage et traitement des valeurs manquantes
- Analyse exploratoire (distributions, corrélations)
- Transformation des variables (normalisation, encodage des variables catégorielles)
- Feature engineering (création de nouvelles variables dérivées)
- Gestion des outliers par winsorisation
- Train/test split (80/20)

### 2. Modélisation

Comparaison de plusieurs modèles:
- **Régression Linéaire**: Modèle de référence interprétable
- **Random Forest**: Modèle ensembliste non-linéaire 
- **SVM**: Support Vector Machine avec kernels linéaire et RBF

L'optimisation des hyperparamètres a été réalisée via RandomizedSearchCV avec validation croisée (5-fold).

### 3. Évaluation

Métriques utilisées:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient de détermination)

### 4. Explicabilité

#### Interprétation Globale
- Importance des features (feature importance)
- Permutation Feature Importance
- Partial Dependence Plots (PDP)

#### Interprétation Locale
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Individual Conditional Expectation (ICE)

## Résultats Principaux

### Performance des Modèles

| Modèle | RMSE | MAE | R² |
|--------|------|-----|---|
| Régression Linéaire | 15.89 | 12.47 | 0.42 |
| Random Forest | 12.34 | 9.18 | 0.68 |
| SVM (RBF) | 14.21 | 10.95 | 0.54 |

Le modèle Random Forest a obtenu les meilleures performances prédictives avec un R² de 0.68.

### Facteurs Influençant la Popularité

Les analyses d'explicabilité ont révélé que:

1. Les facteurs les plus influents sont:
   - La notoriété de l'artiste
   - L'actualité du morceau (date de sortie récente)
   - Le genre musical (pop et rap ayant un avantage)
   - La danceability (caractère dansant)
   - L'energy (intensité énergétique)

2. Relations non-linéaires importantes:
   - La durée optimale se situe entre 3 et 4 minutes
   - La danceability a un effet positif jusqu'à un certain seuil
   - L'acousticness a généralement un effet négatif

3. Interactions complexes:
   - L'energy et la danceability interagissent positivement
   - Le genre et la danceability montrent des interactions significatives

## Conclusion

Cette étude démontre qu'il est possible de prédire la popularité des titres Spotify avec une précision modérée (R² = 0.68). Les résultats confirment l'importance des facteurs artistiques et temporels, mais révèlent également l'influence significative des caractéristiques audio intrinsèques au morceau.

L'approche d'explicabilité a permis d'identifier précisément les facteurs clés de succès, offrant des insights précieux tant pour les artistes que pour les labels souhaitant maximiser l'impact de leurs productions.

## Structure du Projet

```
├── Data/                 # Données Spotify
│   ├── high_popularity_spotify_data.csv
│   └── low_popularity_spotify_data.csv
├── img/                  # Visualisations générées
├── notebook.ipynb        # Notebook principal avec code et analyses
├── README.md             # Documentation du projet
└── requirements.txt      # Dépendances Python
```

## Installation et Utilisation

1. Cloner ce dépôt
2. Créer un environnement virtuel et l'activer:
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/MacOS
# ou
venv\Scripts\activate     # Sur Windows
```
3. Installer les dépendances:
```bash
pip install -r requirements.txt
```
4. Exécuter le notebook:
```bash
jupyter notebook notebook.ipynb
```

## Configuration Technique

- Python 3.9
- Bibliothèques principales: scikit-learn, pandas, numpy, matplotlib, seaborn, shap, lime
- Les hyperparamètres optimaux du modèle Random Forest:
  - n_estimators: 200
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2
  - max_features: 'sqrt'

## Auteur

- Noa