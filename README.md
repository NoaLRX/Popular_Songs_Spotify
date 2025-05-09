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
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient de détermination)
- Ratio MSE Test/Train (pour évaluer le sur-apprentissage)

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

| Modèle | MSE Train | MSE Test | Ratio MSE Test/Train | R² Train | R² Test |
|--------|-----------|----------|----------------------|----------|---------|
| Régression Linéaire | 159.21 | 180.47 | 1.13 | 0.42 | 0.38 |
| Random Forest | 105.33 | 115.23 | 1.09 | 0.73 | 0.71 |
| SVM (RBF) | 120.86 | 138.52 | 1.15 | 0.58 | 0.54 |

Le modèle Random Forest a obtenu les meilleures performances prédictives avec un R² de 0.71 sur l'ensemble de test. Ce modèle a été sélectionné non seulement pour sa performance supérieure mais aussi pour son équilibre entre capacité prédictive et généralisation. Avec un ratio MSE Test/Train de seulement 1.09 (différence de 9.90 points), il présente le risque de sur-apprentissage le plus faible parmi les modèles testés, garantissant ainsi une meilleure robustesse sur des données non vues.

### Facteurs Influençant la Popularité

Les analyses d'explicabilité ont révélé que:

1. Les facteurs les plus influents sont (par ordre d'importance):
   - track_album_id_numeric (identifiant numérique de l'album)
   - artist_id (identifiant de l'artiste)
   - playlist_id_numeric (identifiant numérique de la playlist)
   - track_age_days (âge du morceau en jours)
   - release_date_numeric (date de sortie numérique)
   - loudness_transformed (puissance sonore transformée)
   - loudness (puissance sonore)

2. Relations non-linéaires importantes:
   - La durée optimale se situe entre 3 et 4 minutes
   - La danceability a un effet positif jusqu'à un certain seuil
   - L'acousticness a généralement un effet négatif

3. Interactions complexes:
   - L'energy et la danceability interagissent positivement
   - Le genre et la danceability montrent des interactions significatives

## Conclusion

Cette étude démontre qu'il est possible de prédire la popularité des titres Spotify avec une précision modérée (R² = 0.71). Les résultats confirment l'importance des facteurs artistiques et temporels, mais révèlent également l'influence significative des caractéristiques audio intrinsèques au morceau.

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

- Python 3.9.7
- Bibliothèques principales: 
  - scikit-learn==1.2.1
  - pandas==2.2.3
  - numpy==1.26.4
  - matplotlib==3.6.3
  - seaborn==0.13.2
  - shap==0.47.2
  - lime==0.2.0.1

- Les hyperparamètres optimaux du modèle Random Forest:
  - n_estimators: 200
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2
  - max_features: 'sqrt'

## Auteur

- Noa