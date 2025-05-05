# Spotify Track Popularity Prediction

Ce projet vise à développer un modèle prédictif pour la variable "Track Popularity" en utilisant les méthodes apprises dans le cours de SVM et réseaux de neurones.

## Contexte

La popularité d'une piste sur Spotify est un score allant de 0 à 100 qui est calculé en fonction du nombre total d'écoutes d'une chanson par rapport aux autres chansons. Prédire cette variable pourrait aider les artistes et labels à comprendre quelles caractéristiques audio influencent la popularité d'une chanson.

## Méthodologie

Deux approches principales ont été implémentées et comparées :

1. **Support Vector Machine (SVM)** - Une méthode de régression non-linéaire utilisant différents noyaux
2. **Réseau de Neurones (MLP)** - Un perceptron multicouche pour modéliser des relations complexes

### Jeu de données

Le jeu de données se compose de deux fichiers CSV :
- `high_popularity_spotify_data.csv` - Contenant des pistes à forte popularité
- `low_popularity_spotify_data.csv` - Contenant des pistes à faible popularité

Les caractéristiques numériques utilisées pour la prédiction incluent :
- energy
- tempo
- danceability
- loudness
- liveness
- valence
- speechiness
- instrumentalness
- acousticness
- duration_ms

### Prétraitement des données

1. Fusion des deux jeux de données
2. Suppression des valeurs manquantes
3. Standardisation des caractéristiques (mise à l'échelle)
4. Division en ensembles d'entraînement et de test (80/20)

### Modèle SVM

Selon le cours, les SVM sont efficaces pour des problèmes de régression non-linéaire. Nous avons utilisé un SVR (Support Vector Regression) avec différents noyaux et hyperparamètres :

- Un pipeline incluant la standardisation des données
- Recherche par grille des meilleurs hyperparamètres :
  - C (paramètre de régularisation)
  - Kernel (linéaire ou RBF)
  - Gamma (échelle, auto, ou valeurs spécifiques)
- Validation croisée à 5 plis

### Modèle de Réseau de Neurones

Selon les recommandations du cours ANN, nous avons construit un réseau de neurones avec :

- 3 couches cachées (100, 50, 25 neurones)
- Fonction d'activation ReLU pour les couches cachées
- Pas de fonction d'activation pour la couche de sortie (problème de régression)
- Dropout pour éviter le surapprentissage
- Optimiseur Adam
- Fonction de perte MSE (erreur quadratique moyenne)
- Early stopping pour éviter le surapprentissage

## Évaluation et comparaison des modèles

Les modèles sont évalués en fonction de :
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (coefficient de détermination)

## Résultats et analyse

Après l'entraînement, les performances des deux modèles sont comparées pour déterminer lequel est le plus adapté à la prédiction de la popularité des pistes Spotify.

L'analyse comprend également :
- La distribution de la variable cible
- Les corrélations entre les caractéristiques et la popularité
- Pour le SVM linéaire, l'importance des caractéristiques
- Pour le réseau de neurones, les courbes d'apprentissage

## Conclusion

Le script identifie automatiquement la meilleure méthode (SVM ou réseau de neurones) en fonction du score R², qui mesure la proportion de la variance dans la variable cible qui est prédictible à partir des variables explicatives.

## Comment exécuter le code

```bash
python spotify_popularity_prediction.py
```

## Sorties

Le script génère plusieurs visualisations :
- Distribution de la popularité des pistes
- Corrélations des caractéristiques avec la popularité
- Performances prédictives des modèles
- Courbes d'apprentissage du réseau de neurones
- Importance des caractéristiques pour le SVM (si noyau linéaire)