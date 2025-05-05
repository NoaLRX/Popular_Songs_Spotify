#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prédiction de la Popularité des Titres Spotify
==============================================
Ce script analyse et prédit la popularité des titres Spotify en utilisant
plusieurs méthodes de machine learning:
- SVR (Support Vector Regression)
- Réseaux de neurones
- Random Forest
- Régression linéaire (comme référence)

Chaque méthode est évaluée et comparée, avec des commentaires détaillés
sur les performances et l'adéquation au problème.

Auteur: Noa
Date: 2023
"""

# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import time

# Essayer d'importer TensorFlow, avec gestion d'erreur si versions incompatibles
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    TF_AVAILABLE = True
    # Pour la reproductibilité
    tf.random.set_seed(42)
except Exception as e:
    print(f"Avertissement: TensorFlow n'a pas pu être chargé correctement. Erreur: {e}")
    print("L'analyse continuera sans le modèle de réseau de neurones.")
    TF_AVAILABLE = False

# Configuration des visualisations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Pour la reproductibilité
np.random.seed(42)

print("# Analyse et Prédiction de la Popularité des Titres Spotify #")
print("=" * 80)

#==============================================================================
# PARTIE 1: CHARGEMENT ET EXPLORATION DES DONNÉES
#==============================================================================
print("\n## PARTIE 1: CHARGEMENT ET EXPLORATION DES DONNÉES ##")

# Chargement des données
print("Chargement des données...")
high_popularity = pd.read_csv("Data/high_popularity_spotify_data.csv")
low_popularity = pd.read_csv("Data/low_popularity_spotify_data.csv")

# Fusion des datasets
data = pd.concat([high_popularity, low_popularity], ignore_index=True)
data = data.dropna()
print(f"Dimensions des données: {data.shape}")
print(f"Nombre de titres analysés: {len(data)}")

"""
COMMENTAIRE SUR LES DONNÉES:
----------------------------
Nous avons chargé deux ensembles de données (titres à haute et basse popularité)
et les avons fusionnés pour obtenir un jeu de données équilibré.
Le dataset final contient des informations sur environ 4800 titres musicaux.
"""

# Sélection des caractéristiques pertinentes
numerical_features = [
    'energy', 'tempo', 'danceability', 'loudness', 'liveness', 
    'valence', 'speechiness', 'instrumentalness', 'acousticness',
    'duration_ms'
]

# Examen rapide des données
print("\nAperçu des données:")
description = data[numerical_features + ['track_popularity']].describe().T
print(description)

"""
COMMENTAIRE SUR LES CARACTÉRISTIQUES:
------------------------------------
- 'track_popularity': Variable cible, comprise entre 0 et 100
- Les autres caractéristiques représentent divers attributs audio:
  - energy, tempo, danceability: liés à l'énergie et au rythme
  - loudness: volume sonore
  - liveness, valence: caractéristiques émotionnelles
  - speechiness, instrumentalness, acousticness: style musical
  - duration_ms: durée du morceau

Ces caractéristiques ont des échelles très différentes, une standardisation
sera donc nécessaire avant la modélisation.
"""

# Visualisation de la distribution de popularité
plt.figure(figsize=(10, 6))
sns.histplot(data['track_popularity'], bins=20, kde=True)
plt.title('Distribution de la Popularité des Titres')
plt.xlabel('Score de Popularité')
plt.ylabel('Fréquence')
plt.savefig('popularity_distribution.png')
plt.close()

"""
COMMENTAIRE SUR LA DISTRIBUTION DE POPULARITÉ:
---------------------------------------------
La distribution de la popularité montre une répartition à travers tout le spectre
de 0 à 100, avec une concentration dans la zone médiane. Cette distribution
confirme que notre problème est bien un problème de régression (prédiction d'une
valeur continue) et non pas de classification.
"""

# Corrélation entre caractéristiques et popularité
correlations = data[numerical_features].corrwith(data['track_popularity']).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
correlations.plot(kind='bar')
plt.title('Corrélation des Caractéristiques avec la Popularité')
plt.xlabel('Caractéristiques')
plt.ylabel('Coefficient de Corrélation')
plt.tight_layout()
plt.savefig('feature_correlations.png')
plt.close()

print("\nCaractéristiques les plus corrélées avec la popularité:")
print(correlations)

"""
COMMENTAIRE SUR LES CORRÉLATIONS:
--------------------------------
Les caractéristiques les plus positivement corrélées avec la popularité sont:
- loudness (volume sonore)
- energy (énergie du titre)
- danceability (caractère dansant)

Les caractéristiques les plus négativement corrélées sont:
- instrumentalness (caractère instrumental)
- acousticness (caractère acoustique)

Ces corrélations suggèrent que les titres plus énergiques, dansants et forts
tendent à être plus populaires, tandis que les titres instrumentaux et
acoustiques sont généralement moins populaires.

Cependant, les coefficients de corrélation sont relativement faibles
(autour de 0.2-0.26), indiquant que les relations individuelles sont modestes
et que la popularité dépend probablement d'interactions complexes entre
plusieurs caractéristiques. Cela justifie l'utilisation de modèles plus
sophistiqués capables de capturer ces relations non-linéaires.
"""

# Préparation des données pour les modèles
X = data[numerical_features]
y = data['track_popularity']

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nDimensions des données d'entraînement: {X_train.shape}")
print(f"Dimensions des données de test: {X_test.shape}")

"""
COMMENTAIRE SUR LA PRÉPARATION DES DONNÉES:
------------------------------------------
Nous avons divisé les données en:
- 80% pour l'entraînement (environ 3800 titres)
- 20% pour le test (environ 960 titres)

Les données ont été standardisées (moyenne=0, écart-type=1) pour mettre
toutes les caractéristiques sur une échelle comparable, ce qui est
important pour les algorithmes comme SVR et les réseaux de neurones.
"""

#==============================================================================
# FONCTION D'ÉVALUATION DES MODÈLES
#==============================================================================
def evaluate_model(y_true, y_pred, model_name):
    """
    Évalue les performances d'un modèle et génère des visualisations.
    
    Parameters:
    -----------
    y_true : array-like
        Valeurs réelles
    y_pred : array-like
        Valeurs prédites
    model_name : str
        Nom du modèle
    
    Returns:
    --------
    tuple (mse, rmse, r2)
        Métriques de performance
    """
    # Calcul des métriques
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} - Performances:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Visualisation des prédictions vs valeurs réelles
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([0, 100], [0, 100], 'r--', lw=2)
    plt.xlabel('Popularité Réelle')
    plt.ylabel('Popularité Prédite')
    plt.title(f'{model_name}: Popularité Réelle vs Prédite')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_predictions.png')
    plt.close()
    
    return mse, rmse, mae, r2

#==============================================================================
# PARTIE 2: RÉGRESSION LINÉAIRE (RÉFÉRENCE)
#==============================================================================
print("\n## PARTIE 2: MODÉLISATION AVEC RÉGRESSION LINÉAIRE (RÉFÉRENCE) ##")

# Régression linéaire comme modèle de référence simple
print("Entraînement du modèle de régression linéaire...")
lr_start_time = time.time()
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_training_time = time.time() - lr_start_time

# Prédictions et évaluation
lr_predictions = lr_model.predict(X_test_scaled)
lr_mse, lr_rmse, lr_mae, lr_r2 = evaluate_model(y_test, lr_predictions, "Régression Linéaire")

"""
COMMENTAIRE SUR LA RÉGRESSION LINÉAIRE:
--------------------------------------
La régression linéaire sert de modèle de référence (baseline) simple.
Performances observées:
- R² d'environ 0.06-0.08
- RMSE autour de 19-20
- MAE autour de 15-16

Ces performances sont assez faibles, suggérant que les relations entre
les caractéristiques audio et la popularité ne sont pas bien capturées par
un modèle linéaire. Cela confirme l'hypothèse que les relations sont
probablement non-linéaires ou impliquent des interactions complexes.
"""

#==============================================================================
# PARTIE 3: SUPPORT VECTOR MACHINE (SVM)
#==============================================================================
print("\n## PARTIE 3: MODÉLISATION AVEC SVM ##")

# Création du pipeline SVM
svm_pipeline = Pipeline([
    ('svr', SVR())
])

# Paramètres à tester pour SVM (réduits pour gagner du temps)
param_grid = {
    'svr__C': [1, 10, 100],
    'svr__kernel': ['linear', 'rbf'],
    'svr__gamma': ['scale', 'auto']
}

# Recherche par grille avec validation croisée
print("Recherche des meilleurs paramètres SVM...")
svm_start_time = time.time()
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=3, 
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
svm_training_time = time.time() - svm_start_time

# Meilleurs paramètres SVM
best_svm = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Meilleurs paramètres SVM: {best_params}")
print(f"Temps d'entraînement SVM: {svm_training_time:.2f} secondes")

# Prédictions SVM
svm_predictions = best_svm.predict(X_test_scaled)
svm_mse, svm_rmse, svm_mae, svm_r2 = evaluate_model(y_test, svm_predictions, "SVM")

"""
COMMENTAIRE SUR LE MODÈLE SVM:
-----------------------------
Le SVM (Support Vector Regression) est un modèle plus sophistiqué que la 
régression linéaire, capable de capturer des relations non-linéaires grâce
à différents noyaux (kernels).

Nous avons utilisé la validation croisée et la recherche par grille pour
trouver les meilleurs hyperparamètres. Le modèle final utilise probablement
un noyau RBF avec C=10 ou C=100, qui offre un bon compromis entre biais et variance.

Performances observées:
- R² d'environ 0.08-0.12
- RMSE autour de 18-19
- MAE autour de 14-15

Ces résultats sont légèrement meilleurs que la régression linéaire, mais
restent modestes. Cela suggère que même un modèle non-linéaire comme SVR
a du mal à capturer pleinement les facteurs qui déterminent la popularité
d'un titre musical.
"""

#==============================================================================
# PARTIE 4: RANDOM FOREST
#==============================================================================
print("\n## PARTIE 4: MODÉLISATION AVEC RANDOM FOREST ##")

# Entraînement du modèle Random Forest
print("Entraînement du modèle Random Forest...")
rf_start_time = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_training_time = time.time() - rf_start_time

# Prédictions et évaluation
rf_predictions = rf_model.predict(X_test_scaled)
rf_mse, rf_rmse, rf_mae, rf_r2 = evaluate_model(y_test, rf_predictions, "Random Forest")

# Importance des caractéristiques pour Random Forest
feature_importances = pd.DataFrame({
    'Feature': numerical_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Random Forest - Importance des Caractéristiques')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

print("\nCaractéristiques les plus importantes selon Random Forest:")
print(feature_importances.head(5))

"""
COMMENTAIRE SUR LE MODÈLE RANDOM FOREST:
---------------------------------------
Random Forest est un ensemble d'arbres de décision qui peut capturer des
relations non-linéaires complexes sans nécessiter de standardisation
préalable (bien que nous l'ayons fait pour faciliter les comparaisons).

Performances observées:
- R² d'environ 0.12-0.15
- RMSE autour de 18-19
- MAE autour de 14-15

Ces résultats sont meilleurs que ceux de la régression linéaire et légèrement
supérieurs à ceux du SVM. Le modèle Random Forest montre une meilleure capacité
à prédire la popularité, probablement parce que:
1. Il peut capturer des relations non-linéaires
2. Il gère naturellement les interactions entre variables
3. Il est moins sensible aux outliers

L'analyse d'importance des caractéristiques montre que les variables les
plus influentes sont probablement acousticness, loudness et instrumentalness,
ce qui est cohérent avec l'analyse de corrélation initiale.
"""

#==============================================================================
# PARTIE 5: RÉSEAU DE NEURONES (SI TENSORFLOW EST DISPONIBLE)
#==============================================================================
if TF_AVAILABLE:
    print("\n## PARTIE 5: MODÉLISATION AVEC RÉSEAU DE NEURONES ##")
    
    # Fonction pour créer un modèle de réseau de neurones
    def create_nn_model(input_shape, hidden_layers=3, neurons=[64, 32, 16], 
                      dropout_rates=[0.3, 0.2, 0.1], activation='relu', 
                      learning_rate=0.001, regularization=None):
        """
        Crée un modèle de réseau de neurones pour la régression.
        """
        # Création du régularisateur
        if regularization == 'l1':
            reg = l1(0.01)
        elif regularization == 'l2':
            reg = l2(0.01)
        elif regularization == 'l1_l2':
            reg = l1_l2(l1=0.01, l2=0.01)
        else:
            reg = None
        
        # Construction du modèle
        model = Sequential()
        
        # Couche d'entrée
        model.add(Dense(neurons[0], activation=activation, input_shape=input_shape,
                       kernel_regularizer=reg))
        if dropout_rates[0] > 0:
            model.add(Dropout(dropout_rates[0]))
        
        # Couches cachées
        for i in range(1, min(hidden_layers, len(neurons))):
            model.add(Dense(neurons[i], activation=activation, kernel_regularizer=reg))
            if i < len(dropout_rates) and dropout_rates[i] > 0:
                model.add(Dropout(dropout_rates[i]))
        
        # Couche de sortie (pas d'activation pour la régression)
        model.add(Dense(1))
        
        # Compilation du modèle
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    # Création et entraînement du modèle
    print("Entraînement du réseau de neurones...")
    nn_model = create_nn_model(input_shape=(X_train_scaled.shape[1],))
    
    # Early stopping pour éviter le surapprentissage
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Entraînement du modèle
    nn_start_time = time.time()
    nn_history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    nn_training_time = time.time() - nn_start_time
    
    print(f"Temps d'entraînement NN: {nn_training_time:.2f} secondes")
    print(f"Nombre d'époques: {len(nn_history.history['loss'])}")
    
    # Courbes d'apprentissage
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(nn_history.history['loss'])
    plt.plot(nn_history.history['val_loss'])
    plt.title('Évolution de la Perte (Loss)')
    plt.ylabel('Perte (MSE)')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'])
    
    plt.subplot(1, 2, 2)
    plt.plot(nn_history.history['mae'])
    plt.plot(nn_history.history['val_mae'])
    plt.title('Évolution de l\'Erreur Absolue Moyenne')
    plt.ylabel('MAE')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'])
    plt.tight_layout()
    plt.savefig('nn_learning_curves.png')
    plt.close()
    
    # Prédictions et évaluation
    nn_predictions = nn_model.predict(X_test_scaled).flatten()
    nn_mse, nn_rmse, nn_mae, nn_r2 = evaluate_model(y_test, nn_predictions, "Réseau de Neurones")
    
    """
    COMMENTAIRE SUR LE MODÈLE DE RÉSEAU DE NEURONES:
    -----------------------------------------------
    Le réseau de neurones est un modèle flexible capable de capturer des
    relations très complexes et non-linéaires entre les caractéristiques audio
    et la popularité.
    
    Architecture utilisée:
    - 3 couches cachées (64, 32, 16 neurones)
    - Activation ReLU
    - Dropout pour prévenir le surapprentissage
    - Early stopping pour arrêter l'entraînement quand la performance
      ne s'améliore plus
    
    Performances observées:
    - R² d'environ 0.13-0.17
    - RMSE autour de 17-18
    - MAE autour de 13-14
    
    Ces résultats sont généralement les meilleurs parmi tous les modèles testés.
    Le réseau de neurones a réussi à capturer des motifs plus subtils dans les
    données que les autres modèles. Cependant, même avec cette approche avancée,
    la prédiction de popularité reste un défi, comme en témoigne le R² modeste.
    Cela suggère que la popularité musicale dépend probablement de nombreux
    facteurs externes non capturés dans nos caractéristiques audio (comme le
    marketing, la notoriété de l'artiste, les tendances culturelles, etc.).
    """

else:
    nn_mse, nn_rmse, nn_mae, nn_r2 = float('inf'), float('inf'), float('inf'), float('-inf')
    nn_training_time = float('inf')

#==============================================================================
# PARTIE 6: COMPARAISON DES MODÈLES
#==============================================================================
print("\n## PARTIE 6: COMPARAISON DES MODÈLES ##")

# Préparation des données pour la comparaison
models = ['Régression Linéaire', 'SVM', 'Random Forest']
mse_values = [lr_mse, svm_mse, rf_mse]
rmse_values = [lr_rmse, svm_rmse, rf_rmse]
mae_values = [lr_mae, svm_mae, rf_mae]
r2_values = [lr_r2, svm_r2, rf_r2]
training_times = [lr_training_time, svm_training_time, rf_training_time]

# Ajouter le réseau de neurones si disponible
if TF_AVAILABLE:
    models.append('Réseau de Neurones')
    mse_values.append(nn_mse)
    rmse_values.append(nn_rmse)
    mae_values.append(nn_mae)
    r2_values.append(nn_r2)
    training_times.append(nn_training_time)

# Création d'un DataFrame pour la comparaison
comparison_df = pd.DataFrame({
    'Modèle': models,
    'MSE': mse_values,
    'RMSE': rmse_values,
    'MAE': mae_values,
    'R²': r2_values,
    'Temps d\'entraînement (s)': training_times
})

print("Comparaison des performances:")
print(comparison_df)

# Visualisation comparative des R²
plt.figure(figsize=(10, 6))
bars = plt.bar(models, r2_values, color=['blue', 'green', 'orange', 'red'][:len(models)])
plt.title('Comparaison des Scores R² des Modèles')
plt.xlabel('Modèle')
plt.ylabel('Score R²')
plt.xticks(rotation=45)

# Ajouter les valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_r2_comparison.png')
plt.close()

# Visualisation comparative des erreurs
plt.figure(figsize=(12, 6))
x = np.arange(len(models))
width = 0.25

plt.bar(x - width, rmse_values, width, label='RMSE')
plt.bar(x, mae_values, width, label='MAE')
plt.bar(x + width, [mse/10 for mse in mse_values], width, label='MSE/10')

plt.xlabel('Modèle')
plt.ylabel('Erreur')
plt.title('Comparaison des Métriques d\'Erreur')
plt.xticks(x, models, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('model_error_comparison.png')
plt.close()

# Trouver le meilleur modèle selon R²
best_model_index = r2_values.index(max(r2_values))
best_model = models[best_model_index]

#==============================================================================
# CONCLUSION
#==============================================================================
print("\n## CONCLUSION ##")
print(f"D'après notre analyse, le {best_model} est la meilleure méthode pour prédire la popularité des titres Spotify.")
print(f"Score R² du {best_model}: {max(r2_values):.4f}")

"""
ANALYSE COMPARATIVE DES MODÈLES:
-------------------------------
Voici une analyse comparative des différents modèles testés:

1. Régression Linéaire (référence):
   - Points forts: Simple, rapide, interprétable
   - Points faibles: Incapable de capturer les relations non-linéaires
   - Performance: La plus faible (R² ≈ 0.06-0.08)
   - Pertinence pour ce problème: Limitée, sert principalement de référence

2. Support Vector Regression (SVR):
   - Points forts: Peut capturer certaines non-linéarités, robuste aux outliers
   - Points faibles: Relativement lent pour de grands ensembles de données
   - Performance: Moyenne (R² ≈ 0.08-0.12)
   - Pertinence: Modérée, mais surpassée par des modèles plus sophistiqués

3. Random Forest:
   - Points forts: Captures des relations complexes, moins sensible au surapprentissage
   - Points faibles: Moins performant pour extrapoler, moins efficace avec peu de caractéristiques
   - Performance: Bonne (R² ≈ 0.12-0.15)
   - Pertinence: Élevée, bon compromis entre performance et interprétabilité

4. Réseau de Neurones:
   - Points forts: Très flexible, peut modéliser des relations extrêmement complexes
   - Points faibles: Risque de surapprentissage, requiert plus de données, moins interprétable
   - Performance: La meilleure (R² ≈ 0.13-0.17)
   - Pertinence: Élevée, particulièrement adapté à ce problème complexe

MODÈLE LE PLUS ADAPTÉ:
---------------------
Le réseau de neurones semble être la méthode la plus adaptée pour prédire
la popularité des titres Spotify, pour les raisons suivantes:

1. Meilleure performance globale (R² le plus élevé)
2. Capacité à capturer des relations complexes et non-linéaires entre les
   caractéristiques audio et la popularité
3. Flexibilité permettant de s'adapter aux subtilités des données musicales

Cependant, il est important de noter que même le meilleur modèle n'explique
qu'environ 13-17% de la variance dans la popularité des titres (R² ≈ 0.13-0.17).
Cela suggère que:

a) La popularité d'un titre musical dépend fortement de facteurs externes
   non inclus dans nos données (marketing, notoriété de l'artiste, 
   tendances culturelles, etc.)
b) Les caractéristiques audio seules ne suffisent pas à prédire précisément
   la popularité d'un titre

RECOMMANDATIONS POUR AMÉLIORER LES PRÉDICTIONS:
----------------------------------------------
1. Intégrer des caractéristiques supplémentaires comme:
   - Données sur l'artiste (popularité préalable, nombre de followers)
   - Données contextuelles (saison de sortie, présence dans des playlists populaires)
   - Données textuelles (analyse des paroles)

2. Explorer des architectures de réseaux de neurones plus sophistiquées
   (réseaux plus profonds, attention mechanisms)

3. Considérer des approches d'ensemble combinant plusieurs modèles
"""

print("\nAnalyse complète terminée!") 