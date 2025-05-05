#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prédiction de la Popularité des Titres Spotify
==============================================
Ce script analyse et prédit la popularité des titres Spotify en utilisant
les méthodes de machine learning du cours:
- SVR (Support Vector Regression)
- Réseaux de neurones
- Régression linéaire (comme référence)

Auteur: Noa
"""

# Import des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# Essai d'import TensorFlow avec gestion d'erreur
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    # Pour la reproductibilité
    tf.random.set_seed(42)
except Exception as e:
    print(f"TensorFlow n'a pas pu être chargé: {e}")
    print("L'analyse continuera sans le modèle de réseau de neurones.")
    TF_AVAILABLE = False

# Configuration des visualisations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Pour la reproductibilité
np.random.seed(42)

print("# Prédiction de la Popularité des Titres Spotify #")
print("=" * 60)

#==============================================================================
# PARTIE 1: CHARGEMENT ET PRÉPARATION DES DONNÉES
#==============================================================================
print("\n## CHARGEMENT ET PRÉPARATION DES DONNÉES ##")

# Chargement des données
print("Chargement des données...")
try:
    high_popularity = pd.read_csv("Data/high_popularity_spotify_data.csv")
    low_popularity = pd.read_csv("Data/low_popularity_spotify_data.csv")
    
    # Fusion des datasets
    data = pd.concat([high_popularity, low_popularity], ignore_index=True)
    data = data.dropna()
    print(f"Dimensions des données: {data.shape}")
except FileNotFoundError:
    print("Erreur: Fichiers de données non trouvés.")
    exit()

# 1. Transformation de la date de sortie en format numérique
print("Transformation des dates...")
# Fonction pour convertir les dates au format YYYYMMDD
def convert_date_to_numeric(date_str):
    try:
        # Tenter de convertir automatiquement
        date_obj = pd.to_datetime(date_str)
        return int(date_obj.strftime('%Y%m%d'))
    except:
        # Si échec, vérifier si c'est juste une année (YYYY)
        if isinstance(date_str, str) and len(date_str) == 4 and date_str.isdigit():
            # Pour une année seule, utiliser le 1er janvier
            return int(date_str + '0101')
        else:
            # En cas d'échec total, utiliser une valeur par défaut
            return 19700101  # 1er janvier 1970 comme date par défaut

# Appliquer la conversion sur la colonne de dates
data['release_date_numeric'] = data['track_album_release_date'].apply(convert_date_to_numeric)

# Calculer l'âge approximatif du titre
current_date = pd.Timestamp.now()
def calculate_age_days(date_numeric):
    try:
        # Convertir le nombre en chaîne puis en date
        date_str = str(date_numeric)
        if len(date_str) == 8:  # format YYYYMMDD
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:])
            track_date = pd.Timestamp(year=year, month=month, day=day)
            return (current_date - track_date).days
        else:
            # Si le format n'est pas celui attendu, utiliser l'année seulement
            year = int(date_str[:4])
            track_date = pd.Timestamp(year=year, month=1, day=1)
            return (current_date - track_date).days
    except:
        # En cas d'erreur, retourner une valeur par défaut (20 ans)
        return 365 * 20

data['track_age_days'] = data['release_date_numeric'].apply(calculate_age_days)

# 2. One-hot encoding des genres et sous-genres
print("Application d'encodage pour les genres...")
# Remplacer le one-hot encoding par label encoding pour playlist_genre
genre_encoder = LabelEncoder()
data['genre_encoded'] = genre_encoder.fit_transform(data['playlist_genre'])
# On garde le one-hot encoding seulement pour les sous-genres (moins nombreux)
subgenre_dummies = pd.get_dummies(data['playlist_subgenre'], prefix='subgenre')

# 3. Création d'ID numériques pour les artistes
print("Génération des IDs numériques pour les artistes...")
# Dictionnaire associant chaque artiste à un ID unique
artist_to_id = {artist: idx for idx, artist in enumerate(data['track_artist'].unique())}
data['artist_id'] = data['track_artist'].map(artist_to_id)

# 4. Statistiques par artiste
print("Calcul des statistiques par artiste...")
# Popularité moyenne par artiste
artist_avg_popularity = data.groupby('track_artist')['track_popularity'].mean().reset_index()
artist_avg_popularity.columns = ['track_artist', 'artist_avg_popularity']
# Nombre de titres par artiste
artist_track_count = data.groupby('track_artist').size().reset_index()
artist_track_count.columns = ['track_artist', 'artist_track_count']

# Fusion des statistiques dans le dataframe principal
data = data.merge(artist_avg_popularity, on='track_artist', how='left')
data = data.merge(artist_track_count, on='track_artist', how='left')

# 5. Identifiants uniques encodés numériquement
print("Encodage des identifiants uniques...")
# Encodage des IDs par hachage pour réduire la dimensionnalité
# (alternative à one-hot qui créerait trop de colonnes)
data['track_id_hash'] = data['track_id'].apply(lambda x: hash(x) % 10000)  # Modulo pour limiter la taille
data['album_id_hash'] = data['track_album_id'].apply(lambda x: hash(x) % 10000)
data['playlist_id_hash'] = data['playlist_id'].apply(lambda x: hash(x) % 10000)

# 6. Mise à jour des caractéristiques pour le modèle
print("Mise à jour de la liste des caractéristiques...")
numerical_features = [
    # Caractéristiques audio originales
    'energy', 'tempo', 'danceability', 'loudness', 'liveness', 
    'valence', 'speechiness', 'instrumentalness', 'acousticness',
    'duration_ms',
    
    # Nouvelles caractéristiques numériques
    'release_date_numeric', 'track_age_days',
    'artist_id', 'artist_avg_popularity', 'artist_track_count',
    'track_id_hash', 'album_id_hash', 'playlist_id_hash',
    'genre_encoded'  # Genre encodé comme simple catégorie numérique
]

# Ajout des colonnes one-hot pour les sous-genres à la liste des caractéristiques
numerical_features.extend(subgenre_dummies.columns)

# 7. Fusion des colonnes one-hot avec le dataframe principal
data = pd.concat([data, subgenre_dummies], axis=1)

# 8. Vérification et gestion des valeurs manquantes
print("Vérification et gestion des valeurs manquantes...")
# Créer un dataframe temporaire avec toutes les caractéristiques pour vérifier les NaN
temp_X = data[numerical_features]
na_counts = temp_X.isna().sum()
columns_with_na = na_counts[na_counts > 0]
if len(columns_with_na) > 0:
    print(f"Colonnes avec valeurs manquantes:")
    print(columns_with_na)
    
    # Remplacer les valeurs manquantes par la médiane pour les colonnes numériques
    for col in columns_with_na.index:
        median_value = data[col].median()
        data[col] = data[col].fillna(median_value)
        print(f"  - {col}: {na_counts[col]} valeurs manquantes remplacées par la médiane ({median_value})")

# Vérifier si la variable cible a des valeurs manquantes
target_na_count = data['track_popularity'].isna().sum()
if target_na_count > 0:
    print(f"Variable cible: {target_na_count} valeurs manquantes détectées")
    # Deux options: soit les supprimer, soit les remplacer par la médiane
    # Option 1: supprimer les lignes avec popularité manquante
    data = data.dropna(subset=['track_popularity'])
    print(f"  - Lignes avec popularité manquante supprimées. Dimensions: {data.shape}")
    # Option 2 (alternative): remplacer par la médiane
    # median_popularity = data['track_popularity'].median()
    # data['track_popularity'] = data['track_popularity'].fillna(median_popularity)
    # print(f"  - Valeurs de popularité manquantes remplacées par la médiane ({median_popularity})")

# Préparation des données finales
X = data[numerical_features]
y = data['track_popularity']

# Division en ensembles d'entraînement et de test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisation (importante pour SVM selon le cours)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dimensions des données d'entraînement: {X_train.shape}")
print(f"Dimensions des données de test: {X_test.shape}")

#==============================================================================
# FONCTION D'ÉVALUATION
#==============================================================================
def evaluate_model(y_true, y_pred, model_name):
    """
    Évalue les performances d'un modèle et génère des visualisations.
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
# PARTIE 2: RÉGRESSION LINÉAIRE (BASELINE)
#==============================================================================
print("\n## MODÉLISATION AVEC RÉGRESSION LINÉAIRE (BASELINE) ##")

# Entraînement du modèle
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Prédictions et évaluation
lr_predictions = lr_model.predict(X_test_scaled)
lr_metrics = evaluate_model(y_test, lr_predictions, "Régression Linéaire")

"""
COMMENTAIRE SUR LA RÉGRESSION LINÉAIRE:
--------------------------------------
La régression linéaire sert de modèle de référence (baseline).
Performances observées:
- R² d'environ 0.95
- RMSE autour de 4.4
- MAE autour de 2.2

Ces performances sont remarquablement bonnes pour un modèle linéaire simple.
La prise en compte des variables qualitatives (artistes, genres, dates) a 
considérablement amélioré les résultats par rapport à l'utilisation des seules 
caractéristiques audio. Cela confirme l'importance du contexte et des métadonnées
dans la prédiction de la popularité des titres.
"""

#==============================================================================
# PARTIE 3: SUPPORT VECTOR REGRESSION (SVR)
#==============================================================================
print("\n## MODÉLISATION AVEC SVR ##")

# Création du pipeline SVR avec les paramètres corrects
print("Recherche des meilleurs hyperparamètres SVR...")
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.1, 0.2]  # Tolérance pour la marge SVR
}

# Recherche par grille avec validation croisée (k=3)
grid_search = GridSearchCV(SVR(), param_grid, cv=3, 
                          scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Meilleurs paramètres
best_svr = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Meilleurs paramètres SVR: {best_params}")

# Prédictions et évaluation
svr_predictions = best_svr.predict(X_test_scaled)
svr_metrics = evaluate_model(y_test, svr_predictions, "SVR")

"""
COMMENTAIRE SUR LE MODÈLE SVR:
-----------------------------
Le SVR (Support Vector Regression) offre une légère amélioration par rapport 
à la régression linéaire.

Les meilleurs hyperparamètres trouvés sont:
- Noyau: linéaire
- C: 100 (peu de régularisation, modèle plus flexible)
- Gamma: 'scale' (adaptation automatique à l'échelle des données)
- Epsilon: 0.1 (marge d'erreur tolérée assez fine)

Performances observées:
- R² d'environ 0.953
- RMSE autour de 4.36
- MAE autour de 1.97 (meilleur que la régression linéaire)

Le SVR se montre particulièrement efficace pour minimiser l'erreur absolue moyenne (MAE),
ce qui suggère une meilleure robustesse aux valeurs aberrantes. La différence avec la 
régression linéaire reste cependant modeste, indiquant que les relations entre 
les variables sont relativement bien capturées par un modèle linéaire quand les 
variables qualitatives sont correctement encodées.
"""

#==============================================================================
# PARTIE 4: RÉSEAU DE NEURONES (SI DISPONIBLE)
#==============================================================================
if TF_AVAILABLE:
    print("\n## MODÉLISATION AVEC RÉSEAU DE NEURONES ##")
    
    # Pour le réseau de neurones, architecture plus large pour gérer les nouvelles variables
    def create_nn_model(input_shape):
        model = Sequential()
        
        # Couche d'entrée plus large
        model.add(Dense(128, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.4))
        
        # Couches cachées
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        
        # Couche de sortie
        model.add(Dense(1))
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    # Création et entraînement du modèle
    nn_model = create_nn_model(input_shape=(X_train_scaled.shape[1],))
    
    # Early stopping pour éviter le surapprentissage
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    # Entraînement avec validation sur 20% des données d'entraînement
    nn_history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=100,  # Maximum d'époques
        batch_size=32,  # Taille du lot
        validation_split=0.2,  # 20% pour validation
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Prédictions et évaluation
    nn_predictions = nn_model.predict(X_test_scaled).flatten()
    nn_metrics = evaluate_model(y_test, nn_predictions, "Réseau de Neurones")
    
    # Visualisation des courbes d'apprentissage
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(nn_history.history['loss'])
    plt.plot(nn_history.history['val_loss'])
    plt.title('Évolution de la Perte (MSE)')
    plt.ylabel('Perte')
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

    """
    COMMENTAIRE SUR LE MODÈLE DE RÉSEAU DE NEURONES:
    -----------------------------------------------
    Le réseau de neurones, malgré sa complexité et sa capacité théorique à capturer
    des relations non-linéaires, présente des performances légèrement inférieures aux
    autres modèles sur ce jeu de données enrichi.

    Architecture utilisée:
    - 3 couches cachées (128, 64, 32 neurones) avec dropouts
    - Activation ReLU
    - Early stopping pour éviter le surapprentissage

    Performances observées:
    - R² d'environ 0.94 (légèrement inférieur au SVR et à la régression linéaire)
    - RMSE autour de 4.9
    - MAE autour de 3.1

    Le modèle a convergé relativement rapidement (environ 86 époques), mais semble
    légèrement surapprentir malgré les techniques de régularisation employées. Cela
    pourrait s'expliquer par le fait que la relation entre les variables explicatives
    et la popularité est déjà bien capturée par des modèles plus simples, une fois
    que les variables qualitatives ont été correctement transformées et intégrées.
    """

#==============================================================================
# PARTIE 5: COMPARAISON DES MODÈLES
#==============================================================================
print("\n## COMPARAISON DES MODÈLES ##")

# Préparation des données pour la comparaison
models = ['Régression Linéaire', 'SVR']
r2_values = [lr_metrics[3], svr_metrics[3]]
rmse_values = [lr_metrics[1], svr_metrics[1]]

# Ajout du réseau de neurones si disponible
if TF_AVAILABLE:
    models.append('Réseau de Neurones')
    r2_values.append(nn_metrics[3])
    rmse_values.append(nn_metrics[1])

# Visualisation comparative des R²
plt.figure(figsize=(10, 6))
plt.bar(models, r2_values, color=['blue', 'green', 'red'][:len(models)])
plt.title('Comparaison des Scores R² des Modèles')
plt.xlabel('Modèle')
plt.ylabel('Score R²')
plt.ylim(0, max(r2_values) * 1.2)  # Ajustement de l'échelle
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_r2_comparison.png')
plt.close()

# Visualisation comparative des RMSE
plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['blue', 'green', 'red'][:len(models)])
plt.title('Comparaison des RMSE des Modèles')
plt.xlabel('Modèle')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_rmse_comparison.png')
plt.close()

# Identification du meilleur modèle
best_model_index = r2_values.index(max(r2_values))
best_model = models[best_model_index]

#==============================================================================
# PARTIE 6: EXPLICABILITÉ DES MODÈLES (MÉTHODES DU COURS)
#==============================================================================
print("\n## EXPLICABILITÉ DES MODÈLES (MÉTHODES DU COURS) ##")

# Import des bibliothèques pour l'explicabilité
try:
    import shap
    from alibi.explainers import ALE
    import lime
    import lime.lime_tabular
    EXPLAIN_LIBS_AVAILABLE = True
except ImportError:
    print("Certaines bibliothèques d'explicabilité ne sont pas disponibles.")
    print("Installation recommandée: pip install shap alibi lime")
    EXPLAIN_LIBS_AVAILABLE = False

from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay

# Créer un DataFrame pour l'interprétation
X_interpret = pd.DataFrame(X_test_scaled, columns=numerical_features + list(subgenre_dummies.columns))

# Sélectionner le meilleur modèle pour l'explicabilité (utilisons SVR pour cet exemple)
model_to_explain = best_svr

print("\n1. IMPORTANCE GLOBALE DES CARACTÉRISTIQUES")
print("-----------------------------------------")

# Permutation Feature Importance - RÉDUIT à 5 répétitions et nombre limité de caractéristiques
print("Calcul de l'importance des caractéristiques par permutation...")
# Sélectionner les 30 caractéristiques les plus importantes selon le score absolu en régression linéaire
lr_feature_importance = np.abs(lr_model.coef_)
top_features_idx = np.argsort(lr_feature_importance)[-30:]  # Top 30 features
X_test_reduced = X_test_scaled[:, top_features_idx]

result = permutation_importance(model_to_explain, X_test_reduced, y_test,
                              n_repeats=5,  # Réduit de 10 à 5
                              random_state=42,
                              scoring="neg_mean_squared_error")

# Adapter les noms des caractéristiques au sous-ensemble
reduced_feature_names = np.array(list(X_interpret.columns))[top_features_idx]

# Afficher les résultats
importance_df = pd.DataFrame({
    'Feature': reduced_feature_names,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)
print("\nImportance des caractéristiques par permutation (TOP 10):")
print(importance_df.head(10))

# Limiter à 3 caractéristiques les plus importantes pour les analyses visuelles
top_numeric_features = importance_df.head(3)['Feature'].tolist()
top_features_idx = [list(reduced_feature_names).index(feat) for feat in top_numeric_features]

print("\n2. ANALYSE DES EFFETS PARTIELS (PDP) - VERSION ALLÉGÉE")
print("--------------------------------------------------")
print("Génération des graphiques PDP pour les 3 caractéristiques les plus importantes...")

if len(top_numeric_features) > 0:
    # Tracé PDP pour les 3 caractéristiques les plus importantes
    for feature in top_numeric_features:
        feature_idx = list(X_interpret.columns).index(feature)
        plt.figure(figsize=(10, 6))
        
        # Utiliser un échantillon réduit pour accélérer les calculs
        sample_size = min(200, len(X_test_scaled))
        sample_indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
        X_sample = X_test_scaled[sample_indices]
        
        # Calculer le PDP
        PartialDependenceDisplay.from_estimator(
            model_to_explain, 
            X_sample,
            [feature_idx],
            kind="average",
            grid_resolution=30,  # Réduit de 50 à 30
            random_state=42
        )
        plt.title(f'PDP pour {feature}')
        plt.tight_layout()
        plt.savefig(f'pdp_{feature}.png')
        plt.close()
        print(f"  - Graphique PDP pour {feature} généré")

print("\n3. EXPLICABILITÉ LOCALE - VERSION ALLÉGÉE")
print("--------------------------------------")
# Limiter LIME à un seul exemple représentatif
if EXPLAIN_LIBS_AVAILABLE:
    try:
        # Initialiser LIME avec un échantillon réduit
        sample_for_lime = X_train_scaled[:1000] if len(X_train_scaled) > 1000 else X_train_scaled
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            sample_for_lime, 
            feature_names=list(X_interpret.columns),
            mode='regression',
            verbose=False
        )
        
        # Sélectionner l'exemple médian uniquement
        preds = model_to_explain.predict(X_test_scaled)
        sorted_idx = np.argsort(preds)
        median_idx = sorted_idx[len(sorted_idx) // 2]  # Instance médiane
        
        # Générer et afficher l'explication LIME
        exp = lime_explainer.explain_instance(
            X_test_scaled[median_idx], model_to_explain.predict, num_features=5
        )
        
        # Enregistrer l'explication
        fig = exp.as_pyplot_figure()
        plt.title(f"Explication LIME (Valeur réelle: {y_test.iloc[median_idx]:.2f}, Prédiction: {preds[median_idx]:.2f})")
        plt.tight_layout()
        plt.savefig('lime_explanation.png')
        plt.close()
        print("  - Explication LIME générée pour une instance représentative")
    except Exception as e:
        print(f"Erreur lors de la génération de l'explication LIME: {str(e)}")

# SHAP - limité à un petit échantillon de 10 instances
if EXPLAIN_LIBS_AVAILABLE:
    try:
        print("\n4. ANALYSE SHAP SIMPLIFIÉE")
        print("------------------------")
        print("Calcul des valeurs SHAP sur un échantillon très limité...")
        
        # Échantillon très réduit pour SHAP
        sample_size = 10  # Réduit de 50 à 10
        sample_idx = np.random.choice(len(X_test_scaled), sample_size, replace=False)
        
        # Limite également l'échantillon de référence
        reference_sample = shap.sample(X_train_scaled, 50)  # Réduit de 100 à 50
        
        # Initialiser l'explainer SHAP
        explainer = shap.KernelExplainer(model_to_explain.predict, reference_sample)
        
        # Calculer les valeurs SHAP seulement pour les caractéristiques les plus importantes
        X_test_important = X_test_scaled[sample_idx][:, top_features_idx]
        shap_values = explainer.shap_values(X_test_important)
        
        # SHAP summary plot - uniquement sur les caractéristiques importantes
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_important, 
                        feature_names=reduced_feature_names, 
                        plot_type="bar", 
                        show=False)
        plt.title('SHAP Summary Plot (Caractéristiques importantes)')
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
        print("  - Graphique SHAP Summary généré (version allégée)")
        
    except Exception as e:
        print(f"Erreur lors de la génération des analyses SHAP: {str(e)}")

print("\n5. CONCLUSION DE L'ANALYSE D'EXPLICABILITÉ")
print("----------------------------------------")
print("Points clés identifiés par les méthodes d'explicabilité:")
# Identifier les caractéristiques les plus importantes
top_features = importance_df.head(5)['Feature'].tolist()
print(f"  - Les caractéristiques les plus influentes sur la prédiction sont: {', '.join(top_features)}")

# Générer un rapport détaillé d'explicabilité
print("\nRapport détaillé d'explicabilité sauvegardé dans 'explicability_report.txt'")
with open('explicability_report.txt', 'w') as f:
    f.write("RAPPORT D'EXPLICABILITÉ - PRÉDICTION DE LA POPULARITÉ DES TITRES SPOTIFY\n")
    f.write("==================================================================\n\n")
    f.write("1. IMPORTANCE GLOBALE DES CARACTÉRISTIQUES\n")
    f.write("-----------------------------------------\n")
    f.write(importance_df.to_string())
    f.write("\n\nLes caractéristiques les plus importantes sont celles qui, lorsqu'elles sont permutées, augmentent le plus l'erreur du modèle.\n")
    f.write(f"Les 5 caractéristiques les plus importantes sont: {', '.join(top_features)}\n\n")
    
    f.write("2. INTERPRÉTATION DES CARACTÉRISTIQUES PRINCIPALES\n")
    f.write("------------------------------------------------\n")
    for feature in top_features:
        f.write(f"\n{feature}:\n")
        f.write(f"  - Cette caractéristique est l'une des plus importantes pour prédire la popularité des titres.\n")
    
    f.write("\n3. COMPARAISON DES MODÈLES\n")
    f.write("-------------------------\n")
    f.write("Modèle SVR:\n")
    f.write(f"  - Performances: R²={svr_metrics[3]:.4f}, RMSE={svr_metrics[1]:.4f}, MAE={svr_metrics[2]:.4f}\n")
    f.write("  - Avantages: Capture bien les relations non-linéaires entre les caractéristiques audio et la popularité\n")
    
    f.write("\nModèle NN:\n")
    f.write(f"  - Performances: R²={nn_metrics[3]:.4f}, RMSE={nn_metrics[1]:.4f}, MAE={nn_metrics[2]:.4f}\n")
    f.write("  - Avantages: Bonne capacité à modéliser des interactions complexes entre variables\n")
    
    f.write("\nModèle LR:\n")
    f.write(f"  - Performances: R²={lr_metrics[3]:.4f}, RMSE={lr_metrics[1]:.4f}, MAE={lr_metrics[2]:.4f}\n")
    f.write("  - Avantages: Plus simple et interprétable\n")

print("Analyse d'explicabilité terminée.")

"""
COMMENTAIRE SUR LES MÉTHODES D'EXPLICABILITÉ:
-----------------------------------------
Les différentes méthodes d'explicabilité permettent de comprendre le fonctionnement 
du modèle à différents niveaux:

1. L'importance des caractéristiques par permutation révèle les variables qui 
   influencent le plus la prédiction de popularité.

2. Les PDP montrent comment chaque caractéristique affecte la prédiction en moyenne,
   tandis que les courbes ICE révèlent la variation individuelle de ces effets.

3. LIME fournit des explications locales en approximant le comportement du modèle 
   autour d'instances spécifiques.

4. SHAP offre une analyse unifiée, montrant la contribution moyenne et individuelle 
   de chaque caractéristique à la prédiction finale.

L'utilisation conjointe de ces méthodes, comme enseigné dans le cours, permet une 
compréhension approfondie du comportement des modèles, même les plus complexes
comme les SVR et réseaux de neurones.
"""

#==============================================================================
# CONCLUSION
#==============================================================================
print("\n## CONCLUSION ##")
print(f"D'après notre analyse, le modèle {best_model} est la meilleure méthode pour prédire la popularité des titres Spotify.")
print(f"Score R² du {best_model}: {max(r2_values):.4f}")
print("\nRemarques importantes:")
print("- Même le meilleur modèle n'explique qu'une faible partie de la variance dans la popularité")
print("- Les caractéristiques audio seules ne suffisent pas à prédire précisément la popularité")
print("- L'ajout de données sur l'artiste, le marketing, etc. pourrait améliorer les prédictions")

"""
ANALYSE COMPARATIVE ET CONCLUSION:
--------------------------------
Voici une analyse comparative des différents modèles testés avec le jeu de données enrichi
(incluant les variables qualitatives transformées):

1. Régression Linéaire:
   - Points forts: Simple, interprétable, rapide
   - Performance: Très bonne (R² ≈ 0.95)
   - Pertinence: Surprenamment efficace avec des données bien préparées

2. Support Vector Regression (SVR):
   - Points forts: Robuste aux outliers, bon équilibre complexité/performance
   - Performance: La meilleure (R² ≈ 0.953)
   - Pertinence: Modèle optimal, surtout avec un noyau linéaire

3. Réseau de Neurones:
   - Points forts: Grande flexibilité, adaptatif
   - Points faibles: Plus complexe à paramétrer, risque de surapprentissage
   - Performance: Légèrement inférieure (R² ≈ 0.94)
   - Pertinence: Potentiellement sous-exploité pour ce problème

CONCLUSION PRINCIPALE:
--------------------
La transformation et l'intégration des variables qualitatives (artistes, genres, dates)
ont considérablement amélioré les performances de tous les modèles par rapport à l'utilisation
des seules caractéristiques audio. Les R² sont passés d'environ 0.15 à plus de 0.94,
démontrant que le contexte non-acoustique est déterminant dans la popularité musicale.

Le SVR avec noyau linéaire ressort comme la méthode la plus efficace, mais la simplicité
de la régression linéaire la rend très attractive pour ce problème. Le réseau de neurones,
malgré sa sophistication, n'apporte pas de gain significatif, suggérant que la relation
entre les variables et la popularité est relativement bien capturée par des modèles linéaires
quand les données sont correctement préparées.

Cette analyse confirme l'hypothèse initiale: la popularité d'un titre dépend davantage
de facteurs contextuels (artiste, genre, date) que des caractéristiques audio intrinsèques.
Ces résultats ont des implications importantes pour l'industrie musicale, suggérant que
la stratégie marketing et le choix d'artiste peuvent être plus déterminants que les
qualités sonores pour le succès commercial d'un titre.
"""

print("\nAnalyse complète terminée!")