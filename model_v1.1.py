# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------
# 1) CHARGEMENT DES DONNÉES
# ---------------------------
# On suppose que tes fichiers X_sequences.npy et y_targets.npy
# se trouvent dans le même dossier que ce script.
X = np.load("X_sequences.npy")
y = np.load("y_targets.npy")

print("Forme de X :", X.shape)  # (N, 10, nb_features) si tu as 10 bougies en entrée
print("Forme de y :", y.shape)  # (N, nb_features) ou (N,) selon comment tu as défini la cible

# -----------------------------
# 2) DIVISION TRAIN / TEST
# -----------------------------
# Pour des données séquentielles, il est souvent recommandé
# de ne PAS mélanger le passé et le futur en shuffle=True.
# Ici, on met shuffle=False pour respecter l'ordre temporel.
test_size_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=test_size_ratio, 
    shuffle=False
)

print("Train shape X:", X_train.shape, "y:", y_train.shape)
print("Test shape  X:", X_test.shape,  "y:", y_test.shape)

# ---------------------------------------------------
# 3) DÉFINITION DU MODÈLE LSTM
# ---------------------------------------------------
# Ici on propose un LSTM simple. Tu peux rajouter
# des couches LSTM et plus de neurones si tu as les ressources.

model = Sequential()

# Première couche LSTM
# input_shape = (timesteps, nb_features) 
# timesteps = 10 ici, mais on peut simplement mettre (X_train.shape[1], X_train.shape[2])
model.add(LSTM(64, activation='relu', return_sequences=False, 
               input_shape=(X_train.shape[1], X_train.shape[2])))

# Optionnel: une couche Dense cachée ou un Dropout
model.add(Dropout(0.2))

# Couche de sortie
# Si y est 1D (par exemple prédire un seul prix de clôture), Dense(1)
# Si y est multidimensionnel (par exemple (N, 6) pour OHLC + volume), Dense(y.shape[1])
# Exemple pour un seul prix (Close) :
if len(y_train.shape) == 1:
    # y est un vecteur 1D
    model.add(Dense(1))
else:
    # y est un vecteur multidimensionnel, on prédit la même dimension
    model.add(Dense(y_train.shape[1]))  

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# ---------------------------------------------------
# 4) ENTRAÎNEMENT DU MODÈLE
# ---------------------------------------------------
epochs = 50
batch_size = 32

# Optionnel: callback d'EarlyStopping si tu souhaites arrêter tôt
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5, 
    restore_best_weights=True
)

history = model.fit(
    X_train, 
    y_train, 
    epochs=epochs, 
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],  # ou [] si tu ne veux pas EarlyStopping
    verbose=1
)

# ---------------------------------------------------
# 5) ÉVALUATION DU MODÈLE
# ---------------------------------------------------
# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Pour le R², il faut s'assurer que y_test et y_pred
# ont des dimensions compatibles (par ex. (N,1)).
# Si c'est (N,) vs (N,1), on peut faire un reshape.
if len(y_test.shape) == 1 and len(y_pred.shape) == 2:
    y_pred = y_pred.reshape(-1)

r2 = r2_score(y_test, y_pred)
model.save('eth_usdt_lstm_model.h5')

print("\n--- Évaluation sur l'ensemble de test ---")    
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("MAE:", mae)
print("R²:", r2)

# -----------------------
# 6) PREDICTION FUTURE ?
# -----------------------
# Ci-dessous, un exemple si tu veux prédire la bougie suivante
# en "glissant" sur la dernière séquence du test. 
# Cela dépend de comment tu formules ta prédiction future.
# 
# last_sequence = X_test[-1]  # la dernière séquence du test
# last_sequence = np.expand_dims(last_sequence, axis=0)  # (1, 10, nb_features)
# future_pred = model.predict(last_sequence)
# print("Prédiction future :", future_pred)
#
# Ensuite, tu pourrais 'concaténer' la prédiction à la séquence
# et continuer de prédire pour plusieurs pas de temps (similaire
# au code d'exemple que tu montrais).

