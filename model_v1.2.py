# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt  # Import nécessaire pour les plots


def main():
    # ---------------------------------------------------------
    # 0) CONFIGURATION DE TENSORFLOW POUR UTILISER LE GPU OPTIMISÉMENT
    # ---------------------------------------------------------
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(
                f"{len(gpus)} GPU(s) détecté(s) et configuré(s) pour une utilisation optimisée de la mémoire.")
        except RuntimeError as e:
            print(e)
    else:
        print("Aucun GPU détecté. L'entraînement se fera sur le CPU.")

    # ---------------------------
    # 1) CHARGEMENT DES DONNÉES
    # ---------------------------
    X = np.load("X_sequences.npy")  # Forme : (40470, 10, 15)
    y = np.load("y_targets.npy")    # Forme : (40470, 15)

    # Sélectionner uniquement la colonne 'close' pour la cible
    # Supposons que 'close' est la 5ème colonne (index 4)
    y = y[:, 4]  # Ajustez l'index si nécessaire
    y = y.reshape(-1, 1)  # Forme : (40470, 1)

    print("Forme de X :", X.shape)
    print("Forme de y :", y.shape)

    # -----------------------------
    # 2) DIVISION TRAIN / TEST
    # -----------------------------
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
    model = Sequential()
    # Couche LSTM bidirectionnelle avec return_sequences=True pour empiler une autre couche LSTM
    model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True),
              input_shape=(X_train.shape[1], X_train.shape[2])))
    # Normalisation par lots pour stabiliser l'apprentissage
    model.add(BatchNormalization())
    model.add(LSTM(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Régularisation pour éviter le surapprentissage
    model.add(Dense(1))  # Prédiction de 'close'

    # Configuration de l'optimizer avec un taux d'apprentissage ajustable
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()

    # ---------------------------------------------------
    # 4) ENTRAÎNEMENT DU MODÈLE
    # ---------------------------------------------------
    epochs = 100
    batch_size = 32

    # Callbacks pour arrêter tôt l'entraînement et réduire le taux d'apprentissage si nécessaire
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # ---------------------------------------------------
    # 5) ÉVALUATION DU MODÈLE
    # ---------------------------------------------------
    y_pred = model.predict(X_test)

    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Évaluation sur l'ensemble de test ---")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    print(f"R²: {r2:.4f}")

    # ---------------------------------------------------
    # 6) AFFICHAGE DES COURBES DE PERTE
    # ---------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Entraînement')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title("Courbes de Perte")
    plt.xlabel("Époque")
    plt.ylabel("Perte (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Sauvegarde du modèle
    model.save('eth_usdt_lstm_model.h5')
    print("\nModèle sauvegardé sous 'eth_usdt_lstm_model.h5'")


if __name__ == "__main__":
    main()
