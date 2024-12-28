# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt  # Import nécessaire pour les plots
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import TensorBoard
import datetime


def calculate_accuracy_within_threshold(y_true, y_pred, threshold=1.0):
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    within_threshold = percentage_errors <= threshold
    accuracy = np.mean(within_threshold) * 100
    return accuracy


def directional_accuracy(y_true, y_pred):
    direction_true = np.sign(y_true[1:] - y_true[:-1])
    direction_pred = np.sign(y_pred[1:] - y_pred[:-1])
    correct_directions = direction_true == direction_pred
    accuracy = np.mean(correct_directions) * 100
    return accuracy


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
    X = np.load("X_sequences.npy")
    y = np.load("y_targets.npy")

    # Sélectionner uniquement la colonne 'close' pour la cible

    y = y[:, 4]  # Ajustez l'index si nécessaire
    y = y.reshape(-1, 1)

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
    # 3) CRÉATION DES DATASETS AVEC tf.data
    # ---------------------------------------------------
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(
        buffer_size=10000).batch(64).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

    # ---------------------------------------------------
    # 4) DÉFINITION DU MODÈLE (LSTM MODIFIÉ POUR CUdNN)
    # ---------------------------------------------------
    model = Sequential()
    # Utiliser LSTM avec activation='tanh' pour l'accélération cuDNN
    model.add(Bidirectional(LSTM(
        64,
        activation='tanh',  # Activation modifiée
        recurrent_activation='sigmoid',  # Recurrent activation
        recurrent_dropout=0,  # Recurrent dropout
        unroll=False,  # Unroll
        use_bias=True,  # Use bias
        return_sequences=True
    ), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(LSTM(
        32,
        activation='tanh',  # Activation modifiée
        recurrent_activation='sigmoid',  # Recurrent activation
        recurrent_dropout=0,  # Recurrent dropout
        unroll=False,  # Unroll
        use_bias=True  # Use bias
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, dtype='float32'))  # Prédiction de 'close'

    # Configuration de l'optimizer avec un taux d'apprentissage ajustable
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()

    # ---------------------------------------------------
    # 5) ENTRAÎNEMENT DU MODÈLE
    # ---------------------------------------------------
    epochs = 100
    batch_size = 64  # Augmenté pour mieux utiliser le GPU

    # Créer un dossier pour les logs TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

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
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset,
        callbacks=[early_stopping, reduce_lr, tensorboard_callback],
        verbose=1
    )

    # ---------------------------------------------------
    # 6) ÉVALUATION DU MODÈLE
    # ---------------------------------------------------
    y_pred = model.predict(test_dataset)
    y_pred = y_pred.flatten()
    y_test_flat = y_test.flatten()

    # Calcul des métriques
    mse = mean_squared_error(y_test_flat, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred)
    mape = mean_absolute_percentage_error(
        y_test_flat, y_pred) * 100  # En pourcentage
    r2 = r2_score(y_test_flat, y_pred)
    directional_acc = directional_accuracy(y_test_flat, y_pred)
    accuracy_1 = calculate_accuracy_within_threshold(
        y_test_flat, y_pred, threshold=1.0)
    accuracy_5 = calculate_accuracy_within_threshold(
        y_test_flat, y_pred, threshold=5.0)

    print("\n--- Évaluation sur l'ensemble de test ---")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    print(f"Précision directionnelle: {directional_acc:.2f}%")
    print(f"Taux de réussite (±1%): {accuracy_1:.2f}%")
    print(f"Taux de réussite (±5%): {accuracy_5:.2f}%")

    # ---------------------------------------------------
    # 7) AFFICHAGE DES COURBES DE PERTE
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

    # ---------------------------------------------------
    # 8) OPTIONNEL : VISUALISATION DES PRÉDITIONS VS RÉELLES
    # ---------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_flat, y_pred, alpha=0.3)
    plt.plot([y_test_flat.min(), y_test_flat.max()], [
             y_test_flat.min(), y_test_flat.max()], 'r--')  # Ligne y=x
    plt.xlabel('Valeur Réelle')
    plt.ylabel('Valeur Prédite')
    plt.title('Prédictions vs Valeurs Réelles')
    plt.grid(True)
    plt.show()

    # Histogramme des Erreurs
    errors = y_test_flat - y_pred
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, edgecolor='k')
    plt.xlabel('Erreur de Prédiction (Valeur Réelle - Prédite)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Erreurs de Prédiction')
    plt.grid(True)
    plt.show()

    # Sauvegarde du modèle
    model.save('eth_usdt_lstm_model.h5')
    print("\nModèle sauvegardé sous 'eth_usdt_lstm_model.h5'")


if __name__ == "__main__":
    main()
