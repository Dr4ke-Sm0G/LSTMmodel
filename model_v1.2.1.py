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


def detect_overfitting(history):
    """
    Analyse des courbes de perte pour détecter l'overfitting.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Affichage des courbes de perte
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Perte Entraînement')
    plt.plot(val_loss, label='Perte Validation')
    plt.title("Courbes de Perte (Overfitting Detection)")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Vérification d'overfitting
    if val_loss[-1] > min(val_loss) * 1.1:  # Si la perte de validation augmente de plus de 10%
        print("⚠️ Indicateur potentiel d'overfitting : La perte de validation augmente.")
    else:
        print("✔️ Pas de signe évident d'overfitting.")


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
    model.add(Bidirectional(LSTM(
        64,
        activation='tanh',
        recurrent_activation='sigmoid',
        return_sequences=True
    ), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(LSTM(
        32,
        activation='tanh',
        recurrent_activation='sigmoid'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, dtype='float32'))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()

    # ---------------------------------------------------
    # 5) ENTRAÎNEMENT DU MODÈLE
    # ---------------------------------------------------
    epochs = 100
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

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

    # Détecter l'overfitting
    detect_overfitting(history)

    # ---------------------------------------------------
    # 6) ÉVALUATION DU MODÈLE
    # ---------------------------------------------------
    y_pred = model.predict(test_dataset)
    y_pred = y_pred.flatten()
    y_test_flat = y_test.flatten()

    mse = mean_squared_error(y_test_flat, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred)
    mape = mean_absolute_percentage_error(y_test_flat, y_pred) * 100
    r2 = r2_score(y_test_flat, y_pred)
    directional_acc = directional_accuracy(y_test_flat, y_pred)

    print("\n--- Évaluation sur l'ensemble de test ---")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    print(f"Précision directionnelle: {directional_acc:.2f}%")

    # Courbe des pertes
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Perte Entraînement')
    plt.plot(history.history['val_loss'], label='Perte Validation')
    plt.title("Courbes de Perte")
    plt.xlabel("Époques")
    plt.ylabel("Perte")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
