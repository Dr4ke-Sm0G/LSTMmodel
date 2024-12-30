import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

def predict_and_compare(model_path, input_file, scaler_file, output_file, targets_file):
    """
    Charge le modèle, effectue des prédictions, dénormalise les résultats et compare avec les vraies valeurs.
    
    Args:
    - model_path (str): Chemin vers le modèle LSTM sauvegardé.
    - input_file (str): Chemin vers le fichier numpy (.npy) contenant les séquences des 10 bougies.
    - scaler_file (str): Chemin vers le scaler sauvegardé pour dénormaliser les prédictions.
    - output_file (str): Chemin vers le fichier de sortie CSV contenant les prédictions dénormalisées.
    - targets_file (str): Chemin vers le fichier numpy (.npy) contenant les vraies valeurs (y_targets).
    """
    # Charger le modèle
    model = load_model(model_path)
    print("Modèle chargé avec succès.")

    # Charger les séquences d'entrée
    x_sequences = np.load(input_file)
    print(f"Fichier de séquences chargé : {x_sequences.shape}")

    # Charger les vraies valeurs
    y_targets = np.load(targets_file)
    print(f"Fichier des vraies valeurs chargé : {y_targets.shape}")

    # Charger le scaler pour dénormalisation
    scaler = joblib.load(scaler_file)   
    print("Scaler chargé avec succès.")

    # Effectuer les prédictions
    predicted_closes = model.predict(x_sequences)
    predicted_closes = predicted_closes.flatten()  # Aplatir les prédictions pour un format simple
    print("Prédictions effectuées.")

    # Dénormalisation des prédictions
    dummy_data = np.zeros((len(predicted_closes), scaler.data_max_.shape[0]))
    dummy_data[:, 0] = predicted_closes  # Assurez-vous que `close` est bien la 1ère colonne
    denormalized_closes = scaler.inverse_transform(dummy_data)[:, 0]  # Extraire uniquement la colonne `close`

    # Vérification rapide
    print("Valeurs normalisées des prédictions :", predicted_closes[:5])
    print("Valeurs dénormalisées des prédictions :", denormalized_closes[:5])

    # Dénormalisation des vraies valeurs
    dummy_data_targets = np.zeros((len(y_targets), scaler.data_max_.shape[0]))
    dummy_data_targets[:, 0] = y_targets  # Assurez-vous que `close` est bien la 1ère colonne
    denormalized_targets = scaler.inverse_transform(dummy_data_targets)[:, 0]

    # Vérification rapide
    print("Valeurs normalisées des vraies cibles :", y_targets[:5])
    print("Valeurs dénormalisées des vraies cibles :", denormalized_targets[:5])

    # Comparaison des écarts
    print("Écarts entre prédictions et valeurs réelles (dénormalisées) :", 
        np.abs(denormalized_closes[:5] - denormalized_targets[:5]))


    # Calculer la différence en pourcentage
    percentage_diff = ((denormalized_closes - denormalized_targets) / denormalized_targets) * 100

    # Créer un DataFrame pour les prédictions, les vraies valeurs et la différence
    predictions_df = pd.DataFrame({
        "sequence_index": range(len(denormalized_closes)),
        "predicted_close": denormalized_closes,
        "real_close": denormalized_targets,
        "percentage_diff": percentage_diff
    })

    # Sauvegarder les résultats dans un fichier CSV
    predictions_df.to_csv(output_file, index=False)
    print(f"Prédictions sauvegardées dans le fichier : {output_file}")

    # Graphique avec matplotlib
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_df["sequence_index"], predictions_df["real_close"], label="Valeurs Réelles", linewidth=2)
    plt.plot(predictions_df["sequence_index"], predictions_df["predicted_close"], label="Prédictions", linestyle="--")
    plt.xlabel("Index de Séquence")
    plt.ylabel("Prix")
    plt.title("Prédictions vs Valeurs Réelles")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Chemins des fichiers
    MODEL_PATH = "eth_usdt_lstm_model.h5"
    INPUT_FILE = "Test/test_X_sequences.npy"
    SCALER_FILE = "Test/scaler.pkl"
    OUTPUT_FILE = "Test/predicted_closes_comparison.csv"
    TARGETS_FILE = "Test/test_y_targets.npy"

    # Exécution du script
    predict_and_compare(MODEL_PATH, INPUT_FILE, SCALER_FILE, OUTPUT_FILE, TARGETS_FILE)