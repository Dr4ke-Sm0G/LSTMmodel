import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

def predict_and_denormalize(model_path, input_file, scaler_file, output_file):
    """
    Charge le modèle, effectue des prédictions, et dénormalise les résultats.
    
    Args:
    - model_path (str): Chemin vers le modèle LSTM sauvegardé.
    - input_file (str): Chemin vers le fichier numpy (.npy) contenant les séquences des 10 bougies.
    - scaler_file (str): Chemin vers le scaler sauvegardé pour dénormaliser les prédictions.
    - output_file (str): Chemin vers le fichier de sortie CSV contenant les prédictions dénormalisées.
    """
    # Charger le modèle
    model = load_model(model_path)
    print("Modèle chargé avec succès.")

    # Charger les séquences d'entrée
    x_sequences = np.load(input_file)
    print(f"Fichier de séquences chargé : {x_sequences.shape}")

    # Charger le scaler pour dénormalisation
    scaler = joblib.load(scaler_file)
    print("Scaler chargé avec succès.")

    # Effectuer les prédictions
    predicted_closes = model.predict(x_sequences)
    predicted_closes = predicted_closes.flatten()  # Aplatir les prédictions pour un format simple
    print("Prédictions effectuées.")

    # Dénormaliser les prédictions
    # Créez un tableau avec la même structure que les données normalisées d'origine
    dummy_data = np.zeros((len(predicted_closes), scaler.data_max_.shape[0]))
    dummy_data[:, 0] = predicted_closes  # Placez les prédictions dans la bonne colonne (assumons 'close' est à l'index 0)
    denormalized_closes = scaler.inverse_transform(dummy_data)[:, 0]  # Dénormalisez uniquement la colonne 'close'

    # Créer un DataFrame pour les prédictions dénormalisées
    predictions_df = pd.DataFrame({
        "sequence_index": range(len(denormalized_closes)),
        "predicted_close": denormalized_closes
    })

    # Sauvegarder les prédictions dans un fichier CSV
    predictions_df.to_csv(output_file, index=False)
    print(f"Prédictions sauvegardées dans le fichier : {output_file}")


if __name__ == "__main__":
    # Chemins des fichiers
    MODEL_PATH = "eth_usdt_lstm_model.h5"
    INPUT_FILE = "test_X_sequences.npy"
    SCALER_FILE = "scaler.pkl"
    OUTPUT_FILE = "predicted_closes_denormalized.csv"

    # Exécution du script
    predict_and_denormalize(MODEL_PATH, INPUT_FILE, SCALER_FILE, OUTPUT_FILE)
