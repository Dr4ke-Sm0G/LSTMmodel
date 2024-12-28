import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Charger les données
df = pd.read_csv("signaux_indicateurs.csv")

# Vérifier et afficher les premières lignes des données
print("Aperçu des données avant prétraitement :")
print(df.head())

# Gestion des valeurs manquantes
# Exclure la colonne timestamp avant de remplir les NaN
if 'timestamp' in df.columns:
    # Convertir en format datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    timestamps = df['timestamp']  # Sauvegarder les timestamps
    df = df.drop(columns=['timestamp'])  # Supprimer la colonne temporairement

# Remplir les valeurs manquantes avec la moyenne des colonnes numériques
df.fillna(df.mean(), inplace=True)

# Normalisation des données (MinMaxScaler)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
df_normalized = pd.DataFrame(scaled_data, columns=df.columns)

# Réintégrer les timestamps si nécessaires
df_normalized['timestamp'] = timestamps.values

# Création des séquences pour LSTM


def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])  # Entrées (séquences)
        y.append(data[i+sequence_length])   # Sortie (prochaine bougie)
    return np.array(X), np.array(y)


# Définir la longueur des séquences
sequence_length = 20

# Conversion en tableau numpy
# Exclure les timestamps pour créer des séquences
data_array = df_normalized.drop(columns=['timestamp']).values
X, y = create_sequences(data_array, sequence_length)

# Vérification des dimensions
print(f"Forme des séquences d'entrée X : {X.shape}")
print(f"Forme des cibles de sortie y : {y.shape}")

# Export des données
np.save("X_sequences.npy", X)
np.save("y_targets.npy", y)

print("Exportation terminée : fichiers X_sequences.npy et y_targets.npy créés.")
