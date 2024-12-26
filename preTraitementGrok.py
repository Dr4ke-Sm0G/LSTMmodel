import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_csv("signaux_indicateurs.csv")

# Vérifier et afficher les premières lignes des données
print("Aperçu des données avant prétraitement :")
print(df.head())

# Gestion des valeurs manquantes
# Exclure la colonne timestamp avant de remplir les NaN
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convertir en format datetime
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

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(df_normalized.drop(columns=['timestamp']), test_size=0.4, random_state=42)

# Création des séquences pour LSTM
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])  # Entrées (séquences)
        y.append(data[i+sequence_length])   # Sortie (prochaine bougie)
    return np.array(X), np.array(y)

# Définir la longueur des séquences
sequence_length = 10

# Créer les séquences pour l'entraînement
X_train, y_train = create_sequences(train_data.values, sequence_length)

# Créer les séquences pour le test
X_test, y_test = create_sequences(test_data.values, sequence_length)

# Vérification des dimensions
print(f"Forme des séquences d'entrée pour entraînement X : {X_train.shape}")
print(f"Forme des cibles de sortie pour entraînement y : {y_train.shape}")
print(f"Forme des séquences d'entrée pour test X : {X_test.shape}")
print(f"Forme des cibles de sortie pour test y : {y_test.shape}")

# Export des données
np.save("X_train_sequences.npy", X_train)
np.save("y_train_targets.npy", y_train)
np.save("X_test_sequences.npy", X_test)
np.save("y_test_targets.npy", y_test)

print("Exportation terminée : fichiers créés pour l'entraînement et le test.")