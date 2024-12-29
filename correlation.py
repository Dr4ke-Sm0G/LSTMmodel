import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("signaux_indicateurs.csv")

# Vérifier les premières lignes des données
print("Aperçu des données :")
print(df.head())

# Supprimer la colonne 'timestamp' si elle existe
if 'timestamp' in df.columns:
    df = df.drop(columns=['timestamp'])

# Gestion des valeurs manquantes : remplir avec la moyenne des colonnes numériques
df.fillna(df.mean(), inplace=True)

# Calcul des corrélations
correlation_matrix = df.corr()

# Afficher les corrélations avec la colonne 'close'
correlation_with_close = correlation_matrix['close'].sort_values(ascending=False)
print("\nCorrélations avec la colonne 'close' :")
print(correlation_with_close)

# Visualiser les corrélations avec une heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True)
plt.title("Matrice de corrélation")
plt.show()
